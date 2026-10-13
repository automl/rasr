from __future__ import annotations

import argparse
import itertools
import math
import os
import pathlib
import random
import tomllib
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm

from src.rasr.data.generation.dataset_sampler import DatasetSampler
from src.rasr.data.generation.expression_sampler import ExpressionSampler
from src.rasr.data.generation.tabicl_v2_prior._dataset import GraphPrior
from src.rasr.data.generation.tabicl_v2_prior.graph_lib._config import PriorConfig
from src.rasr.post_train.constants import N_SELECTION_ROWS, N_VALIDATION_ROWS


def load_toml_config(path: str) -> dict:
    """Load a TOML config file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _suppress_simplipy_runtime_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"simplipy\.operators",
    )


@dataclass
class WorkerContext:
    graph_prior: GraphPrior | None
    expression_sampler: ExpressionSampler | None
    dataset_sampler: DatasetSampler | None
    retries: dict | None
    filter_iqr: bool = True


_WORKER_CONTEXT: WorkerContext | None = None


def _init_worker(
    pretraining_config: dict | None,
    graph_prior_kwargs: dict | None,
    base_seed: int,
    split_name: str,
    task_id: int,
) -> None:
    global _WORKER_CONTEXT
    _suppress_simplipy_runtime_warnings()

    worker_seed = hash((base_seed, task_id, split_name, os.getpid())) & 0xFFFFFFFF
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.set_num_threads(1)

    graph_prior = GraphPrior(**graph_prior_kwargs) if graph_prior_kwargs is not None else None

    expression_sampler = None
    dataset_sampler = None
    retries = None
    filter_iqr = True
    if pretraining_config is not None:
        expression_sampler = ExpressionSampler.from_config(pretraining_config)
        dataset_sampler = DatasetSampler.from_config(pretraining_config)
        retries = pretraining_config["data_generation"]["sampling_retries"]
        filter_iqr = bool(pretraining_config["data_generation"].get("filter_iqr", True))

    _WORKER_CONTEXT = WorkerContext(
        graph_prior=graph_prior,
        expression_sampler=expression_sampler,
        dataset_sampler=dataset_sampler,
        retries=retries,
        filter_iqr=filter_iqr,
    )


def _sample_tabicl() -> np.ndarray | None:
    X, y, d, _, _ = _WORKER_CONTEXT.graph_prior.get_batch()
    X_np = X[0].numpy()
    y_np = y[0].numpy()
    di = int(d[0].item())
    return np.concatenate([y_np.reshape(-1, 1), X_np[:, :di]], axis=1).astype(np.float32)


def _sample_pretraining() -> np.ndarray | None:
    expr_sampler = _WORKER_CONTEXT.expression_sampler
    data_sampler = _WORKER_CONTEXT.dataset_sampler
    retries = _WORKER_CONTEXT.retries

    for _ in range(retries["pairs"] + 1):
        try:
            skeleton, variables = expr_sampler.sample_skeleton_only_with_retries(
                retries=retries["expressions"]
            )
            result = data_sampler.sample_with_retries(
                skeleton=skeleton,
                variables=variables,
                num_samples=None,
                retries=retries["datasets"],
                filter_iqr=True,
            )
            if result is None:
                continue
            x_data, y_hat = result[0], result[1]
            if np.isnan(x_data).any() or np.isinf(x_data).any() or np.isnan(y_hat).any() or np.isinf(y_hat).any():
                continue
            return np.concatenate([y_hat.reshape(-1, 1), x_data], axis=1).astype(np.float32)
        except Exception:
            continue
    return None


def _sample_data(source: str) -> np.ndarray | None:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker not initialized correctly.")
    if source == "pretrain":
        return _sample_pretraining()
    return _sample_tabicl()


class PostTrainDatasetCreator:
    """Generate Arrow post-training datasets mixing the pretraining and tabicl_v2 GraphSCM priors."""

    def __init__(self, config: dict, *, task_id: int = 0, num_tasks: int = 1):
        self.task_id = task_id
        self.num_tasks = num_tasks

        dg = config["data_generation"]
        self.output_path = pathlib.Path(dg["output_path"])
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.seed = int(dg["seed"])
        self.num_workers = int(dg["num_workers"])
        self.training_dataset_size = int(dg["training_dataset_size"])
        self.validation_dataset_size = int(dg["validation_dataset_size"])
        self.write_chunk_size = int(dg["arrow_write_batch_size"])

        self.pretraining_fraction = float(dg.get("pretraining_fraction", 0.0))
        if not 0.0 <= self.pretraining_fraction <= 1.0:
            raise ValueError("data_generation.pretraining_fraction must be within [0, 1].")

        pretraining_prior_config_path = dg.get("pretraining_prior_config")
        self.pretraining_config: dict | None = None
        if self.pretraining_fraction > 0.0:
            if not pretraining_prior_config_path:
                raise ValueError(
                    "data_generation.pretraining_prior_config must be set when "
                    "pretraining_fraction > 0."
                )
            self.pretraining_config = load_toml_config(pretraining_prior_config_path)
            ns_cfg = self.pretraining_config["prior"]["datasets"]["num_samples"]
            ns_cfg["min"] = int(ns_cfg["min"]) + N_SELECTION_ROWS + N_VALIDATION_ROWS
            ns_cfg["max"] = int(ns_cfg["max"]) + N_SELECTION_ROWS + N_VALIDATION_ROWS

        self.graph_prior_kwargs: dict | None = None
        if self.pretraining_fraction < 1.0:
            pr = config["prior"]
            self.graph_prior_kwargs = dict(
                regression=True,
                numerical_only=True,
                batch_size=1,
                batch_size_per_gp=1,
                min_features=int(pr["min_features"]),
                max_features=int(pr["max_features"]),
                test_size=N_SELECTION_ROWS + N_VALIDATION_ROWS,
                min_train_size=int(pr["min_train_size"]),
                max_train_size=int(pr["max_train_size"]),
                config=PriorConfig(
                    filter_unpredictable_graphs=True,
                    filter_unpredictable_datasets=True,
                ),
                n_jobs=1,
                device="cpu",
            )

    def _get_output_path(self, split_name: str) -> pathlib.Path:
        if self.num_tasks > 1:
            fname = f"{split_name}_dataset_task{self.task_id}.arrow"
        else:
            fname = f"{split_name}_dataset.arrow"
        return self.output_path / fname

    @staticmethod
    def _flush_buffer(writer: pa.ipc.RecordBatchFileWriter, buffer: list) -> None:
        if not buffer:
            return
        writer.write_batch(pa.RecordBatch.from_arrays(
            [pa.array(buffer, type=pa.list_(pa.list_(pa.float32())))],
            names=["dataset"],
        ))
        buffer.clear()

    def _make_source_schedule(self, shard_target: int, split_name: str) -> list[str]:
        """Return a reproducibly shuffled list of per-sample prior labels matching ``pretraining_fraction``."""
        rng = np.random.default_rng((self.seed, self.task_id, hash(split_name) & 0xFFFF))
        n_pretrain = int(round(shard_target * self.pretraining_fraction))
        n_pretrain = min(max(n_pretrain, 0), shard_target)
        schedule = ["pretrain"] * n_pretrain + ["tabicl"] * (shard_target - n_pretrain)
        rng.shuffle(schedule)
        return schedule

    def _generate_split(self, split_name: str, target_size: int) -> pathlib.Path:
        shard_target = math.ceil(target_size / self.num_tasks)
        source_schedule = self._make_source_schedule(shard_target, split_name)
        output_path = self._get_output_path(split_name)
        schema = pa.schema([("dataset", pa.list_(pa.list_(pa.float32())))])
        sink = pa.OSFile(output_path.as_posix(), "wb")
        writer = pa.ipc.new_file(sink, schema)
        buffer: list = []
        written = 0

        def _process_result(dataset_np: np.ndarray | None) -> bool:
            nonlocal written
            if dataset_np is None or np.isnan(dataset_np).any() or np.isinf(dataset_np).any():
                return False
            buffer.append([row.tolist() for row in dataset_np])
            written += 1
            if len(buffer) >= self.write_chunk_size:
                self._flush_buffer(writer, buffer)
            return True

        init_args = (self.pretraining_config, self.graph_prior_kwargs, self.seed, split_name, self.task_id)
        schedule_iter = iter(source_schedule)

        with tqdm(total=shard_target, desc=f"Generating {split_name} (task {self.task_id})") as progress:
            if self.num_workers == 0:
                _init_worker(*init_args)
                for label in schedule_iter:
                    while not _process_result(_sample_data(label)):
                        pass
                    progress.update(1)
            else:
                with ProcessPoolExecutor(
                    max_workers=self.num_workers,
                    initializer=_init_worker,
                    initargs=init_args,
                ) as executor:
                    window_size = self.num_workers * 2
                    futures: deque = deque()

                    def _submit(label: str) -> None:
                        futures.append((executor.submit(_sample_data, label), label))

                    for label in itertools.islice(schedule_iter, window_size):
                        _submit(label)

                    while written < shard_target:
                        future, label = futures.popleft()
                        if _process_result(future.result()):
                            progress.update(1)
                            next_label = next(schedule_iter, None)
                            if next_label is not None:
                                _submit(next_label)
                        else:
                            _submit(label)  # Retry with the same source to preserve the mix.

        self._flush_buffer(writer, buffer)
        writer.close()
        sink.close()
        return output_path

    def create_datasets(self, split: str | None = None) -> dict[str, pathlib.Path]:
        if split not in {None, "train", "val"}:
            raise ValueError(f"Unknown split option '{split}'. Use None, 'train' or 'val'.")

        outputs = {}
        if split in {None, "train"}:
            outputs["train"] = self._generate_split("train", self.training_dataset_size)
        if split in {None, "val"}:
            outputs["val"] = self._generate_split("val", self.validation_dataset_size)
        return outputs

    def run(self) -> None:
        outputs = self.create_datasets()
        names = ", ".join(path.name for path in outputs.values())
        print(f"Done. task {self.task_id}: {names}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate post-training regression datasets, mixing the pretraining "
        "prior and the tabicl_v2 GraphSCM prior."
    )
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val"],
        help="Generate only one split. Omit to generate both.",
    )
    args = parser.parse_args()

    config = load_toml_config(args.config)
    creator = PostTrainDatasetCreator(config, task_id=args.task_id, num_tasks=args.num_tasks)
    creator.create_datasets(split=args.split)


if __name__ == "__main__":
    main()
