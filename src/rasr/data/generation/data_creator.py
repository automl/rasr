"""Offline generation of Arrow-format RASR training shards from the expression/dataset priors."""

from __future__ import annotations

import argparse
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
from src.rasr.data.generation.expression_wrapper import ExpressionWrapper
from src.rasr.data.tokenization import EquationTokenizer


def load_toml_config(path: str) -> dict:
    """Load a TOML config file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


@dataclass
class WorkerContext:
    expression_sampler: ExpressionSampler
    dataset_sampler: DatasetSampler
    expression_wrapper: ExpressionWrapper
    tokenizer: EquationTokenizer
    retries: dict
    filter_iqr: bool = True
    constant_strategy: str = "mantissa_exponent"


_WORKER_CONTEXT: WorkerContext | None = None


def _suppress_simplipy_runtime_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"simplipy\.operators",
    )


def _remap_expression_variables(expression: tuple[str, ...], variable_mapping: dict[str, str]) -> tuple[str, ...]:
    if not variable_mapping:
        return expression

    # Avoid replacement collisions by swapping through temporary placeholders first.
    placeholder_mapping = {source: f"__TMP_VAR_{int(source[1:])}__" for source in variable_mapping}
    remapped = [placeholder_mapping.get(token, token) for token in expression]
    inverse_placeholder_mapping = {
        placeholder: variable_mapping[source]
        for source, placeholder in placeholder_mapping.items()
    }
    remapped = [inverse_placeholder_mapping.get(token, token) for token in remapped]
    return tuple(remapped)


def _init_worker(config: dict, split_name: str, base_seed: int, task_id: int, filter_iqr: bool = True):
    global _WORKER_CONTEXT
    _suppress_simplipy_runtime_warnings()

    worker_seed = hash((base_seed, split_name, task_id, os.getpid())) & 0xFFFFFFFF
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    expression_sampler = ExpressionSampler.from_config(config)
    dataset_sampler = DatasetSampler.from_config(config)
    expression_wrapper = ExpressionWrapper(config["prior"]["expressions"]["operator_arity"])
    tok = config["tokenizer"]
    tokenizer = EquationTokenizer(
        tok["special_tokens"], tok["operators"], tok["variables"], tok["digits"], tok["special_chars"]
    )
    if len(tokenizer.all_tokens) > 256:
        raise ValueError("Tokenizer vocabulary must contain <= 256 tokens for uint8 serialization.")
    _WORKER_CONTEXT = WorkerContext(
        expression_sampler=expression_sampler,
        dataset_sampler=dataset_sampler,
        expression_wrapper=expression_wrapper,
        tokenizer=tokenizer,
        retries=config["data_generation"]["sampling_retries"],
        filter_iqr=filter_iqr,
        constant_strategy=str(tok.get("constant_strategy", "mantissa_exponent")),
    )


def _sample_data(num_samples_override: int | None = None):
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker not initialized correctly.")

    expr_sampler = _WORKER_CONTEXT.expression_sampler
    data_sampler = _WORKER_CONTEXT.dataset_sampler
    tokenizer = _WORKER_CONTEXT.tokenizer

    pair_retries = _WORKER_CONTEXT.retries["pairs"]
    expr_retries = _WORKER_CONTEXT.retries["expressions"]
    data_retries = _WORKER_CONTEXT.retries["datasets"]

    for _ in range(pair_retries + 1):
        try:
            skeleton, variables = expr_sampler.sample_skeleton_only_with_retries(
                retries=expr_retries
            )

            result = data_sampler.sample_with_retries(
                skeleton=skeleton,
                variables=variables,
                num_samples=num_samples_override,
                retries=data_retries,
                filter_iqr=_WORKER_CONTEXT.filter_iqr,
            )
            if result is None:
                warnings.warn("Dataset sampling returned None; retrying.", RuntimeWarning)
                continue

            x_data, y_hat, variable_mapping, noise_scale, raw_constants, mu_y, sigma_y = result

            sig_figs = expr_sampler.sig_figs
            expression, executable_expression = expr_sampler.build_expression_from_skeleton(
                skeleton, raw_constants, raw_constants, variables
            )

            wrapped_expression = _WORKER_CONTEXT.expression_wrapper.wrap(expression, mu_y, sigma_y)
            wrapped_expression = _WORKER_CONTEXT.expression_wrapper.round_constants(
                wrapped_expression, sig_figs, strategy=_WORKER_CONTEXT.constant_strategy
            )

            if len(wrapped_expression) > expr_sampler.max_length:
                warnings.warn("Wrapped expression too long; retrying.", RuntimeWarning)
                continue

            remapped_wrapped = _remap_expression_variables(wrapped_expression, variable_mapping)
            remapped_executable = _remap_expression_variables(executable_expression, variable_mapping)
            token_ids = tokenizer.encode_tokens(remapped_wrapped)
            if not token_ids:
                warnings.warn("Tokenization produced empty sequence; retrying.", RuntimeWarning)
                continue

            return x_data, y_hat, np.array(token_ids, dtype=np.uint8), float(noise_scale), " ".join(remapped_executable)
        except Exception:
            continue

    return None


class DatasetCreator:
    def __init__(
        self,
        config: dict,
        *,
        force_creation: bool = True,
        task_id: int = 0,
        num_tasks: int = 1,
        store_noise_scale: bool = False,
        store_expression: bool = False,
        filter_iqr: bool = True,
    ):
        self.config = config
        self.force_creation = force_creation
        self.task_id = task_id
        self.num_tasks = num_tasks
        self.store_noise_scale = store_noise_scale
        self.store_expression = store_expression
        self.filter_iqr = filter_iqr

        data_gen_cfg = self.config["data_generation"]
        self.num_workers = int(data_gen_cfg["num_workers"])
        self.training_dataset_size = int(data_gen_cfg["training_dataset_size"])
        self.validation_dataset_size = int(data_gen_cfg["validation_dataset_size"])
        self.base_seed = int(data_gen_cfg["seed"])
        self.data_dir = pathlib.Path(data_gen_cfg["output_path"])
        self.write_chunk_size = int(data_gen_cfg["arrow_write_batch_size"])
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _flush_buffer(writer, dataset_rows, token_rows, noise_scale_rows=None, expression_rows=None):
        if not dataset_rows:
            return
        arrays = [
            pa.array(dataset_rows, type=pa.list_(pa.list_(pa.float32()))),
            pa.array(token_rows, type=pa.list_(pa.uint8())),
        ]
        names = ["dataset", "expression_tokens"]
        if noise_scale_rows is not None:
            arrays.append(pa.array(noise_scale_rows, type=pa.float32()))
            names.append("noise_scale")
        if expression_rows is not None:
            arrays.append(pa.array(expression_rows, type=pa.string()))
            names.append("expression")
        record_batch = pa.RecordBatch.from_arrays(arrays, names=names)
        writer.write_batch(record_batch)
        dataset_rows.clear()
        token_rows.clear()
        if noise_scale_rows is not None:
            noise_scale_rows.clear()
        if expression_rows is not None:
            expression_rows.clear()

    def _get_output_path(self, split_name: str) -> pathlib.Path:
        if self.num_tasks > 1:
            file_name = f"{split_name}_dataset_task{self.task_id}.arrow"
        else:
            file_name = f"{split_name}_dataset.arrow"
        return self.data_dir / file_name

    def _generate_split(self, split_name: str, target_size: int) -> pathlib.Path:
        if target_size <= 0:
            raise ValueError(f"Invalid target size {target_size} for split '{split_name}'.")

        shard_target = int(np.ceil(target_size / max(self.num_tasks, 1)))
        output_path = self._get_output_path(split_name)
        if output_path.exists() and not self.force_creation:
            print(
                f"Skipping '{split_name}' dataset creation: output already exists at {output_path} "
                "(use --force_creation to overwrite)."
            )
            return output_path

        fields = [
            ("dataset", pa.list_(pa.list_(pa.float32()))),
            ("expression_tokens", pa.list_(pa.uint8())),
        ]
        if self.store_noise_scale:
            fields.append(("noise_scale", pa.float32()))
        if self.store_expression:
            fields.append(("expression", pa.string()))
        schema = pa.schema(fields)

        sink = pa.OSFile(output_path.as_posix(), "wb")
        writer = pa.ipc.new_file(sink, schema)

        futures = deque()
        window_size = max(1, self.num_workers * 2)
        written_samples = 0
        dataset_rows_buffer: list[list[list[float]]] = []
        token_rows_buffer: list[list[int]] = []
        noise_scale_rows_buffer: list[float] | None = [] if self.store_noise_scale else None
        expression_rows_buffer: list[str] | None = [] if self.store_expression else None

        with ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_init_worker,
            initargs=(self.config, split_name, self.base_seed, self.task_id, self.filter_iqr),
        ) as executor:
            progress = tqdm(total=shard_target, desc=f"Generating {split_name}")
            while written_samples < shard_target:
                while len(futures) < window_size and written_samples + len(futures) < shard_target:
                    futures.append(executor.submit(_sample_data, None))

                result = futures.popleft().result()
                if result is None:
                    continue
                x_data, y_data, token_tensor, noise_scale, executable_expression = result

                if np.isnan(x_data).any() or np.isinf(x_data).any() or np.isnan(y_data).any() or np.isinf(y_data).any():
                    continue

                dataset_np = np.concatenate([y_data.reshape(-1, 1), x_data], axis=1).astype(np.float32)
                dataset_rows_buffer.append([list(row) for row in dataset_np])
                token_rows_buffer.append(token_tensor.tolist())
                if noise_scale_rows_buffer is not None:
                    noise_scale_rows_buffer.append(noise_scale)
                if expression_rows_buffer is not None:
                    expression_rows_buffer.append(executable_expression)
                written_samples += 1
                progress.update(1)
                if len(dataset_rows_buffer) >= self.write_chunk_size:
                    self._flush_buffer(
                        writer, dataset_rows_buffer, token_rows_buffer, noise_scale_rows_buffer, expression_rows_buffer
                    )

            progress.close()
            self._flush_buffer(
                writer, dataset_rows_buffer, token_rows_buffer, noise_scale_rows_buffer, expression_rows_buffer
            )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic symbolic regression datasets.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to a TOML data generation config file.",
    )
    parser.add_argument(
        "-f",
        "--force_creation",
        default=True,
        action="store_true",
        help="Force dataset creation if file exists.",
    )
    parser.add_argument("--task_id", type=int, default=0, help="Task ID for multi-node execution.")
    parser.add_argument("--num_tasks", type=int, default=1, help="Total number of tasks/nodes.")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val"],
        help="Generate only one split. Omit to generate both.",
    )
    parser.add_argument(
        "--num_samples_extra",
        type=int,
        default=0,
        help="Add this many extra datapoints to both min and max of the num_samples distribution.",
    )
    parser.add_argument(
        "--store_noise_scale",
        action="store_true",
        default=False,
        help="Store the sampled noise scale per dataset as a 'noise_scale' column in the Arrow file.",
    )
    parser.add_argument(
        "--store_expression",
        action="store_true",
        default=False,
        help=(
            "Store the ground-truth prefix expression (with exact unrounded constants) "
            "as an 'expression' column in the Arrow file."
        ),
    )
    parser.add_argument(
        "--filter_iqr",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Drop y outliers (IQR k=5) from each dataset during sampling. Samples tolerance "
            "is increased by 0.05 to compensate. Overrides the config's "
            "data_generation.filter_iqr when given; otherwise that config value is used "
            "(defaulting to true if absent)."
        ),
    )

    args = parser.parse_args()
    config = load_toml_config(args.config)

    if args.num_samples_extra != 0:
        ns_cfg = config["prior"]["datasets"]["num_samples"]
        ns_cfg["min"] = int(ns_cfg["min"]) + args.num_samples_extra
        ns_cfg["max"] = int(ns_cfg["max"]) + args.num_samples_extra

    filter_iqr = args.filter_iqr
    if filter_iqr is None:
        filter_iqr = bool(config["data_generation"].get("filter_iqr", True))

    creator = DatasetCreator(
        config=config,
        force_creation=args.force_creation,
        task_id=args.task_id,
        num_tasks=args.num_tasks,
        store_noise_scale=args.store_noise_scale,
        store_expression=args.store_expression,
        filter_iqr=filter_iqr,
    )
    creator.create_datasets(split=args.split)
