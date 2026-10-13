"""Arrow-shard dataloader for RASR training with torchtitan."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.tools.logging import logger

from src.rasr.constants import IGNORE_INDEX
from src.rasr.data.tokenization import EquationTokenizer
from src.rasr.masks import make_rasr_masks


def _scan_arrow_shards(data_path: str, set_name: str):
    """Scan Arrow shard metadata and return paths + row offsets."""
    data_dir = Path(data_path)
    pattern = f"{set_name}_dataset*.arrow"
    shard_paths = sorted(data_dir.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No files matched '{pattern}' in {data_dir}.")

    shard_path_strs = []
    shard_batch_row_ends = []
    shard_row_ends = []
    total_rows = 0

    for path in shard_paths:
        memory_map = pa.memory_map(path.as_posix())
        reader = pa.ipc.open_file(memory_map)
        batch_row_ends = []
        running_rows = 0
        for batch_idx in range(reader.num_record_batches):
            running_rows += reader.get_record_batch(batch_idx).num_rows
            batch_row_ends.append(running_rows)

        total_rows += running_rows
        shard_row_ends.append(total_rows)
        shard_path_strs.append(path.as_posix())
        shard_batch_row_ends.append(np.array(batch_row_ends, dtype=np.int64))
        logger.info(f"[RASR dataloader] Loaded {path.name:>40}  —  {running_rows} samples")

    return shard_path_strs, shard_batch_row_ends, np.array(shard_row_ends, dtype=np.int64), total_rows


def _open_arrow_readers(shard_paths):
    """Open Arrow file readers in the current process."""
    shard_readers = []
    for path in shard_paths:
        memory_map = pa.memory_map(path)
        shard_readers.append(pa.ipc.open_file(memory_map))
    return shard_readers


def _read_sample(
    idx: int,
    shard_readers,
    shard_batch_row_ends,
    shard_row_ends,
    cache: dict | None = None,
    extra_columns: list[str] | None = None,
):
    """Read a (dataset_tensor, token_tensor) sample by global index, plus a dict of *extra_columns* if given."""
    global_idx = int(idx)
    shard_idx = int(np.searchsorted(shard_row_ends, global_idx, side="right"))
    shard_start = 0 if shard_idx == 0 else int(shard_row_ends[shard_idx - 1])
    local_idx = global_idx - shard_start

    batch_row_ends = shard_batch_row_ends[shard_idx]
    batch_idx = int(np.searchsorted(batch_row_ends, local_idx, side="right"))
    batch_start = 0 if batch_idx == 0 else int(batch_row_ends[batch_idx - 1])
    row_idx = local_idx - batch_start

    cache_key = (shard_idx, batch_idx)
    if cache is not None and cache.get("key") == cache_key:
        dataset_column = cache["dataset_column"]
        token_column = cache["token_column"]
        record_batch = cache.get("record_batch")
    else:
        record_batch = shard_readers[shard_idx].get_record_batch(batch_idx)
        dataset_column = record_batch.column("dataset")
        try:
            token_column = record_batch.column("expression_tokens")
        except KeyError:
            token_column = None
        if cache is not None:
            cache["key"] = cache_key
            cache["dataset_column"] = dataset_column
            cache["token_column"] = token_column
            cache["record_batch"] = record_batch

    dataset_row = np.asarray(dataset_column[row_idx].as_py(), dtype=np.float32)
    token_row = (
        np.asarray(token_column[row_idx].as_py(), dtype=np.int64)
        if token_column is not None
        else np.empty(0, dtype=np.int64)
    )

    dataset_tensor = torch.from_numpy(dataset_row)
    token_tensor = torch.from_numpy(token_row)

    if extra_columns is None:
        return dataset_tensor, token_tensor

    extras: dict = {}
    for col in extra_columns:
        try:
            extras[col] = record_batch.column(col)[row_idx].as_py()
        except Exception:
            extras[col] = None
    return dataset_tensor, token_tensor, extras


class _RASRIterator:
    """Checkpointable iterator for RASRArrowIterableDataset; readers open lazily to stay fork-safe."""

    def __init__(self, worker_indices, shard_paths, shard_batch_row_ends, shard_row_ends, infinite):
        self._worker_indices = worker_indices
        self._shard_paths = shard_paths
        self._shard_batch_row_ends = shard_batch_row_ends
        self._shard_row_ends = shard_row_ends
        self._infinite = infinite
        self._pos = 0
        self._readers = None
        self._cache: dict = {}

    def __iter__(self):
        return self

    def _ensure_readers(self):
        if self._readers is None:
            self._readers = _open_arrow_readers(self._shard_paths)

    def __next__(self):
        n = len(self._worker_indices)
        if n == 0:
            raise StopIteration
        if not self._infinite and self._pos >= n:
            raise StopIteration
        self._ensure_readers()
        idx = self._worker_indices[self._pos % n]
        sample = _read_sample(idx, self._readers, self._shard_batch_row_ends, self._shard_row_ends, self._cache)
        self._pos += 1
        return sample

    def state_dict(self) -> dict:
        return {"samples_yielded": self._pos}

    def load_state_dict(self, state: dict) -> None:
        self._pos = state["samples_yielded"]


class RASRArrowIterableDataset(IterableDataset):
    """IterableDataset over RASR Arrow shards, strided across DP ranks."""

    def __init__(
        self,
        data_path: str,
        set_name: str,
        dp_rank: int,
        dp_world_size: int,
        infinite: bool = True,
    ):
        super().__init__()
        (
            self._shard_paths,
            self._shard_batch_row_ends,
            self._shard_row_ends,
            total_rows,
        ) = _scan_arrow_shards(data_path, set_name)
        # Finite validation needs equal work per rank to avoid collective deadlocks.
        indices = np.arange(dp_rank, total_rows, dp_world_size, dtype=np.int64)
        if not infinite:
            per_rank_count = total_rows // dp_world_size
            dropped_remainder = total_rows - (per_rank_count * dp_world_size)
            if dropped_remainder > 0 and dp_rank == 0:
                logger.info(
                    f"[RASR dataloader] validation drops {dropped_remainder} remainder samples "
                    "to keep rank lengths equal"
                )
            indices = indices[:per_rank_count]
        self._indices = indices
        self._infinite = infinite
        logger.info(
            f"[RASR dataloader] rank {dp_rank}/{dp_world_size}: "
            f"{len(self._indices)} samples (out of {total_rows} total)"
        )

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_indices = self._indices[worker_id::num_workers]
        return _RASRIterator(
            worker_indices=worker_indices,
            shard_paths=self._shard_paths,
            shard_batch_row_ends=self._shard_batch_row_ends,
            shard_row_ends=self._shard_row_ends,
            infinite=self._infinite,
        )


def rasr_collate_fn(
    samples,
    max_len: int,
    pad_index: int,
    soe_index: int,
    eoe_index: int,
    ignore_index: int = IGNORE_INDEX,
):
    """Collate a list of (dataset_tensor, token_tensor) into a torchtitan (input_dict, labels) batch."""
    datasets, token_tensors = zip(*samples)

    # Padding with -inf doubles as the source of tbl_mask.
    max_num_realizations = max(s.shape[0] for s in datasets)
    max_num_features = max(s.shape[1] for s in datasets)
    dataset_stack = torch.full(
        (len(datasets), max_num_realizations, max_num_features),
        float("-inf"),
        dtype=torch.float32,
    )
    for i, sample in enumerate(datasets):
        r, f = sample.shape
        dataset_stack[i, :r, :f] = sample

    max_token_len = max(min(int(tokens.shape[0]), max_len - 2) for tokens in token_tensors)
    seq_len = max_token_len + 1  # Room for SOE in inputs and EOE in labels.
    batch_size = len(token_tensors)

    input_seq_stack = torch.full((batch_size, seq_len), pad_index, dtype=torch.long)
    target_seq_stack = torch.full((batch_size, seq_len), ignore_index, dtype=torch.long)
    trg_lens = torch.empty(batch_size, dtype=torch.long)

    for i, tokens in enumerate(token_tensors):
        valid_len = min(int(tokens.shape[0]), max_len - 2)
        input_seq_stack[i, 0] = soe_index
        if valid_len > 0:
            sliced_tokens = tokens[:valid_len]
            input_seq_stack[i, 1 : 1 + valid_len] = sliced_tokens
            target_seq_stack[i, :valid_len] = sliced_tokens
        target_seq_stack[i, valid_len] = eoe_index
        trg_lens[i] = valid_len + 1

    tbl_mask = dataset_stack != float('-inf')
    key_is_valid = tbl_mask[:, :, 0]
    decoder_self_attn_mask, decoder_cross_attn_mask = make_rasr_masks(key_is_valid, seq_len, trg_lens)

    input_dict = {
        "input": dataset_stack,                                    # (B, R, F)
        "decoder_input_ids": input_seq_stack,                      # (B, L)
        "trg_len": trg_lens,                                        # (B,)
        "tbl_mask": tbl_mask,                                       # (B, R, F)
        "decoder_self_attn_mask": decoder_self_attn_mask,          # (B, 1, L, L)
        "decoder_cross_attn_mask": decoder_cross_attn_mask,        # (B, 1, L, R)
    }
    labels = target_seq_stack  # (B, L)

    return input_dict, labels


def build_rasr_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer,
    job_config,
) -> ParallelAwareDataloader:
    """Build the RASR dataloader and write vocab size and special token ids into job_config.tokenizer."""
    tokenizer_cfg = job_config.tokenizer
    dl_cfg = job_config.dataloader

    rasr_tokenizer = EquationTokenizer.from_config(tokenizer_cfg)

    tokenizer_cfg.trg_vocab_size = len(rasr_tokenizer.all_tokens)
    tokenizer_cfg.soe_token_id = rasr_tokenizer.token_to_index["<SOE>"]
    tokenizer_cfg.eoe_token_id = rasr_tokenizer.token_to_index["<EOE>"]
    tokenizer_cfg.pad_token_id = rasr_tokenizer.token_to_index["<PAD>"]

    logger.info(
        f"[RASR dataloader] vocab_size={tokenizer_cfg.trg_vocab_size}, "
        f"soe={tokenizer_cfg.soe_token_id}, eoe={tokenizer_cfg.eoe_token_id}, pad={tokenizer_cfg.pad_token_id}"
    )

    dataset = RASRArrowIterableDataset(
        data_path=dl_cfg.data_path,
        set_name=dl_cfg.set_name,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )

    collate_fn = partial(
        rasr_collate_fn,
        max_len=job_config.training.seq_len,
        pad_index=tokenizer_cfg.pad_token_id,
        soe_index=tokenizer_cfg.soe_token_id,
        eoe_index=tokenizer_cfg.eoe_token_id,
    )

    num_workers = int(job_config.dataloader.num_workers)

    return ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.local_batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        prefetch_factor=job_config.dataloader.prefetch_factor,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def build_rasr_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer,  # Unused; RASR builds its own tokenizer from the config.
    job_config,
) -> ParallelAwareDataloader:
    """Build the finite RASR validation dataloader from dataloader.val_set_name."""
    _ = tokenizer
    tokenizer_cfg = job_config.tokenizer
    dl_cfg = job_config.dataloader

    rasr_tokenizer = EquationTokenizer.from_config(tokenizer_cfg)
    tokenizer_cfg.trg_vocab_size = len(rasr_tokenizer.all_tokens)
    tokenizer_cfg.soe_token_id = rasr_tokenizer.token_to_index["<SOE>"]
    tokenizer_cfg.eoe_token_id = rasr_tokenizer.token_to_index["<EOE>"]
    tokenizer_cfg.pad_token_id = rasr_tokenizer.token_to_index["<PAD>"]

    dataset = RASRArrowIterableDataset(
        data_path=dl_cfg.data_path,
        set_name=dl_cfg.val_set_name,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=False,
    )

    val_local_batch_size = int(getattr(job_config.validation, "local_batch_size", -1))
    if val_local_batch_size <= 0:
        val_local_batch_size = int(job_config.training.local_batch_size)

    collate_fn = partial(
        rasr_collate_fn,
        max_len=job_config.training.seq_len,
        pad_index=tokenizer_cfg.pad_token_id,
        soe_index=tokenizer_cfg.soe_token_id,
        eoe_index=tokenizer_cfg.eoe_token_id,
    )
    num_workers = int(job_config.dataloader.num_workers)

    return ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=val_local_batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
        prefetch_factor=job_config.dataloader.prefetch_factor,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
