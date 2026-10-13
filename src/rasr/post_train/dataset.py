from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset

from src.rasr.data.dataloader import (
    _open_arrow_readers,
    _read_sample,
    _scan_arrow_shards,
)


class RASRArrowMapDataset(Dataset):
    """Map-style Arrow dataset for TRL post-training."""

    def __init__(self, data_path: str, set_name: str, num_vars: int | None = None) -> None:
        (
            self._shard_paths,
            self._shard_batch_row_ends,
            self._shard_row_ends,
            self._total_rows,
        ) = _scan_arrow_shards(data_path, set_name)
        self._num_vars = num_vars
        self._reader_cache: dict[str, object] = {}
        self._worker_pid: int | None = None

    def __len__(self) -> int:
        return int(self._total_rows)

    def _ensure_readers(self) -> None:
        pid = os.getpid()
        if self._worker_pid == pid and "readers" in self._reader_cache:
            return
        self._worker_pid = pid
        self._reader_cache = {"readers": _open_arrow_readers(self._shard_paths), "batch": {}}

    def __getitem__(self, idx: int) -> dict[str, object]:
        self._ensure_readers()
        readers = self._reader_cache["readers"]
        batch_cache = self._reader_cache["batch"]
        tbl, _ = _read_sample(
            idx=idx,
            shard_readers=readers,
            shard_batch_row_ends=self._shard_batch_row_ends,
            shard_row_ends=self._shard_row_ends,
            cache=batch_cache,
        )

        if self._num_vars is not None:
            max_features = 1 + int(self._num_vars)
            tbl = tbl[:, :max_features]

        return {"tbl": tbl.to(dtype=torch.float32), "prompt": ""}
