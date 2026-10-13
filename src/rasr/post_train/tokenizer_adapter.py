from __future__ import annotations

from typing import Iterable

import torch
from transformers.tokenization_utils import PreTrainedTokenizer

from src.rasr.data.tokenization import EquationTokenizer


class RASRProcessingClass(PreTrainedTokenizer):
    """Processing class for TRL's GRPOTrainer, which requires a ``PreTrainedTokenizerBase`` instance.

    ``__call__`` collates tabular inputs rather than tokenizing text.
    """

    def __init__(self, tokenizer: EquationTokenizer) -> None:
        # Must be set before super().__init__, which calls get_vocab.
        object.__setattr__(self, "tokenizer", tokenizer)
        super().__init__(padding_side="left")
        self.pad_token = "<PAD>"
        self.bos_token = "<SOE>"
        self.eos_token = "<EOE>"
        self.pad_token_id = tokenizer.token_to_index[self.pad_token]
        self.bos_token_id = tokenizer.token_to_index[self.bos_token]
        self.eos_token_id = tokenizer.token_to_index[self.eos_token]
        self.padding_side = "left"

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def get_vocab(self) -> dict[str, int]:
        return self.tokenizer.get_vocab()

    def __call__(self, samples, **_: object) -> dict[str, torch.Tensor]:
        if not isinstance(samples, list):
            raise TypeError("Expected a list of samples for RASRProcessingClass.")

        tbl_tensors = [sample["tbl"] for sample in samples]
        max_rows = max(tbl.shape[0] for tbl in tbl_tensors)
        max_features = max(tbl.shape[1] for tbl in tbl_tensors)

        batch = torch.full(
            (len(tbl_tensors), max_rows, max_features),
            float("-inf"),
            dtype=torch.float32,
        )
        for idx, tbl in enumerate(tbl_tensors):
            rows, features = tbl.shape
            batch[idx, :rows, :features] = tbl

        tbl_mask = batch != float("-inf")
        return {"tbl": batch, "tbl_mask": tbl_mask}

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False, **_: object) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(
            ids,
            stop_at_eoe=skip_special_tokens,
            skip_special_tokens=skip_special_tokens,
        )

    def batch_decode(self, ids_batch, skip_special_tokens: bool = False, **_: object) -> list[str]:
        if isinstance(ids_batch, torch.Tensor):
            ids_batch = ids_batch.tolist()
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in ids_batch]

    def convert_tokens_to_ids(self, token):
        if isinstance(token, str):
            return self.tokenizer.convert_tokens_to_ids(token)
        return [self.tokenizer.convert_tokens_to_ids(t) for t in token]

    def convert_ids_to_tokens(self, ids, skip_special_tokens: bool = False):
        if isinstance(ids, int):
            return self.tokenizer.index_to_token[ids]
        return self.tokenizer.convert_ids_to_tokens(
            ids,
            skip_special_tokens=skip_special_tokens,
        )

    def _tokenize(self, text, **_: object) -> list[str]:
        return [char for char in text if char in self.tokenizer.token_to_index]

    def _convert_token_to_id(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.tokenizer.index_to_token.get(index, self.pad_token)

    def convert_tokens_to_string(self, tokens: Iterable[str]) -> str:
        return "".join(token for token in tokens if token in self.tokenizer.token_to_index)

    def __len__(self) -> int:
        return len(self.tokenizer)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple:
        return ()

    def save_pretrained(self, save_directory, **_: object) -> tuple:
        return ()
