from __future__ import annotations

from typing import Iterable


class EquationTokenizer:
    """Config-driven tokenizer for equations and expressions."""

    def __init__(
        self,
        special_tokens: list[str],
        operators: list[str],
        variables: list[str],
        digits: list[str],
        special_chars: list[str],
    ) -> None:
        special_tokens = list(special_tokens)
        operators = list(operators)
        variables = list(variables)
        digits = list(digits)
        special_chars = list(special_chars)
        self.all_tokens = special_tokens + operators + variables + digits + special_chars
        self.token_to_index = {token: idx for idx, token in enumerate(self.all_tokens)}
        self.index_to_token = {idx: token for token, idx in self.token_to_index.items()}
        self._constant_charset = set(digits + special_chars + ["-"])

    @classmethod
    def from_config(cls, tokenizer_cfg) -> "EquationTokenizer":
        """Build from ``job_config.tokenizer`` (TOML ``[tokenizer]``)."""
        return cls(
            list(tokenizer_cfg.special_tokens),
            list(tokenizer_cfg.operators),
            list(tokenizer_cfg.variables),
            list(tokenizer_cfg.digits),
            list(tokenizer_cfg.special_chars),
        )

    def _encode_constant(self, token: str) -> list[int]:
        encoded: list[int] = []
        for char in token:
            if char.isspace():
                continue
            if char in self._constant_charset and char in self.token_to_index:
                encoded.append(self.token_to_index[char])
                continue
            else:
                raise ValueError(f"Unsupported constant character '{char}' in token '{token}'.")
        return encoded

    def encode_tokens(self, expression_tokens: Iterable[str]) -> list[int]:
        encoded: list[int] = []
        for token in expression_tokens:
            if token in self.token_to_index:
                encoded.append(self.token_to_index[token])
                continue
            encoded.extend(self._encode_constant(token))
        return encoded

    def convert_ids_to_tokens(
        self,
        indices: Iterable[int],
        skip_special_tokens: bool = False,
        stop_at_eoe: bool = True,
    ) -> list[str]:
        indices_list = [int(idx) for idx in indices]
        if stop_at_eoe and "<EOE>" in self.token_to_index:
            eoe_index = self.token_to_index["<EOE>"]
            if eoe_index in indices_list:
                indices_list = indices_list[: indices_list.index(eoe_index)]

        skip = {"<PAD>", "<SOE>", "<EOE>"} if skip_special_tokens else set()
        tokens: list[str] = []
        for idx in indices_list:
            token = self.index_to_token.get(idx)
            if token is None or token in skip:
                continue
            tokens.append(token)
        return tokens

    def decode(
        self,
        indices: Iterable[int],
        stop_at_eoe: bool = True,
        skip_special_tokens: bool = False,
    ) -> str:
        return "".join(
            self.convert_ids_to_tokens(
                indices,
                skip_special_tokens=skip_special_tokens,
                stop_at_eoe=stop_at_eoe,
            )
        )

    def __len__(self) -> int:
        return len(self.all_tokens)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.token_to_index)

    def add_tokens(self, tokens: Iterable[str]) -> int:
        _ = tokens
        return 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_to_index[token]
