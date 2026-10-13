"""RASR mask utilities shared by the dataloader, model.sample(), and inference wrappers."""

from __future__ import annotations

import torch


def make_rasr_masks(
    key_is_valid: torch.Tensor,
    seq_len: int,
    trg_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute decoder self-attention (B, 1, L, L) and cross-attention (B, 1, L, R) masks."""
    device = trg_len.device

    query_valid = torch.arange(seq_len, device=device).unsqueeze(0) < trg_len.unsqueeze(1)

    causal = torch.triu(
        torch.ones(1, seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    ).eq(0)

    decoder_self_attn_mask = (query_valid.unsqueeze(-1) & causal).unsqueeze(1)

    decoder_cross_attn_mask = (
        query_valid.unsqueeze(-1) & key_is_valid.unsqueeze(1)
    ).unsqueeze(1)

    return decoder_self_attn_mask, decoder_cross_attn_mask
