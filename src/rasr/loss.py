"""RASR loss function builder for torchtitan."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchtitan.tools.logging import logger

from src.rasr.constants import IGNORE_INDEX


def build_rasr_loss(job_config, **kwargs):
    """Build a cross-entropy loss over (B, L, V) logits that ignores PAD labels."""
    ignore_index = IGNORE_INDEX

    def ce_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            pred.flatten(0, 1).float(),
            labels.flatten(0, 1),
            ignore_index=ignore_index,
            reduction="mean",
        )

    if job_config.compile.enable and "loss" in job_config.compile.components:
        ce_loss = torch.compile(ce_loss, backend=job_config.compile.backend)
        logger.info("RASR loss: compiled cross-entropy")
    else:
        logger.info(f"RASR loss: cross-entropy (ignore_index={ignore_index}, reduction='mean')")

    return ce_loss
