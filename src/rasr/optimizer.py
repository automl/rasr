from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import OptimizersContainer, build_optimizers
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from src.rasr.model.module.norm import RMSNorm


def _group_parameters_for_weight_decay(
    model: nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Split parameters into AdamW groups; biases, norms, embeddings and 1D params get no decay."""
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay: set[str] = set()
    no_decay: set[str] = set()

    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (
        nn.Embedding,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LazyBatchNorm1d,
        nn.LazyBatchNorm2d,
        nn.LazyBatchNorm3d,
        nn.GroupNorm,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.LocalResponseNorm,
        RMSNorm,
    )

    for mn, m in model.named_modules():
        # Use recurse=False to avoid visiting the same parameter repeatedly.
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if not p.requires_grad or fpn not in param_dict:
                continue
            if p.ndim < 2:
                no_decay.add(fpn)
            elif pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    decay |= (param_dict.keys() - no_decay)

    inter = decay & no_decay
    if inter:
        raise ValueError(f"Parameters in both decay/no_decay groups: {sorted(inter)}")

    if not no_decay or weight_decay == 0.0:
        return [
            {
                "params": [param_dict[pn] for pn in sorted(decay | no_decay)],
                "weight_decay": float(weight_decay),
            }
        ]

    return [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": float(weight_decay)},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]


class GroupedOptimizersContainer(OptimizersContainer):
    """Optimizers container that accepts per-model parameter groups."""

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[Optimizer],
        optimizer_kwargs: dict[str, object],
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts

        for model in self.model_parts:
            grouped_params = _group_parameters_for_weight_decay(
                model,
                weight_decay=float(optimizer_kwargs["weight_decay"]),
            )
            self.optimizers.append(optimizer_cls(grouped_params, **optimizer_kwargs))
            for group in grouped_params:
                all_params.extend(group["params"])  # type: ignore[arg-type]

        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)


def build_rasr_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Build AdamW optimizer with old RASR no-decay grouping."""
    if optimizer_config.early_step_in_backward:
        logger.warning("RASR grouped optimizer: falling back to default for early_step_in_backward=true.")
        return build_optimizers(
            model_parts=model_parts,
            optimizer_config=optimizer_config,
            parallel_dims=parallel_dims,
            ft_manager=ft_manager,
        )
    if ft_manager and ft_manager.enabled:
        logger.warning("RASR grouped optimizer: falling back to default when torchft is enabled.")
        return build_optimizers(
            model_parts=model_parts,
            optimizer_config=optimizer_config,
            parallel_dims=parallel_dims,
            ft_manager=ft_manager,
        )

    name = optimizer_config.name
    if name != "AdamW":
        raise NotImplementedError(f"RASR grouped optimizer currently supports AdamW only, got {name}.")

    optim_implementation = optimizer_config.implementation
    if optim_implementation not in ("fused", "foreach", "for-loop"):
        raise ValueError(f"Unsupported optimizer implementation: {optim_implementation}")

    optimizer_cls = torch.optim.AdamW
    optimizer_kwargs: dict[str, object] = {
        "lr": optimizer_config.lr,
        "betas": (optimizer_config.beta1, optimizer_config.beta2),
        "eps": optimizer_config.eps,
        "weight_decay": optimizer_config.weight_decay,
        "fused": optim_implementation == "fused",
        "foreach": optim_implementation == "foreach",
    }

    logger.info(
        "Building RASR grouped AdamW optimizer "
        "(bias+norm no_decay, weight_decay applied to decay group only)."
    )
    return GroupedOptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)
