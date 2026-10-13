"""Parallelism for RASRTransformer, whose blocks live in model.encoder/model.decoder."""

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed._composable.replicate import replicate
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torchtitan.config import TORCH_DTYPE_MAP, JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

# Ops to save for selective activation checkpointing.
_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops.aten.max.default,
}


def _apply_ac_to_block(module: nn.Module, ac_config) -> nn.Module:
    """Wrap a single block with activation checkpointing."""
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=True)

    assert ac_config.mode == "selective"
    use_op_sac = ac_config.selective_ac_option == "op"
    use_layer_sac = ac_config.selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )

    if use_op_sac:
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                to_save = func in _save_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=True,
        )
    else:
        ac_freq = int(ac_config.selective_ac_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=True)
        else:
            return module


def _apply_rasr_ac(model: nn.Module, ac_config) -> None:
    """Apply activation checkpointing to RASR's encoder and decoder blocks."""
    for i, block in enumerate(model.encoder):
        model.encoder[i] = _apply_ac_to_block(block, ac_config)
    for i, block in enumerate(model.decoder):
        model.decoder[i] = _apply_ac_to_block(block, ac_config)
    logger.info(f"Applied {ac_config.mode} activation checkpointing to RASR encoder and decoder blocks")


def _apply_rasr_compile(model: nn.Module, compile_config) -> None:
    for i, block in enumerate(model.encoder):
        model.encoder[i] = torch.compile(block, backend=compile_config.backend, fullgraph=False, dynamic=True)
    for i, block in enumerate(model.decoder):
        model.decoder[i] = torch.compile(block, backend=compile_config.backend, fullgraph=False, dynamic=True)


def _apply_rasr_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
) -> None:
    """Apply FSDP2 to RASR encoder and decoder blocks, then the whole model."""
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    n_enc = len(model.encoder)
    n_dec = len(model.decoder)
    total_blocks = n_enc + n_dec

    def _reshard(idx: int) -> bool:
        if reshard_after_forward_policy == "always":
            return True
        elif reshard_after_forward_policy == "never":
            return False
        else:  # "default": don't reshard the last block.
            return idx < total_blocks - 1

    for i, block in enumerate(model.encoder):
        fully_shard(block, **fsdp_config, reshard_after_forward=_reshard(i))
    for i, block in enumerate(model.decoder):
        fully_shard(block, **fsdp_config, reshard_after_forward=_reshard(n_enc + i))

    fully_shard(model, **fsdp_config, reshard_after_forward=True)


def parallelize_rasr(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
) -> nn.Module:
    """Apply torch.compile, activation checkpointing and FSDP/DDP to RASRTransformer.

    Tensor parallelism is unsupported since AxialAttention lacks the standard wq/wk/wv/wo layout.
    """
    if parallel_dims.tp_enabled:
        logger.warning(
            "Tensor Parallelism is not supported for RASR model. "
            "Ignoring TP configuration and proceeding without TP."
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    # Compile must happen before AC.
    if model_compile_enabled:
        _apply_rasr_compile(model, job_config.compile)

    if job_config.activation_checkpoint.mode != "none":
        _apply_rasr_ac(model, job_config.activation_checkpoint)

    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
        else:
            dp_mesh = parallel_dims.get_mesh("fsdp")

        _apply_rasr_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to RASR model")
        else:
            logger.info("Applied FSDP to RASR model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to RASR model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to RASR model")

    elif parallel_dims.dp_replicate_enabled:
        world_mesh = parallel_dims.world_mesh
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        if model_compile_enabled:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"
        replicate(model, device_mesh=world_mesh, bucket_cap_mb=100)
        logger.info("Applied DDP to RASR model")

    return model