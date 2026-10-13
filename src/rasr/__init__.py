"""RASR (Reinforced Amortized Symbolic Regression) model for torchtitan.

Registers the "rasr" train spec; select it with `[model] name = "rasr"` in the TOML config.
"""

from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

from src.rasr.torchtitan_components.lr_scheduler_universal import build_lr_schedulers_auto
from src.rasr.torchtitan_components.metrics_with_parameter_logging import (
    build_metrics_processor_with_parameter_logging,
)

from .data.dataloader import build_rasr_dataloader
from .infra.parallelize import parallelize_rasr
from .loss import build_rasr_loss
from .model.args import RASRModelArgs
from .model.model import RASRTransformer
from .optimizer import build_rasr_optimizers
from .validator import build_rasr_validator

__all__ = [
    "RASRModelArgs",
    "RASRTransformer",
    "parallelize_rasr",
    "rasr_configs",
]

rasr_configs = {
    "debugmodel": RASRModelArgs(
        model_dim=64,
        n_layers_enc=1,
        n_layers_dec=1,
        num_head=2,
        max_len=64,
        ff_factor=2,
        rms_norm=True,
        add_unit_offset=True,
        use_bias=True,
        softmax_scale=True,
        initializer_range=0.02,
        max_var_pos=11,
        use_ssmax=False,
    ),
    "6M": RASRModelArgs(
        model_dim=256,
        n_layers_enc=3,
        n_layers_dec=3,
        num_head=4,
        max_len=256,
        ff_factor=4,
        softmax_scale=True,
        rms_norm=True,
        add_unit_offset=True,
        use_bias=True,
        learn_ln=True,
        initializer_range=0.02,
        max_var_pos=11,
        use_ssmax=True,
        use_attention_gate=True,
    ),
    "24M": RASRModelArgs(
        model_dim=384,
        n_layers_enc=5,
        n_layers_dec=5,
        num_head=6,
        max_len=256,
        ff_factor=4,
        softmax_scale=True,
        rms_norm=True,
        add_unit_offset=True,
        use_bias=True,
        learn_ln=True,
        initializer_range=0.02,
        max_var_pos=11,
        use_ssmax=True,
        use_attention_gate=True,
    ),
    "94M": RASRModelArgs(
        model_dim=512,
        n_layers_enc=11,
        n_layers_dec=11,
        num_head=8,
        max_len=256,
        ff_factor=4,
        softmax_scale=True,
        rms_norm=True,
        add_unit_offset=True,
        use_bias=True,
        learn_ln=True,
        initializer_range=0.02,
        max_var_pos=11,
        use_ssmax=True,
        use_attention_gate=True,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=RASRTransformer,
        model_args=rasr_configs,
        parallelize_fn=parallelize_rasr,
        pipelining_fn=None,  # Pipeline parallel not supported for RASR.
        build_optimizers_fn=build_rasr_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers_auto,
        build_dataloader_fn=build_rasr_dataloader,
        build_tokenizer_fn=None,  # Tokenizer is built inside build_rasr_dataloader.
        build_loss_fn=build_rasr_loss,
        build_validator_fn=build_rasr_validator,
        build_metrics_processor_fn=build_metrics_processor_with_parameter_logging,
    )


register_train_spec("rasr", get_train_spec())
