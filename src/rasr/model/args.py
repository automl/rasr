from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torchtitan.protocols.model import BaseModelArgs


@dataclass
class RASRModelArgs(BaseModelArgs):
    """Model arguments for RASRTransformer; all fields need defaults (BaseModelArgs)."""

    model_dim: int = 256
    n_layers_enc: int = 3
    n_layers_dec: int = 3
    trg_vocab_size: int = 128  # Overwritten from the tokenizer in update_from_config.
    num_head: int = 4
    max_len: int = 256
    ff_factor: int = 4
    activation: str = "silu"
    glu: bool = False
    softmax_scale: bool = True
    pos_embedding: bool = True
    rel_pos_enc: bool = False
    initializer_range: float = 0.02
    rms_norm: bool = False
    resi_dropout: float = 0.1
    attn_dropout: float = 0.1
    ln_eps: float = 1e-5
    add_unit_offset: bool = False
    use_bias: bool = True
    learn_ln: bool = True
    last_zero: bool = False
    max_var_pos: int = 11
    use_ssmax: bool = False
    use_attention_gate: bool = False
    # Filled from job_config.tokenizer once the dataloader has run.
    soe_token_id: int = -1
    eoe_token_id: int = -1

    def update_from_config(self, job_config, **kwargs) -> None:
        """Update args from JobConfig (called after the dataloader is built)."""
        self.max_len = job_config.training.seq_len
        tokenizer_cfg = job_config.tokenizer
        if tokenizer_cfg.trg_vocab_size > 0:
            self.trg_vocab_size = tokenizer_cfg.trg_vocab_size
        if tokenizer_cfg.soe_token_id >= 0:
            self.soe_token_id = tokenizer_cfg.soe_token_id
            self.eoe_token_id = tokenizer_cfg.eoe_token_id

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        """Estimate total parameters and FLOPs per step."""
        nparams = sum(p.numel() for p in model.parameters())
        # Rough estimates: ~6 * d^2 per token per decoder layer, ~4 * d^2 per encoder layer.
        dec_flops = self.n_layers_dec * 6 * self.model_dim ** 2 * seq_len
        enc_flops = self.n_layers_enc * 4 * self.model_dim ** 2 * seq_len
        return nparams, enc_flops + dec_flops
