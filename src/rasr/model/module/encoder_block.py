import torch
import torch.nn as nn

from src.rasr.model.module.axial_attention import AxialAttention
from src.rasr.model.module.feed_forward import FeedForward
from src.rasr.model.module.norm import RMSNorm


class EncoderBlock(nn.Module):


    def __init__(self, config):
        super().__init__()

        ff_dim = int(config.ff_factor * config.model_dim)

        self.residual_in_fp32 = True

        if config.rms_norm:
            self.norm1 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
            self.norm2 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
            self.norm3 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
        else:
            self.norm1 = nn.LayerNorm(
                config.model_dim, eps=config.ln_eps, bias=config.use_bias, elementwise_affine=config.learn_ln
            ).float()
            self.norm2 = nn.LayerNorm(
                config.model_dim, eps=config.ln_eps, bias=config.use_bias, elementwise_affine=config.learn_ln
            ).float()
            self.norm3 = nn.LayerNorm(
                config.model_dim, eps=config.ln_eps, bias=config.use_bias, elementwise_affine=config.learn_ln
            ).float()

        self.sample_attn = AxialAttention(
            config.model_dim,
            config.num_head,
            use_bias=config.use_bias,
            softmax_scale=config.softmax_scale,
            max_position_embeddings=config.max_len,
            orientation='per_row',
            use_attention_gate=config.use_attention_gate,
            use_ssmax=False,
        )

        self.feature_attn = AxialAttention(
            config.model_dim,
            config.num_head,
            use_bias=config.use_bias,
            softmax_scale=config.softmax_scale,
            max_position_embeddings=config.max_len,
            orientation='per_column',
            use_attention_gate=config.use_attention_gate,
            use_ssmax=config.use_ssmax,
        )

        self.transition = FeedForward(
            config.model_dim, ff_dim, use_bias=config.use_bias, activation=config.activation, glu=config.glu
        )

    def _residual_add(self, x, y):
        if self.residual_in_fp32 and x.dtype in (torch.float16, torch.bfloat16):
            return (x.float() + y.float()).to(x.dtype)
        else:
            return x + y

    def forward(self, tbl_states, tbl_mask=None):
        hidden_states = self._residual_add(tbl_states, self.sample_attn(self.norm1(tbl_states), tbl_mask=tbl_mask))
        hidden_states = self._residual_add(
            hidden_states, self.feature_attn(self.norm2(hidden_states), tbl_mask=tbl_mask)
        )
        hidden_states = self._residual_add(hidden_states, self.transition(self.norm3(hidden_states)))

        return hidden_states
