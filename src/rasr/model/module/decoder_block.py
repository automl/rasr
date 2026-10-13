import torch
import torch.nn as nn

from src.rasr.model.module.attention import AttentionLayer
from src.rasr.model.module.feed_forward import FeedForward
from src.rasr.model.module.norm import RMSNorm


class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        ff_dim = int(config.ff_factor * config.model_dim)

        self.residual_in_fp32 = True

        if config.rms_norm:
            self.norm1 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
            self.norm2 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
            self.norm3 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
            self.norm4 = RMSNorm(config.model_dim, eps=config.ln_eps, add_unit_offset=config.add_unit_offset)
        else:
            self.norm1 = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias,
                                      elementwise_affine=config.learn_ln).float()
            self.norm2 = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias,
                                      elementwise_affine=config.learn_ln).float()
            self.norm3 = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias,
                                      elementwise_affine=config.learn_ln).float()
            self.norm4 = nn.LayerNorm(config.model_dim, eps=config.ln_eps, bias=config.use_bias,
                                      elementwise_affine=config.learn_ln).float()

        self.cross_attn = AttentionLayer(config.model_dim, config.num_head, is_causal=False, is_cross=True,
                                         use_bias=config.use_bias, softmax_scale=config.softmax_scale,
                                         max_position_embeddings=config.max_len, use_ssmax=config.use_ssmax,
                                         use_attention_gate=config.use_attention_gate)

        self.self_attn = AttentionLayer(config.model_dim, config.num_head, is_causal=True, is_cross=False,
                                        use_bias=config.use_bias, softmax_scale=config.softmax_scale,
                                        max_position_embeddings=config.max_len, use_ssmax=False,
                                        use_attention_gate=config.use_attention_gate)

        self.transition = FeedForward(config.model_dim, ff_dim, use_bias=config.use_bias, activation=config.activation,
                                      glu=config.glu)

    def _residual_add(self, x, y):
        if self.residual_in_fp32 and x.dtype in (torch.float16, torch.bfloat16):
            return (x.float() + y.float()).to(x.dtype)
        else:
            return x + y

    def forward(self, self_states, cross_states, self_attn_mask=None, cross_attn_mask=None):
        hidden_states = self._residual_add(
            self_states,
            self.cross_attn(self.norm1(cross_states), self.norm2(self_states), attn_mask=cross_attn_mask),
        )
        hidden_states = self._residual_add(
            hidden_states,
            self.self_attn(self.norm3(hidden_states), None, attn_mask=self_attn_mask),
        )
        hidden_states = self._residual_add(hidden_states, self.transition(self.norm4(hidden_states)))

        return hidden_states
