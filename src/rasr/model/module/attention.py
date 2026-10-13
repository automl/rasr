from typing import Optional

import torch
import torch.nn as nn

from src.rasr.model.module.norm import RMSNorm


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, end: int, theta: float = 10000.0, use_complex_rope: bool = False):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.end = end
        self.theta = theta
        self.use_complex_rope = use_complex_rope

        if self.use_complex_rope:
            self.register_buffer(
                "rope_cache_complex",
                self._precompute_freqs_cis(dim=self.dim, max_seq_len=self.end, theta=self.theta),
                persistent=False,
            )
            self.register_buffer(
                "rope_cache",
                torch.empty(0),
                persistent=False,
            )
        else:
            self.register_buffer(
                "rope_cache",
                self._precompute_rope_cache(dim=self.dim, max_seq_len=self.end, base=self.theta),
                persistent=False,
            )
            self.register_buffer(
                "rope_cache_complex",
                torch.empty(0, dtype=torch.complex64),
                persistent=False,
            )
    
    def _rebuild_rope_cache(self, device: torch.device) -> None:
        """Rebuild RoPE caches on demand for explicit non-trainable state reset."""
        if self.use_complex_rope:
            rebuilt = self._precompute_freqs_cis(
                dim=self.dim,
                max_seq_len=self.end,
                theta=self.theta,
            ).to(device=device)
            self.rope_cache_complex = rebuilt
            return

        rebuilt = self._precompute_rope_cache(
            dim=self.dim,
            max_seq_len=self.end,
            base=self.theta,
        ).to(device=device)
        self.rope_cache = rebuilt

    @staticmethod
    def _precompute_rope_cache(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)
        idx_theta = torch.outer(t, freqs).float()
        freqs = torch.cat([idx_theta, idx_theta], dim=-1)
        return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    @staticmethod
    def _precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(max_seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _reshape_for_broadcast(
        rope_cache: torch.Tensor,
        x: torch.Tensor,
        positions: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        bz, seqlen, _, head_dim = x.shape
        if positions is None:
            rope_cache = rope_cache[0:seqlen]
            return rope_cache.view(1, seqlen, 1, head_dim * 2)
        if positions.size(0) == 1:
            rope_cache = rope_cache[positions.squeeze(0)]
            return rope_cache.view(1, seqlen, 1, head_dim * 2)

        rope_cache_expanded = rope_cache[None, :, None, :].expand(bz, -1, -1, -1)
        return torch.gather(
            rope_cache_expanded,
            dim=1,
            index=positions.view(bz, seqlen, 1, 1).expand(bz, seqlen, 1, head_dim * 2),
        )

    @staticmethod
    def _reshape_freqs_cis(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        seqlen = x.shape[1]
        freqs_cis = freqs_cis[0:seqlen]
        shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def _apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        positions: Optional[torch.LongTensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = xq.shape[-1]
        rope_cache = self._reshape_for_broadcast(self.rope_cache.to(device=xq.device), xq, positions)
        cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
        sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)
        xq_out = (xq * cos) + (self._rotate_half(xq) * sin)
        xk_out = (xk * cos) + (self._rotate_half(xk) * sin)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def _apply_rotary_emb_complex(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        positions: Optional[torch.LongTensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions is not None:
            raise ValueError("Complex RoPE currently supports only contiguous positions (positions=None).")
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self._reshape_freqs_cis(self.rope_cache_complex.to(device=xq.device), xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(
            self,
            xq: torch.Tensor,  # [batch_size, seq_length, num_heads, d_qk]
            xk: torch.Tensor,  # [batch_size, seq_length, num_heads, d_qk]
            position_ids: Optional[torch.LongTensor],  # [batch_size, seq_length]
    ):
        _, seqlen, _, _ = xq.shape
        if seqlen > self.end:
            raise ValueError(f"Sequence length {seqlen} exceeds configured RoPE cache size {self.end}.")
        if self.use_complex_rope:
            return self._apply_rotary_emb_complex(xq, xk, position_ids)
        return self._apply_rotary_emb(xq, xk, position_ids)


class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_head, is_causal, is_cross, use_bias, softmax_scale,
                 max_position_embeddings, use_complex_rope: bool = False, use_ssmax: bool = False,
                 use_attention_gate: bool = False) -> None:

        super().__init__()
        self.embed_dim = embed_dim
        self.is_causal = is_causal
        self.is_cross = is_cross

        self.num_head = num_head
        assert self.embed_dim % num_head == 0, "self.kdim must be divisible by num_head"
        self.head_dim = self.embed_dim // num_head
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        if softmax_scale:
            self.scale = self.head_dim ** (-0.5)
        else:
            self.scale = None

        self.use_attention_gate = use_attention_gate
        if self.use_attention_gate:
            self.Wg = nn.Linear(embed_dim, num_head, bias=use_bias)

        self.use_ssmax = use_ssmax
        if self.use_ssmax:
            self.qassmax_base_mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.GELU(),
                nn.Linear(64, self.num_head * self.head_dim),
            )
            self.qassmax_query_mlp = nn.Sequential(
                nn.Linear(self.head_dim, 64),
                nn.GELU(),
                nn.Linear(64, self.head_dim),
            )
            nn.init.zeros_(self.qassmax_query_mlp[-1].weight)
            nn.init.zeros_(self.qassmax_query_mlp[-1].bias)

        if is_cross:
            self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=use_bias)
            self.Wq = nn.Linear(embed_dim, 1 * embed_dim, bias=use_bias)
        else:
            self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.use_rope = not is_cross
        self.rotary_embedding = None
        if self.use_rope:
            self.rotary_embedding = RotaryEmbedding(
                dim=embed_dim // num_head,
                end=max_position_embeddings,
                use_complex_rope=use_complex_rope,
            )

    def forward(self, keyvalue, query, attn_mask=None):
        if self.use_attention_gate:
            gate_input = query if self.is_cross else keyvalue

        if self.is_cross:
            key, value = self.Wkv(keyvalue).chunk(2, dim=-1)
            query = self.Wq(query)
        else:
            query, key, value = self.Wqkv(keyvalue).chunk(3, dim=-1)

        query = query.reshape(query.size(0), query.size(1), self.num_head, self.head_dim)
        key = key.reshape(key.size(0), key.size(1), self.num_head, self.head_dim)
        value = value.reshape(value.size(0), value.size(1), self.num_head, self.head_dim)

        query = self.q_norm(query)
        key = self.k_norm(key)
        if self.use_rope:
            query, key = self.rotary_embedding(query, key, position_ids=None)

        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        if self.use_ssmax:
            n = key.size(-2)
            logn = torch.log(torch.tensor(float(n), device=query.device, dtype=query.dtype)).view(1, 1)
            base_scales = self.qassmax_base_mlp(logn).view(1, self.num_head, 1, self.head_dim)
            modulation = 1 + torch.tanh(self.qassmax_query_mlp(query))
            query = query * (base_scales * modulation)

        if attn_mask is not None:
            # A fully-masked row would produce NaN from softmax; force its first position True.
            if attn_mask.dtype == torch.bool and attn_mask.shape[-1] > 0:
                all_masked = ~torch.any(attn_mask, dim=-1, keepdim=True)
                safe_mask = attn_mask.clone()
                safe_mask[..., :1] = safe_mask[..., :1] | all_masked
                attn_mask = safe_mask

            attn_out = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0,
                                                                         attn_mask=attn_mask, scale=self.scale)
        else:
            attn_out = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0,
                                                                         scale=self.scale, is_causal=self.is_causal)

        if self.use_attention_gate:
            gate = torch.sigmoid(self.Wg(gate_input))
            gate = gate.transpose(1, 2).unsqueeze(-1)
            attn_out = attn_out * gate

        attn_out = attn_out.transpose(1,2).reshape(query.size(0), query.size(2), -1)

        return self.out_proj(attn_out)
