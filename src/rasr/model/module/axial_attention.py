import torch
import torch.nn as nn

from src.rasr.model.module.norm import RMSNorm


class AxialAttention(nn.Module):
    def __init__(self, embed_dim, num_head, use_bias, softmax_scale,
                 max_position_embeddings, orientation, use_attention_gate: bool = False,
                 use_ssmax: bool = False) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        assert orientation in ['per_row', 'per_column']
        self.orientation = orientation

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

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, tbl_act, tbl_mask):

        assert len(tbl_act.shape) == 4

        if self.orientation == 'per_column':
            tbl_act = torch.swapaxes(tbl_act, -2, -3).contiguous()
            if tbl_mask is not None:
                tbl_mask = torch.swapaxes(tbl_mask, -1, -2).contiguous()

        batch_size = tbl_act.shape[0]
        seqlenX = tbl_act.shape[1]
        seqlenY = tbl_act.shape[2]
        extended_batch_size = batch_size * seqlenX

        query, key, value = self.Wqkv(tbl_act).chunk(3, dim=-1)

        query = query.reshape(extended_batch_size, seqlenY, self.num_head, self.head_dim).transpose(1, 2).contiguous()
        key = key.reshape(extended_batch_size, seqlenY, self.num_head, self.head_dim).transpose(1, 2).contiguous()
        value = value.reshape(extended_batch_size, seqlenY, self.num_head, self.head_dim).transpose(1, 2).contiguous()
        query = self.q_norm(query)
        key = self.k_norm(key)

        if self.use_ssmax:
            n = key.size(-2)
            logn = torch.log(torch.tensor(float(n), device=query.device, dtype=query.dtype)).view(1, 1)
            base_scales = self.qassmax_base_mlp(logn).view(1, self.num_head, 1, self.head_dim)
            modulation = 1 + torch.tanh(self.qassmax_query_mlp(query))
            query = query * (base_scales * modulation)

        if tbl_mask is not None:
            tbl_mask = tbl_mask.unsqueeze(-2)
            tbl_mask = tbl_mask.reshape(extended_batch_size, 1, 1, seqlenY)

            # A fully-masked row would produce NaN from softmax; force its first position True.
            if tbl_mask.dtype == torch.bool and tbl_mask.shape[-1] > 0:
                all_masked = ~torch.any(tbl_mask, dim=-1, keepdim=True)
                safe_mask = tbl_mask.clone()
                safe_mask[..., :1] = safe_mask[..., :1] | all_masked
                tbl_mask = safe_mask

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=tbl_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale
        )

        if self.use_attention_gate:
            gate = torch.sigmoid(self.Wg(tbl_act))
            gate = gate.reshape(extended_batch_size, seqlenY, self.num_head)
            gate = gate.transpose(1, 2).unsqueeze(-1).contiguous()
            attn_out = attn_out * gate

        attn_out = attn_out.transpose(1, 2).contiguous().reshape(
            batch_size, seqlenX, seqlenY, self.num_head * self.head_dim
        )

        tbl_act = self.out_proj(attn_out)

        if self.orientation == 'per_column':
            tbl_act = torch.swapaxes(tbl_act, -2, -3).contiguous()

        return tbl_act