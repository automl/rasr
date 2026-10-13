import torch
import torch.nn as nn


class RMSNorm(torch.nn.Module):

    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x_dtype = x.dtype
        x_fp32 = self._norm(x.float())
        weight_fp32 = self.weight.float()
        if self.add_unit_offset:
            output = x_fp32 * (1 + weight_fp32)
        else:
            output = x_fp32 * weight_fp32
        return output.to(dtype=x_dtype)
