"""Taken from https://github.com/psaegert/flash-ansr/tree/main"""

import torch
from torch import nn


def float32_to_ieee754_bits(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    # Reinterpret bits as int32.
    i = x.view(torch.int32)
    bit_idx = torch.arange(31, -1, -1, device=x.device, dtype=torch.int32)
    bits = ((i.unsqueeze(-1) >> bit_idx) & 1).to(torch.int8)

    return bits


class IEEE75432PreEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape (..., input_size, 32), values in {-1, 1}.
        return (float32_to_ieee754_bits(x) - 0.5) * 2
