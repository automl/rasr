import torch
import torch.nn as nn


class AxialDropout(nn.Module):
    """Structured dropout that shares one mask per row or per column."""

    def __init__(self, p: float, orientation: str):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        if orientation not in ("per_row", "per_column"):
            raise ValueError("orientation must be 'per_row' or 'per_column'")

        self.p = float(p)
        self.orientation = orientation
        self.keep_rate = 1.0 - self.p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x

        broadcast_dim = 1 if self.orientation == "per_row" else 2

        mask_shape = [1] * x.dim()
        mask_shape[broadcast_dim] = x.size(broadcast_dim)

        mask = (torch.rand(mask_shape, device=x.device, dtype=x.dtype) < self.keep_rate).div(
            self.keep_rate
        )
        return x * mask
