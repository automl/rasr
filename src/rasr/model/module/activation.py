import torch
from torch import nn


class Activation(nn.Module):
    def __init__(self, act_fn_name: str, is_glu: bool):
        super().__init__()

        self.is_glu = is_glu

        if "silu" in act_fn_name:
            self.act = nn.SiLU()
        elif "relu" in act_fn_name:
            self.act = nn.ReLU()
        elif "gelu" in act_fn_name:
            self.act = nn.GELU(approximate='tanh')

    def forward(self, x: torch.Tensor):

        if self.is_glu:
            gate_states, up_states = x.chunk(2, dim=-1)
            return self.act(gate_states) * up_states
        else:
            return self.act(x)
