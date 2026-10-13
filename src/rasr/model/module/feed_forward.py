import torch.nn as nn

from src.rasr.model.module.activation import Activation


class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, use_bias, activation, glu):
        super(FeedForward, self).__init__()

        self.is_glu = glu

        if glu:
            ff_dim_out = ff_dim // 2
        else:
            ff_dim_out = ff_dim

        self.in_linear = nn.Linear(model_dim, ff_dim, bias=use_bias)
        self.out_linear = nn.Linear(ff_dim_out, model_dim, bias=use_bias)
        self.act = Activation(activation, glu)

    def forward(self, x):

        x = self.act(self.in_linear(x))
        return self.out_linear(x)
