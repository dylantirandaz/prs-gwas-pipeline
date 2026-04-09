"""Residual block for PRSNet."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Linear → BatchNorm → GELU + skip connection → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Skip connection: project if dims differ
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.linear(x)
        out = self.bn(out)
        out = self.act(out)
        out = out + residual
        out = self.dropout(out)
        return out
