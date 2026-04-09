"""PRSNet: Embedding + Residual MLP for predicting variant effect sizes."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from models.layers import ResidualBlock


class PRSNet(nn.Module):
    """
    Architecture:
        15 numeric features + chr_embed(8) + a1_embed(4) + a2_embed(4) = 31 dims
        → BatchNorm1d(31)
        → ResidualBlock(31 → 256)
        → ResidualBlock(256 → 128)
        → ResidualBlock(128 → 64)
        → Linear(64 → 32) → ReLU → Linear(32 → 1)
    """

    def __init__(self):
        super().__init__()

        # Embeddings
        self.chr_embed = nn.Embedding(config.CHR_VOCAB, config.CHR_EMBED_DIM)
        self.a1_embed = nn.Embedding(config.ALLELE_VOCAB, config.ALLELE_EMBED_DIM)
        self.a2_embed = nn.Embedding(config.ALLELE_VOCAB, config.ALLELE_EMBED_DIM)

        # Input normalization
        self.input_bn = nn.BatchNorm1d(config.TOTAL_INPUT_DIM)

        # Residual blocks
        dims = [config.TOTAL_INPUT_DIM] + config.HIDDEN_DIMS  # [31, 256, 128, 64]
        self.res_blocks = nn.ModuleList([
            ResidualBlock(dims[i], dims[i + 1], dropout=config.DROPOUT)
            for i in range(len(dims) - 1)
        ])

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIMS[-1], config.HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(config.HEAD_HIDDEN, 1),
        )

    def forward(self, numeric: torch.Tensor, chr_idx: torch.Tensor,
                a1_idx: torch.Tensor, a2_idx: torch.Tensor) -> torch.Tensor:
        # Embeddings
        chr_e = self.chr_embed(chr_idx)
        a1_e = self.a1_embed(a1_idx)
        a2_e = self.a2_embed(a2_idx)

        # Concatenate all features
        x = torch.cat([numeric, chr_e, a1_e, a2_e], dim=-1)

        # Input batch norm
        x = self.input_bn(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Head
        out = self.head(x).squeeze(-1)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
