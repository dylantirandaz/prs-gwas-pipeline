import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from models.layers import ResidualBlock


class PRSNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.chr_embed = nn.Embedding(config.CHR_VOCAB, config.CHR_EMBED_DIM)
        self.a1_embed = nn.Embedding(config.ALLELE_VOCAB, config.ALLELE_EMBED_DIM)
        self.a2_embed = nn.Embedding(config.ALLELE_VOCAB, config.ALLELE_EMBED_DIM)

        self.input_norm = nn.LayerNorm(config.TOTAL_INPUT_DIM)

        dims = [config.TOTAL_INPUT_DIM] + config.HIDDEN_DIMS
        self.res_blocks = nn.ModuleList([
            ResidualBlock(dims[i], dims[i + 1], dropout=config.DROPOUT)
            for i in range(len(dims) - 1)
        ])

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIMS[-1], config.HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HEAD_HIDDEN, 1),
        )

    def forward(self, numeric: torch.Tensor, chr_idx: torch.Tensor,
                a1_idx: torch.Tensor, a2_idx: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            numeric,
            self.chr_embed(chr_idx),
            self.a1_embed(a1_idx),
            self.a2_embed(a2_idx),
        ], dim=-1)

        x = self.input_norm(x)
        for block in self.res_blocks:
            x = block(x)

        return self.head(x).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
