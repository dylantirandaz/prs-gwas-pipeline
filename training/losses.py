"""SE-weighted MSE loss for PRS training."""

import torch
import torch.nn as nn


class SEWeightedMSELoss(nn.Module):
    """MSE loss weighted by inverse SE squared (precision weighting).

    weight_i = 1 / SE_i^2, clamped and normalized to mean=1.
    Variants with smaller standard errors get more weight.
    """

    def __init__(self, max_weight: float = 100.0):
        super().__init__()
        self.max_weight = max_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                se: torch.Tensor) -> torch.Tensor:
        # Inverse-variance weights
        weights = 1.0 / (se ** 2 + 1e-8)
        weights = weights.clamp(max=self.max_weight)
        # Normalize so mean weight = 1
        weights = weights / (weights.mean() + 1e-8)

        loss = weights * (pred - target) ** 2
        return loss.mean()
