import torch
import torch.nn as nn


class SEWeightedMSELoss(nn.Module):
    def __init__(self, max_weight: float = 100.0):
        super().__init__()
        self.max_weight = max_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                se: torch.Tensor) -> torch.Tensor:
        # Inverse-variance weighting: variants with smaller SE are more precise
        weights = (1.0 / (se ** 2 + 1e-8)).clamp(max=self.max_weight)
        weights = weights / (weights.mean() + 1e-8)
        return (weights * (pred - target) ** 2).mean()
