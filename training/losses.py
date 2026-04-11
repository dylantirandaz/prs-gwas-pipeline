import torch
import torch.nn as nn


class SEWeightedMSELoss(nn.Module):
    def __init__(self, max_weight: float = 10.0, max_sample_loss: float = 10.0):
        super().__init__()
        self.max_weight = max_weight
        self.max_sample_loss = max_sample_loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                se: torch.Tensor) -> torch.Tensor:
        weights = (1.0 / (se ** 2 + 1e-8)).clamp(max=self.max_weight)
        weights = weights / (weights.mean() + 1e-8)
        sample_losses = (weights * (pred - target) ** 2).clamp(max=self.max_sample_loss)
        return sample_losses.mean()
