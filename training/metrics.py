"""Evaluation metrics for PRS model."""

import torch
import numpy as np


class MetricsAccumulator:
    """Accumulates predictions and targets across batches for epoch-level metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.preds.append(pred.detach().cpu())
        self.targets.append(target.detach().cpu())

    def compute(self) -> dict:
        preds = torch.cat(self.preds).numpy()
        targets = torch.cat(self.targets).numpy()

        mae = float(np.mean(np.abs(preds - targets)))

        # R²
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))

        # Pearson correlation
        if len(preds) > 1:
            pearson_r = float(np.corrcoef(preds, targets)[0, 1])
        else:
            pearson_r = 0.0

        return {
            "r2": r2,
            "pearson_r": pearson_r,
            "mae": mae,
            "n_samples": len(preds),
        }
