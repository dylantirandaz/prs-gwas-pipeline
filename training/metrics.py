import torch
import numpy as np


class MetricsAccumulator:
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

        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)

        return {
            "r2": float(1 - ss_res / (ss_tot + 1e-8)),
            "pearson_r": float(np.corrcoef(preds, targets)[0, 1]) if len(preds) > 1 else 0.0,
            "mae": float(np.mean(np.abs(preds - targets))),
            "n_samples": len(preds),
        }
