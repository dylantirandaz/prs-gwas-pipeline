import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from training.losses import SEWeightedMSELoss
from training.metrics import MetricsAccumulator


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device = config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.criterion = SEWeightedMSELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = []

        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def setup_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
            min_lr=config.MIN_LR,
        )

    def _to_device(self, batch: dict) -> tuple:
        return (
            batch["numeric"].to(self.device, non_blocking=True),
            batch["chr_idx"].to(self.device, non_blocking=True),
            batch["a1_idx"].to(self.device, non_blocking=True),
            batch["a2_idx"].to(self.device, non_blocking=True),
            batch["target"].to(self.device, non_blocking=True),
            batch["se"].to(self.device, non_blocking=True),
        )

    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.model.train()
        metrics = MetricsAccumulator()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        for batch in pbar:
            numeric, chr_idx, a1_idx, a2_idx, target, se = self._to_device(batch)

            pred = self.model(numeric, chr_idx, a1_idx, a2_idx)
            loss = self.criterion(pred, target, se)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_NORM)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            metrics.update(pred, target)
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        epoch_metrics = metrics.compute()
        epoch_metrics["loss"] = total_loss / max(n_batches, 1)
        return epoch_metrics

    @torch.no_grad()
    def validate(self, dataloader, epoch: int) -> dict:
        self.model.eval()
        metrics = MetricsAccumulator()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]  "):
            numeric, chr_idx, a1_idx, a2_idx, target, se = self._to_device(batch)

            pred = self.model(numeric, chr_idx, a1_idx, a2_idx)
            loss = self.criterion(pred, target, se)

            total_loss += loss.item()
            n_batches += 1
            metrics.update(pred, target)

        epoch_metrics = metrics.compute()
        epoch_metrics["loss"] = total_loss / max(n_batches, 1)
        return epoch_metrics

    def save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
        }
        torch.save(state, config.CHECKPOINT_DIR / f"checkpoint_epoch{epoch+1}.pt")

        if is_best:
            torch.save(state, config.CHECKPOINT_DIR / "best_model.pt")
            print(f"  New best model saved (val_loss={val_metrics['loss']:.6f})")

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict, elapsed: float):
        entry = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_sec": elapsed,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        self.history.append(entry)

        print(
            f"  Train — loss: {train_metrics['loss']:.6f}, "
            f"R²: {train_metrics['r2']:.4f}, "
            f"r: {train_metrics['pearson_r']:.4f}, "
            f"MAE: {train_metrics['mae']:.6f}"
        )
        print(
            f"  Val   — loss: {val_metrics['loss']:.6f}, "
            f"R²: {val_metrics['r2']:.4f}, "
            f"r: {val_metrics['pearson_r']:.4f}, "
            f"MAE: {val_metrics['mae']:.6f}"
        )
        print(f"  Time: {elapsed:.1f}s, LR: {entry['lr']:.6f}")

        with open(config.LOG_DIR / "training_log.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def fit(self, train_loader, val_loader, steps_per_epoch: int):
        self.setup_scheduler()

        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Epochs: {config.EPOCHS}, Batch size: {config.BATCH_SIZE}")
        print(f"Estimated batches/epoch: ~{steps_per_epoch}\n")

        for epoch in range(config.EPOCHS):
            if epoch < config.WARMUP_EPOCHS:
                warmup_lr = config.LEARNING_RATE * (epoch + 1) / config.WARMUP_EPOCHS
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            if self.device.type == "mps":
                torch.mps.synchronize()
            start = time.time()

            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)

            if self.device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.time() - start

            if epoch >= config.WARMUP_EPOCHS:
                self.scheduler.step(val_metrics["loss"])

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            self.log_epoch(epoch, train_metrics, val_metrics, elapsed)

            if self.device.type == "mps":
                torch.mps.empty_cache()
            print()

            if self.patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {config.EARLY_STOPPING_PATIENCE} epochs)")
                break

        print(f"Training complete. Best val loss: {self.best_val_loss:.6f}")
