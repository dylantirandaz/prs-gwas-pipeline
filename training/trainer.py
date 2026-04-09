"""Training and validation loops with MPS handling."""

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
        self.history = []

        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def setup_scheduler(self, steps_per_epoch: int):
        """Set up cosine annealing with linear warmup."""
        total_steps = config.EPOCHS * steps_per_epoch
        warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def train_epoch(self, dataloader, epoch: int) -> dict:
        self.model.train()
        metrics = MetricsAccumulator()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")

        for batch in pbar:
            # Transfer to device with non_blocking
            numeric = batch["numeric"].to(self.device, non_blocking=True)
            chr_idx = batch["chr_idx"].to(self.device, non_blocking=True)
            a1_idx = batch["a1_idx"].to(self.device, non_blocking=True)
            a2_idx = batch["a2_idx"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            se = batch["se"].to(self.device, non_blocking=True)

            # Forward
            pred = self.model(numeric, chr_idx, a1_idx, a2_idx)
            loss = self.criterion(pred, target, se)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.GRAD_CLIP_NORM
            )
            self.optimizer.step()

            if hasattr(self, "scheduler"):
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            metrics.update(pred, target)

            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = total_loss / max(n_batches, 1)
        epoch_metrics = metrics.compute()
        epoch_metrics["loss"] = avg_loss
        return epoch_metrics

    @torch.no_grad()
    def validate(self, dataloader, epoch: int) -> dict:
        self.model.eval()
        metrics = MetricsAccumulator()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]  ")

        for batch in pbar:
            numeric = batch["numeric"].to(self.device, non_blocking=True)
            chr_idx = batch["chr_idx"].to(self.device, non_blocking=True)
            a1_idx = batch["a1_idx"].to(self.device, non_blocking=True)
            a2_idx = batch["a2_idx"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            se = batch["se"].to(self.device, non_blocking=True)

            pred = self.model(numeric, chr_idx, a1_idx, a2_idx)
            loss = self.criterion(pred, target, se)

            total_loss += loss.item()
            n_batches += 1
            metrics.update(pred, target)

        avg_loss = total_loss / max(n_batches, 1)
        epoch_metrics = metrics.compute()
        epoch_metrics["loss"] = avg_loss
        return epoch_metrics

    def save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_metrics": val_metrics,
        }
        path = config.CHECKPOINT_DIR / f"checkpoint_epoch{epoch+1}.pt"
        torch.save(state, path)

        if is_best:
            best_path = config.CHECKPOINT_DIR / "best_model.pt"
            torch.save(state, best_path)
            print(f"  New best model saved (val_loss={val_metrics['loss']:.6f})")

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict,
                  elapsed: float):
        entry = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_sec": elapsed,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        self.history.append(entry)

        # Print summary
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

        # Save log
        log_path = config.LOG_DIR / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def fit(self, train_loader, val_loader, steps_per_epoch: int):
        """Full training loop."""
        self.setup_scheduler(steps_per_epoch)

        print(f"Training on {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Epochs: {config.EPOCHS}, Batch size: {config.BATCH_SIZE}")
        print(f"Estimated batches/epoch: ~{steps_per_epoch}")
        print()

        for epoch in range(config.EPOCHS):
            # MPS timing
            if self.device.type == "mps":
                torch.mps.synchronize()
            start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            if self.device.type == "mps":
                torch.mps.synchronize()
            elapsed = time.time() - start

            # Checkpoint
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Log
            self.log_epoch(epoch, train_metrics, val_metrics, elapsed)

            # MPS memory cleanup
            if self.device.type == "mps":
                torch.mps.empty_cache()

            print()

        print(f"Training complete. Best val loss: {self.best_val_loss:.6f}")
