"""Main entry point: train PRS neural network on GWAS summary statistics."""

import sys
from pathlib import Path

import torch

import config
from data.dataset import get_dataloader, count_batches
from models.prs_net import PRSNet
from training.trainer import Trainer


def main():
    # Verify processed data exists
    sample_file = config.DATA_PROCESSED_DIR / "chr1_normalized.parquet"
    if not sample_file.exists():
        print("ERROR: Processed data not found. Run these first:")
        print("  python data/download.py")
        print("  python data/preprocessing.py")
        sys.exit(1)

    # Estimate steps per epoch
    print("Counting training batches...")
    steps_per_epoch = count_batches(config.TRAIN_CHRS)
    print(f"Estimated batches per epoch: {steps_per_epoch}")

    # Create data loaders
    train_loader = get_dataloader(config.TRAIN_CHRS, shuffle=True)
    val_loader = get_dataloader(config.VAL_CHRS, shuffle=False)

    # Create model
    model = PRSNet()
    print(f"\nPRSNet architecture:")
    print(model)
    print(f"Trainable parameters: {model.count_parameters():,}\n")

    # Train
    trainer = Trainer(model, device=config.DEVICE)
    trainer.fit(train_loader, val_loader, steps_per_epoch)

    # Final test evaluation
    print("\n" + "=" * 60)
    print("Test set evaluation (chr22):")
    test_loader = get_dataloader(config.TEST_CHRS, shuffle=False)

    # Load best model
    best_path = config.CHECKPOINT_DIR / "best_model.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=config.DEVICE, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(config.DEVICE)

    test_metrics = trainer.validate(test_loader, epoch=config.EPOCHS - 1)
    print(f"  Test — loss: {test_metrics['loss']:.6f}, "
          f"R²: {test_metrics['r2']:.4f}, "
          f"r: {test_metrics['pearson_r']:.4f}, "
          f"MAE: {test_metrics['mae']:.6f}, "
          f"n={test_metrics['n_samples']:,}")


if __name__ == "__main__":
    main()
