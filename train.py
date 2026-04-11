import sys

import torch

import config
from data.dataset import get_dataloader, count_batches
from models.prs_net import PRSNet
from training.trainer import Trainer


def main():
    if not (config.DATA_PROCESSED_DIR / "chr1_normalized.parquet").exists():
        print("Processed dataset not found. Run these first:")
        print("  python data/download.py")
        print("  python data/preprocessing.py")
        sys.exit(1)

    print("Counting training batches...")
    steps_per_epoch = count_batches("train")
    print(f"Estimated batches per epoch: {steps_per_epoch}")

    train_loader = get_dataloader("train", shuffle=True)
    val_loader = get_dataloader("val", shuffle=False)

    model = PRSNet()
    print(f"\nPRSNet architecture:")
    print(model)
    print(f"Trainable parameters: {model.count_parameters():,}\n")

    trainer = Trainer(model, device=config.DEVICE)
    trainer.fit(train_loader, val_loader, steps_per_epoch)

    print("\n" + "=" * 60)
    print("Test set evaluation (chr22):")
    test_loader = get_dataloader("test", shuffle=False)

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
