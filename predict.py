import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm

import config
from data.dataset import TENSOR_COLUMNS, _parquet_paths
from models.prs_net import PRSNet


def generate_weights(output_path: Path | None = None):
    if output_path is None:
        output_path = config.PROJECT_ROOT / "prs_weights.tsv"

    best_path = config.CHECKPOINT_DIR / "best_model.pt"
    if not best_path.exists():
        print("No trained model found. Run `python train.py` first.")
        sys.exit(1)

    model = PRSNet()
    checkpoint = torch.load(best_path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"Val metrics: {checkpoint['val_metrics']}")
    print(f"Device: {config.DEVICE}")

    all_chrs = sorted(config.TRAIN_CHRS + config.VAL_CHRS + config.TEST_CHRS)
    paths = _parquet_paths(all_chrs)
    all_data = Dataset.from_parquet(paths)

    snps = all_data["SNP"]
    chrs = all_data["CHR"]
    bps = all_data["BP"]
    log_or_raw = np.array(all_data["logOR"], dtype=np.float64)

    tensor_ds = all_data.with_format("torch", columns=TENSOR_COLUMNS)

    preds_list = []
    with torch.no_grad():
        for start in tqdm(range(0, len(tensor_ds), config.BATCH_SIZE), desc="Generating PRS weights"):
            end = min(start + config.BATCH_SIZE, len(tensor_ds))
            batch = tensor_ds[start:end]

            numeric = torch.stack(
                [batch[f] for f in config.NUMERIC_FEATURES], dim=-1
            ).to(config.DEVICE, non_blocking=True)
            chr_idx = batch["chr_idx"].long().to(config.DEVICE, non_blocking=True)
            a1_idx = batch["a1_idx"].long().to(config.DEVICE, non_blocking=True)
            a2_idx = batch["a2_idx"].long().to(config.DEVICE, non_blocking=True)

            preds_list.append(model(numeric, chr_idx, a1_idx, a2_idx).cpu().numpy())

    preds = np.concatenate(preds_list)
    combined = pd.DataFrame({
        "SNP": snps[:len(preds)],
        "CHR": chrs[:len(preds)],
        "BP": bps[:len(preds)],
        "PRS_WEIGHT": preds,
        "ORIGINAL_logOR": log_or_raw[:len(preds)],
    })
    combined.to_csv(output_path, sep="\t", index=False, float_format="%.6f")

    print(f"\nPRS weights saved to {output_path}")
    print(f"Total variants: {len(combined):,}")
    print(f"  Mean:   {combined['PRS_WEIGHT'].mean():.6f}")
    print(f"  Std:    {combined['PRS_WEIGHT'].std():.6f}")
    print(f"  Min:    {combined['PRS_WEIGHT'].min():.6f}")
    print(f"  Max:    {combined['PRS_WEIGHT'].max():.6f}")
    print(f"  Correlation with original logOR: {combined['PRS_WEIGHT'].corr(combined['ORIGINAL_logOR']):.4f}")


if __name__ == "__main__":
    generate_weights()
