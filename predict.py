"""Generate PRS weights from trained model."""

import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

import config
from data.dataset import get_dataloader
from models.prs_net import PRSNet


def generate_weights(output_path: Path | None = None):
    """Load best model and generate PRS weights for all variants."""
    if output_path is None:
        output_path = config.PROJECT_ROOT / "prs_weights.tsv"

    # Load model
    best_path = config.CHECKPOINT_DIR / "best_model.pt"
    if not best_path.exists():
        print("ERROR: No trained model found at", best_path)
        print("Run `python train.py` first.")
        sys.exit(1)

    model = PRSNet()
    checkpoint = torch.load(best_path, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    print(f"Val metrics: {checkpoint['val_metrics']}")
    print(f"Device: {config.DEVICE}")

    # Generate predictions for all chromosomes
    all_results = []

    for chrom in tqdm(range(1, 23), desc="Generating PRS weights"):
        norm_path = config.DATA_PROCESSED_DIR / f"chr{chrom}_normalized.parquet"
        feat_path = config.DATA_PROCESSED_DIR / f"chr{chrom}_features.parquet"
        if not norm_path.exists() or not feat_path.exists():
            continue

        # Read normalized data for model input
        df_norm = pd.read_parquet(norm_path)
        # Read features data for SNP identifiers
        df_feat = pd.read_parquet(feat_path, columns=["SNP", "CHR", "BP"])

        # Process in chunks to avoid memory issues
        chunk_size = config.BATCH_SIZE
        preds_list = []

        with torch.no_grad():
            for start in range(0, len(df_norm), chunk_size):
                end = min(start + chunk_size, len(df_norm))
                chunk = df_norm.iloc[start:end]

                numeric = torch.tensor(
                    chunk[config.NUMERIC_FEATURES].values, dtype=torch.float32
                ).to(config.DEVICE, non_blocking=True)
                chr_idx = torch.tensor(
                    chunk["chr_idx"].values, dtype=torch.long
                ).to(config.DEVICE, non_blocking=True)
                a1_idx = torch.tensor(
                    chunk["a1_idx"].values, dtype=torch.long
                ).to(config.DEVICE, non_blocking=True)
                a2_idx = torch.tensor(
                    chunk["a2_idx"].values, dtype=torch.long
                ).to(config.DEVICE, non_blocking=True)

                pred = model(numeric, chr_idx, a1_idx, a2_idx)
                preds_list.append(pred.cpu().numpy())

        import numpy as np
        preds = np.concatenate(preds_list)

        result = pd.DataFrame({
            "SNP": df_feat["SNP"].values[:len(preds)],
            "CHR": df_feat["CHR"].values[:len(preds)],
            "BP": df_feat["BP"].values[:len(preds)],
            "PRS_WEIGHT": preds,
            "ORIGINAL_logOR": df_norm["logOR"].values[:len(preds)],
        })
        all_results.append(result)

    # Combine and save
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_path, sep="\t", index=False, float_format="%.6f")

    print(f"\nPRS weights saved to {output_path}")
    print(f"Total variants: {len(combined):,}")
    print(f"Weight statistics:")
    print(f"  Mean:   {combined['PRS_WEIGHT'].mean():.6f}")
    print(f"  Std:    {combined['PRS_WEIGHT'].std():.6f}")
    print(f"  Min:    {combined['PRS_WEIGHT'].min():.6f}")
    print(f"  Max:    {combined['PRS_WEIGHT'].max():.6f}")
    corr = combined["PRS_WEIGHT"].corr(combined["ORIGINAL_logOR"])
    print(f"  Correlation with original logOR: {corr:.4f}")


if __name__ == "__main__":
    generate_weights()
