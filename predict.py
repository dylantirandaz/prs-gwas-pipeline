import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import config
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

    all_results = []

    for chrom in tqdm(range(1, 23), desc="Generating PRS weights"):
        norm_path = config.DATA_PROCESSED_DIR / f"chr{chrom}_normalized.parquet"
        if not norm_path.exists():
            continue

        df_norm = pd.read_parquet(norm_path)

        preds_list = []
        with torch.no_grad():
            for start in range(0, len(df_norm), config.BATCH_SIZE):
                end = min(start + config.BATCH_SIZE, len(df_norm))
                chunk = df_norm.iloc[start:end]

                numeric = torch.tensor(chunk[config.NUMERIC_FEATURES].values, dtype=torch.float32).to(config.DEVICE, non_blocking=True)
                chr_idx = torch.tensor(chunk["chr_idx"].values, dtype=torch.long).to(config.DEVICE, non_blocking=True)
                a1_idx = torch.tensor(chunk["a1_idx"].values, dtype=torch.long).to(config.DEVICE, non_blocking=True)
                a2_idx = torch.tensor(chunk["a2_idx"].values, dtype=torch.long).to(config.DEVICE, non_blocking=True)

                preds_list.append(model(numeric, chr_idx, a1_idx, a2_idx).cpu().numpy())

        preds = np.concatenate(preds_list)
        all_results.append(pd.DataFrame({
            "SNP": df_norm["SNP"].values[:len(preds)],
            "CHR": df_norm["CHR"].values[:len(preds)],
            "BP": df_norm["BP"].values[:len(preds)],
            "PRS_WEIGHT": preds,
            "ORIGINAL_logOR": df_norm["logOR"].values[:len(preds)],
        }))

    combined = pd.concat(all_results, ignore_index=True)
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
