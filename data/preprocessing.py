import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ALLELE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def process_chromosome(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["INFO"] >= config.MIN_INFO]
    df = df[df["FRQ_A_1"].between(config.MIN_MAF, config.MAX_MAF)]
    df = df[df["FRQ_U_1"].between(config.MIN_MAF, config.MAX_MAF)]
    df = df[(df["SE"] > 0) & (df["OR"] > 0)]
    df = df.dropna(subset=["OR", "SE", "INFO"])

    df["logOR"] = np.log(df["OR"])

    bp_min, bp_max = df["BP"].min(), df["BP"].max()
    bp_range = bp_max - bp_min if bp_max > bp_min else 1.0
    df["BP_norm"] = (df["BP"] - bp_min) / bp_range

    df["FRQ_A"] = df["FRQ_A_1"]
    df["FRQ_U"] = df["FRQ_U_1"]
    df["FRQ_DIFF"] = (df["FRQ_A_1"] - df["FRQ_U_1"]).abs()
    df["FRQ_MEAN"] = (df["FRQ_A_1"] + df["FRQ_U_1"]) / 2.0

    df["log_SE"] = np.log(df["SE"])
    df["neg_log10_HetPVal"] = -np.log10(df["HetPVa"].clip(lower=1e-300))
    df["HetISqt_scaled"] = df["HetISqt"] / 100.0
    df["log_HetDf"] = np.log(df["HetDf"].clip(lower=1))
    df["log_Nca"] = np.log(df["Nca"].clip(lower=1))
    df["log_Nco"] = np.log(df["Nco"].clip(lower=1))
    df["log_Neff"] = np.log(df["Neff"].clip(lower=1))
    df["ngt"] = df["ngt"].fillna(0).astype(float)
    df["direction_ratio"] = df["Direction"].apply(_direction_ratio)

    df["chr_idx"] = df["CHR"].clip(1, 22).astype(int) - 1
    df["a1_idx"] = df["A1"].str.upper().map(ALLELE_MAP).fillna(4).astype(int)
    df["a2_idx"] = df["A2"].str.upper().map(ALLELE_MAP).fillna(4).astype(int)

    keep_cols = (
        config.NUMERIC_FEATURES
        + ["chr_idx", "a1_idx", "a2_idx", "logOR", "SE", "SNP", "CHR", "BP"]
    )
    return df[keep_cols].reset_index(drop=True)


def _direction_ratio(d) -> float:
    if not isinstance(d, str) or len(d) == 0:
        return 0.5
    plus_count = d.count("+")
    total = d.count("+") + d.count("-")
    return plus_count / total if total > 0 else 0.5


def compute_normalization_stats() -> dict:
    samples = []
    for chrom in sorted(config.TRAIN_CHRS + config.VAL_CHRS + config.TEST_CHRS):
        path = config.DATA_RAW_DIR / f"chr{chrom}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = process_chromosome(df)
        n_sample = max(1, int(len(df) * config.NORM_SAMPLE_FRAC))
        samples.append(df[config.NUMERIC_FEATURES].sample(n=n_sample, random_state=42))

    combined = pd.concat(samples, ignore_index=True)
    stats = {}
    for feat in config.NUMERIC_FEATURES:
        stats[feat] = {
            "mean": float(combined[feat].mean()),
            "std": float(max(combined[feat].std(), 1e-8)),
        }
    return stats


def save_normalization_stats(stats: dict):
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.NORM_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization stats to {config.NORM_STATS_PATH}")


def load_normalization_stats() -> dict:
    with open(config.NORM_STATS_PATH) as f:
        return json.load(f)


def normalize_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()
    for feat in config.NUMERIC_FEATURES:
        df[feat] = (df[feat] - stats[feat]["mean"]) / stats[feat]["std"]
    return df


def process_and_save_all():
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Pass 1: Feature engineering...")
    for chrom in tqdm(range(1, 23), desc="Processing chromosomes"):
        raw_path = config.DATA_RAW_DIR / f"chr{chrom}.parquet"
        if not raw_path.exists():
            print(f"  Skipping chr{chrom} (not found)")
            continue
        df = pd.read_parquet(raw_path)
        df = process_chromosome(df)
        df.to_parquet(config.DATA_PROCESSED_DIR / f"chr{chrom}_features.parquet", index=False)
        print(f"  chr{chrom}: {len(df):,} variants after filtering")

    print("\nComputing normalization stats from 1% sample...")
    stats = compute_normalization_stats()
    save_normalization_stats(stats)

    print("\nPass 2: Normalizing features...")
    for chrom in tqdm(range(1, 23), desc="Normalizing"):
        feat_path = config.DATA_PROCESSED_DIR / f"chr{chrom}_features.parquet"
        if not feat_path.exists():
            continue
        df = pd.read_parquet(feat_path)
        df = normalize_features(df, stats)
        df.to_parquet(config.DATA_PROCESSED_DIR / f"chr{chrom}_normalized.parquet", index=False)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    process_and_save_all()
