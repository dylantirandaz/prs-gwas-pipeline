import json
import sys
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ALLELE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}

FINAL_COLUMNS = (
    config.NUMERIC_FEATURES
    + ["chr_idx", "a1_idx", "a2_idx", "logOR", "SE", "SNP", "CHR", "BP"]
)


def _direction_ratio(d) -> float:
    if not isinstance(d, str) or len(d) == 0:
        return 0.5
    plus_count = d.count("+")
    total = plus_count + d.count("-")
    return plus_count / total if total > 0 else 0.5


def _compute_bp_stats(ds) -> dict[int, dict]:
    chr_vals = np.array(ds["CHR"], dtype=np.int64)
    bp_vals = np.array(ds["BP"], dtype=np.float64)
    stats = {}
    for chrom in range(1, 23):
        mask = chr_vals == chrom
        if not mask.any():
            continue
        bp_chr = bp_vals[mask]
        stats[chrom] = {"min": float(np.nanmin(bp_chr)), "max": float(np.nanmax(bp_chr))}
    return stats


def qc_filter(batch):
    info = np.array(batch["INFO"], dtype=np.float64)
    frq_a = np.array(batch["FRQ_A_1"], dtype=np.float64)
    frq_u = np.array(batch["FRQ_U_1"], dtype=np.float64)
    se = np.array(batch["SE"], dtype=np.float64)
    or_val = np.array(batch["OR"], dtype=np.float64)

    keep = (
        ~np.isnan(info) & ~np.isnan(se) & ~np.isnan(or_val)
        & (info >= config.MIN_INFO)
        & (frq_a >= config.MIN_MAF) & (frq_a <= config.MAX_MAF)
        & (frq_u >= config.MIN_MAF) & (frq_u <= config.MAX_MAF)
        & (se > 0) & (or_val > 0)
    )
    return keep.tolist()


def engineer_features(batch, bp_stats):
    chr_vals = np.array(batch["CHR"], dtype=np.int64)
    bp_vals = np.array(batch["BP"], dtype=np.float64)
    or_vals = np.array(batch["OR"], dtype=np.float64)
    se_vals = np.array(batch["SE"], dtype=np.float64)
    frq_a = np.array(batch["FRQ_A_1"], dtype=np.float64)
    frq_u = np.array(batch["FRQ_U_1"], dtype=np.float64)
    het_pva = np.array(batch["HetPVa"], dtype=np.float64)
    het_isqt = np.array(batch["HetISqt"], dtype=np.float64)
    het_df = np.array(batch["HetDf"], dtype=np.float64)
    nca = np.array(batch["Nca"], dtype=np.float64)
    nco = np.array(batch["Nco"], dtype=np.float64)
    neff = np.array(batch["Neff"], dtype=np.float64)

    ngt_raw = np.array(batch["ngt"], dtype=np.float64)
    ngt_vals = np.where(np.isnan(ngt_raw), 0.0, ngt_raw)

    bp_norm = np.zeros(len(chr_vals), dtype=np.float64)
    for chrom in np.unique(chr_vals):
        mask = chr_vals == chrom
        s = bp_stats[int(chrom)]
        bp_range = max(s["max"] - s["min"], 1.0)
        bp_norm[mask] = (bp_vals[mask] - s["min"]) / bp_range

    direction_ratio = np.array(
        [_direction_ratio(d) for d in batch["Direction"]], dtype=np.float64
    )
    a1_idx = np.array(
        [ALLELE_MAP.get(str(a).upper(), 4) for a in batch["A1"]], dtype=np.int64
    )
    a2_idx = np.array(
        [ALLELE_MAP.get(str(a).upper(), 4) for a in batch["A2"]], dtype=np.int64
    )
    chr_idx = (np.clip(chr_vals, 1, 22) - 1).astype(np.int64)

    return {
        "BP_norm": bp_norm.tolist(),
        "FRQ_A": frq_a.tolist(),
        "FRQ_U": frq_u.tolist(),
        "FRQ_DIFF": np.abs(frq_a - frq_u).tolist(),
        "FRQ_MEAN": ((frq_a + frq_u) / 2.0).tolist(),
        "log_SE": np.log(se_vals).tolist(),
        "neg_log10_HetPVal": (-np.log10(np.clip(het_pva, 1e-300, None))).tolist(),
        "HetISqt_scaled": (het_isqt / 100.0).tolist(),
        "log_HetDf": np.log(np.clip(het_df, 1, None)).tolist(),
        "log_Nca": np.log(np.clip(nca, 1, None)).tolist(),
        "log_Nco": np.log(np.clip(nco, 1, None)).tolist(),
        "log_Neff": np.log(np.clip(neff, 1, None)).tolist(),
        "ngt": ngt_vals.tolist(),
        "direction_ratio": direction_ratio.tolist(),
        "chr_idx": chr_idx.tolist(),
        "a1_idx": a1_idx.tolist(),
        "a2_idx": a2_idx.tolist(),
        "logOR": np.log(or_vals).tolist(),
    }


def compute_normalization_stats(ds) -> dict:
    n_sample = max(1, int(len(ds) * config.NORM_SAMPLE_FRAC))
    indices = np.random.RandomState(42).choice(len(ds), size=n_sample, replace=False)
    sample = ds.select(indices)

    stats = {}
    for feat in config.NUMERIC_FEATURES:
        vals = np.array(sample[feat], dtype=np.float64)
        stats[feat] = {
            "mean": float(np.nanmean(vals)),
            "std": float(max(np.nanstd(vals), 1e-8)),
        }
    return stats


def normalize_features(batch, stats):
    result = {}
    for feat in config.NUMERIC_FEATURES:
        vals = np.array(batch[feat], dtype=np.float64)
        result[feat] = ((vals - stats[feat]["mean"]) / stats[feat]["std"]).tolist()
    return result


def split_filter(batch, chromosomes):
    chr_vals = np.array(batch["CHR"])
    return np.isin(chr_vals, chromosomes).tolist()


def process_and_save_all():
    print("Loading raw dataset...")
    ds = load_from_disk(str(config.DATASET_RAW_PATH))
    print(f"  {len(ds):,} variants")

    print("\nQC filtering...")
    ds = ds.filter(qc_filter, batched=True, batch_size=10_000, desc="QC filter")
    print(f"  {len(ds):,} variants after filtering")

    print("\nComputing per-chromosome BP stats...")
    bp_stats = _compute_bp_stats(ds)

    print("Feature engineering...")
    ds = ds.map(
        engineer_features,
        batched=True,
        batch_size=10_000,
        fn_kwargs={"bp_stats": bp_stats},
        desc="Engineering features",
    )

    available_final = [c for c in FINAL_COLUMNS if c in ds.column_names]
    ds = ds.select_columns(available_final)

    print(f"\nComputing normalization stats from {config.NORM_SAMPLE_FRAC:.0%} sample...")
    stats = compute_normalization_stats(ds)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.NORM_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved normalization stats to {config.NORM_STATS_PATH}")

    print("\nNormalizing features...")
    ds = ds.map(
        normalize_features,
        batched=True,
        batch_size=10_000,
        fn_kwargs={"stats": stats},
        desc="Normalizing",
    )

    print("\nSplitting by chromosome...")
    train_ds = ds.filter(
        split_filter, batched=True,
        fn_kwargs={"chromosomes": config.TRAIN_CHRS}, desc="Train split",
    )
    val_ds = ds.filter(
        split_filter, batched=True,
        fn_kwargs={"chromosomes": config.VAL_CHRS}, desc="Val split",
    )
    test_ds = ds.filter(
        split_filter, batched=True,
        fn_kwargs={"chromosomes": config.TEST_CHRS}, desc="Test split",
    )

    dataset_dict = DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})
    config.DATASET_PATH.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(config.DATASET_PATH))

    print(f"\nSaved to {config.DATASET_PATH}")
    print(f"  Train: {len(train_ds):,} variants (chr {config.TRAIN_CHRS})")
    print(f"  Val:   {len(val_ds):,} variants (chr {config.VAL_CHRS})")
    print(f"  Test:  {len(test_ds):,} variants (chr {config.TEST_CHRS})")


if __name__ == "__main__":
    process_and_save_all()
