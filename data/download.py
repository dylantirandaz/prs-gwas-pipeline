import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

FRQ_A_RE = re.compile(r"^FRQ_A_\d+$")
FRQ_U_RE = re.compile(r"^FRQ_U_\d+$")

KEEP_COLS = [
    "CHR", "SNP", "BP", "A1", "A2", "FRQ_A_1", "FRQ_U_1",
    "INFO", "OR", "SE", "P", "ngt", "Direction",
    "HetISqt", "HetDf", "HetPVa", "Nca", "Nco", "Neff",
]


def download_and_save():
    print(f"Loading {config.HF_DATASET} ({config.HF_CONFIG}) via HF datasets...")
    ds = load_dataset(config.HF_DATASET, config.HF_CONFIG, split="train")
    print(f"Loaded {len(ds):,} rows, {len(ds.column_names)} columns")

    frq_a_cols = sorted(c for c in ds.column_names if FRQ_A_RE.match(c))
    frq_u_cols = sorted(c for c in ds.column_names if FRQ_U_RE.match(c))

    needs_coalesce = len(frq_a_cols) > 1 or (frq_a_cols and frq_a_cols != ["FRQ_A_1"])

    if needs_coalesce:
        print(f"Coalescing FRQ columns: {frq_a_cols} -> FRQ_A_1, {frq_u_cols} -> FRQ_U_1")

        def coalesce_frq(batch):
            frq_a = np.array(batch[frq_a_cols[0]], dtype=np.float64)
            for col in frq_a_cols[1:]:
                vals = np.array(batch[col], dtype=np.float64)
                mask = np.isnan(frq_a)
                frq_a[mask] = vals[mask]

            frq_u = np.array(batch[frq_u_cols[0]], dtype=np.float64)
            for col in frq_u_cols[1:]:
                vals = np.array(batch[col], dtype=np.float64)
                mask = np.isnan(frq_u)
                frq_u[mask] = vals[mask]

            return {"FRQ_A_1": frq_a.tolist(), "FRQ_U_1": frq_u.tolist()}

        ds = ds.map(coalesce_frq, batched=True, batch_size=10_000, desc="Coalescing FRQ columns")

    def is_autosome(batch):
        results = []
        for chr_val in batch["CHR"]:
            try:
                results.append(1 <= int(chr_val) <= 22)
            except (ValueError, TypeError):
                results.append(False)
        return results

    ds = ds.filter(is_autosome, batched=True, batch_size=10_000, desc="Filtering autosomes")

    available = [c for c in KEEP_COLS if c in ds.column_names]
    ds = ds.select_columns(available)

    config.DATASET_RAW_PATH.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(config.DATASET_RAW_PATH))
    print(f"\nSaved {len(ds):,} variants to {config.DATASET_RAW_PATH}")


if __name__ == "__main__":
    download_and_save()
