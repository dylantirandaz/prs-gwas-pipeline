import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

FRQ_A_RE = re.compile(r"^FRQ_A_\d+$")
FRQ_U_RE = re.compile(r"^FRQ_U_\d+$")

KEEP_COLS = [
    "CHR", "SNP", "BP", "A1", "A2", "FRQ_A_1", "FRQ_U_1",
    "INFO", "OR", "SE", "P", "ngt", "Direction",
    "HetISqt", "HetDf", "HetPVa", "Nca", "Nco", "Neff",
]

OUTPUT_SCHEMA = pa.schema([
    ("CHR", pa.int64()),
    ("SNP", pa.string()),
    ("BP", pa.float64()),
    ("A1", pa.string()),
    ("A2", pa.string()),
    ("FRQ_A_1", pa.float64()),
    ("FRQ_U_1", pa.float64()),
    ("INFO", pa.float64()),
    ("OR", pa.float64()),
    ("SE", pa.float64()),
    ("P", pa.float64()),
    ("ngt", pa.float64()),
    ("Direction", pa.string()),
    ("HetISqt", pa.float64()),
    ("HetDf", pa.float64()),
    ("HetPVa", pa.float64()),
    ("Nca", pa.float64()),
    ("Nco", pa.float64()),
    ("Neff", pa.float64()),
])


def download_and_save():
    print(f"Loading {config.HF_DATASET} via HF datasets (streaming)...")
    ds = load_dataset(config.HF_DATASET, split="train", streaming=True)

    frq_a_cols = sorted(c for c in ds.column_names if FRQ_A_RE.match(c))
    frq_u_cols = sorted(c for c in ds.column_names if FRQ_U_RE.match(c))
    print(f"FRQ_A variants: {frq_a_cols}")
    print(f"FRQ_U variants: {frq_u_cols}")

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    writers: dict[int, pq.ParquetWriter] = {}
    chr_counts: dict[int, int] = {}
    total = 0

    try:
        for batch in tqdm(ds.iter(batch_size=10_000), desc="Streaming"):
            df = pd.DataFrame(batch)
            total += len(df)

            # Coalesce FRQ columns
            df["FRQ_A_1"] = df[frq_a_cols].bfill(axis=1).iloc[:, 0].infer_objects(copy=False)
            df["FRQ_U_1"] = df[frq_u_cols].bfill(axis=1).iloc[:, 0].infer_objects(copy=False)

            # Convert CHR to numeric, drop non-autosomes
            df["CHR"] = pd.to_numeric(df["CHR"], errors="coerce")
            df = df.dropna(subset=["CHR"])
            df["CHR"] = df["CHR"].astype(int)
            df = df[df["CHR"].between(1, 22)]

            if df.empty:
                continue

            available = [c for c in KEEP_COLS if c in df.columns]
            df = df[available]

            for chrom, group in df.groupby("CHR"):
                chrom = int(chrom)
                table = pa.Table.from_pandas(group, preserve_index=False)
                table = table.cast(OUTPUT_SCHEMA)

                if chrom not in writers:
                    outpath = config.DATA_RAW_DIR / f"chr{chrom}.parquet"
                    writers[chrom] = pq.ParquetWriter(str(outpath), OUTPUT_SCHEMA)
                    chr_counts[chrom] = 0

                writers[chrom].write_table(table)
                chr_counts[chrom] += len(group)

    finally:
        for w in writers.values():
            w.close()

    print(f"\nDone. Total rows streamed: {total:,}")
    for chrom in sorted(chr_counts):
        print(f"  chr{chrom}: {chr_counts[chrom]:,} variants")


if __name__ == "__main__":
    download_and_save()
