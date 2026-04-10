import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# FRQ columns embed study-specific sample sizes (e.g. FRQ_A_32446, FRQ_A_6630)
# which vary across parquet shard sets — normalize them to stable names
FRQ_A_RE = re.compile(r"^FRQ_A_\d+$")
FRQ_U_RE = re.compile(r"^FRQ_U_\d+$")

KEEP_COLS = [
    "CHR", "SNP", "BP", "A1", "A2", "FRQ_A_1", "FRQ_U_1",
    "INFO", "OR", "SE", "P", "ngt", "Direction",
    "HetISqt", "HetDf", "HetPVa", "Nca", "Nco", "Neff",
]


def list_parquet_files() -> list[str]:
    api = HfApi()
    files = api.list_repo_tree(
        "OpenMed/pgc-schizophrenia",
        repo_type="dataset",
        path_in_repo="data/scz2022",
    )
    return sorted(
        f.path for f in files
        if hasattr(f, "path") and f.path.endswith(".parquet")
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if FRQ_A_RE.match(col):
            rename_map[col] = "FRQ_A_1"
        elif FRQ_U_RE.match(col):
            rename_map[col] = "FRQ_U_1"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df[[c for c in KEEP_COLS if c in df.columns]]


def _ensure_all_columns(table: pa.Table) -> pa.Table:
    for col_name in KEEP_COLS:
        if col_name not in table.column_names:
            table = table.append_column(col_name, pa.nulls(len(table), type=pa.float64()))
    return table.select([c for c in KEEP_COLS if c in table.column_names])


def _align_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    for field in schema:
        if field.name not in table.column_names:
            table = table.append_column(field.name, pa.nulls(len(table), type=field.type))
    table = table.select([f.name for f in schema])
    return table.cast(schema)


def download_and_save():
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Listing parquet files for {config.HF_DATASET} ({config.HF_CONFIG})...")
    parquet_paths = list_parquet_files()
    print(f"Found {len(parquet_paths)} parquet shards\n")

    writers: dict[int, pq.ParquetWriter] = {}
    writer_schemas: dict[int, pa.Schema] = {}
    chr_counts: dict[int, int] = {}
    total = 0

    try:
        for shard_path in tqdm(parquet_paths, desc="Processing shards"):
            local_path = hf_hub_download(
                "OpenMed/pgc-schizophrenia",
                shard_path,
                repo_type="dataset",
            )

            df = pd.read_parquet(local_path)
            df = normalize_columns(df)
            total += len(df)

            for chrom, group in df.groupby("CHR"):
                try:
                    chrom = int(chrom)
                except (ValueError, TypeError):
                    continue
                if not (1 <= chrom <= 22):
                    continue
                table = pa.Table.from_pandas(group, preserve_index=False)

                if chrom not in writers:
                    table = _ensure_all_columns(table)
                    outpath = config.DATA_RAW_DIR / f"chr{chrom}.parquet"
                    writers[chrom] = pq.ParquetWriter(str(outpath), table.schema)
                    writer_schemas[chrom] = table.schema
                    chr_counts[chrom] = 0
                else:
                    table = _align_to_schema(table, writer_schemas[chrom])

                writers[chrom].write_table(table)
                chr_counts[chrom] += len(group)

    finally:
        for w in writers.values():
            w.close()

    print(f"\nDone. Total variants: {total:,}")
    for chrom in sorted(chr_counts):
        print(f"  chr{chrom}: {chr_counts[chrom]:,} variants")


if __name__ == "__main__":
    download_and_save()
