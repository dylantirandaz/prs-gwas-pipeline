import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

TENSOR_COLUMNS = (
    config.NUMERIC_FEATURES
    + ["chr_idx", "a1_idx", "a2_idx", "logOR", "SE"]
)

SPLITS = {
    "train": config.TRAIN_CHRS,
    "val": config.VAL_CHRS,
    "test": config.TEST_CHRS,
}


def _parquet_paths(chromosomes: list[int]) -> list[str]:
    return [
        str(config.DATA_PROCESSED_DIR / f"chr{c}_normalized.parquet")
        for c in chromosomes
        if (config.DATA_PROCESSED_DIR / f"chr{c}_normalized.parquet").exists()
    ]


def gwas_collate(batch: list[dict]) -> dict:
    collated = {}
    for key in batch[0]:
        collated[key] = torch.stack([item[key] for item in batch])

    numeric = torch.stack(
        [collated[f] for f in config.NUMERIC_FEATURES], dim=-1
    )
    return {
        "numeric": numeric,
        "chr_idx": collated["chr_idx"].long(),
        "a1_idx": collated["a1_idx"].long(),
        "a2_idx": collated["a2_idx"].long(),
        "target": collated["logOR"],
        "se": collated["SE"],
    }


def get_dataloader(split: str, shuffle: bool = True) -> DataLoader:
    paths = _parquet_paths(SPLITS[split])
    ds = Dataset.from_parquet(paths)
    ds = ds.with_format("torch", columns=TENSOR_COLUMNS)
    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=gwas_collate,
        drop_last=False,
    )


def count_batches(split: str) -> int:
    total = 0
    for c in SPLITS[split]:
        p = config.DATA_PROCESSED_DIR / f"chr{c}_normalized.parquet"
        if p.exists():
            total += pq.read_metadata(str(p)).num_rows
    return (total + config.BATCH_SIZE - 1) // config.BATCH_SIZE
