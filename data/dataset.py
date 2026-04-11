import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SPLITS = {
    "train": config.TRAIN_CHRS,
    "val": config.VAL_CHRS,
    "test": config.TEST_CHRS,
}


class GWASDataset(IterableDataset):
    def __init__(self, chromosomes: list[int], batch_size: int = config.BATCH_SIZE,
                 shuffle: bool = True, chr_per_group: int = 1):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chr_per_group = chr_per_group
        self.file_paths = [
            config.DATA_PROCESSED_DIR / f"chr{c}_normalized.parquet"
            for c in chromosomes
            if (config.DATA_PROCESSED_DIR / f"chr{c}_normalized.parquet").exists()
        ]

    def _yield_batches(self, df: pd.DataFrame):
        numeric = torch.tensor(df[config.NUMERIC_FEATURES].values, dtype=torch.float32)
        chr_idx = torch.tensor(df["chr_idx"].values, dtype=torch.long)
        a1_idx = torch.tensor(df["a1_idx"].values, dtype=torch.long)
        a2_idx = torch.tensor(df["a2_idx"].values, dtype=torch.long)
        targets = torch.tensor(df["logOR"].values, dtype=torch.float32)
        se = torch.tensor(df["SE"].values, dtype=torch.float32)

        for start in range(0, len(df), self.batch_size):
            end = min(start + self.batch_size, len(df))
            yield {
                "numeric": numeric[start:end],
                "chr_idx": chr_idx[start:end],
                "a1_idx": a1_idx[start:end],
                "a2_idx": a2_idx[start:end],
                "target": targets[start:end],
                "se": se[start:end],
            }

    def __iter__(self):
        paths = list(self.file_paths)
        if self.shuffle:
            random.shuffle(paths)

        for group_start in range(0, len(paths), self.chr_per_group):
            group_paths = paths[group_start:group_start + self.chr_per_group]
            dfs = [pd.read_parquet(p) for p in group_paths]
            combined = pd.concat(dfs, ignore_index=True)

            if self.shuffle:
                combined = combined.sample(frac=1.0).reset_index(drop=True)

            yield from self._yield_batches(combined)


def get_dataloader(split: str, shuffle: bool = True) -> DataLoader:
    chr_per_group = config.CHR_PER_GROUP if shuffle else 1
    return DataLoader(
        GWASDataset(SPLITS[split], shuffle=shuffle, chr_per_group=chr_per_group),
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )


def count_batches(split: str) -> int:
    total = 0
    for c in SPLITS[split]:
        p = config.DATA_PROCESSED_DIR / f"chr{c}_normalized.parquet"
        if p.exists():
            total += pq.read_metadata(str(p)).num_rows
    return (total + config.BATCH_SIZE - 1) // config.BATCH_SIZE
