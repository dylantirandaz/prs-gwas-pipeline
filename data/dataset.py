import random
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class GWASDataset(IterableDataset):
    def __init__(self, chromosomes: list[int], batch_size: int = config.BATCH_SIZE, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_paths = [
            config.DATA_PROCESSED_DIR / f"chr{c}_normalized.parquet"
            for c in chromosomes
            if (config.DATA_PROCESSED_DIR / f"chr{c}_normalized.parquet").exists()
        ]

    def __iter__(self):
        paths = list(self.file_paths)
        if self.shuffle:
            random.shuffle(paths)

        for path in paths:
            df = pd.read_parquet(path)
            if self.shuffle:
                df = df.sample(frac=1.0).reset_index(drop=True)

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


def get_dataloader(chromosomes: list[int], shuffle: bool = True) -> DataLoader:
    # num_workers=0 required for MPS compatibility
    return DataLoader(
        GWASDataset(chromosomes, shuffle=shuffle),
        batch_size=None,
        num_workers=0,
        pin_memory=False,
    )


def count_batches(chromosomes: list[int]) -> int:
    total_rows = 0
    for chrom in chromosomes:
        p = config.DATA_PROCESSED_DIR / f"chr{chrom}_normalized.parquet"
        if p.exists():
            total_rows += len(pd.read_parquet(p, columns=["logOR"]))
    return (total_rows + config.BATCH_SIZE - 1) // config.BATCH_SIZE
