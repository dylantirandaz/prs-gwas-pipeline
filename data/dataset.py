"""PyTorch IterableDataset for streaming per-chromosome parquet files."""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class GWASDataset(IterableDataset):
    """Streams batches from per-chromosome normalized parquet files.

    - Shuffles chromosome file order each epoch
    - Shuffles rows within each chromosome file
    - Yields pre-batched tensors of size batch_size
    """

    def __init__(self, chromosomes: list[int], batch_size: int = config.BATCH_SIZE, shuffle: bool = True):
        self.chromosomes = chromosomes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_paths = []
        for chrom in chromosomes:
            p = config.DATA_PROCESSED_DIR / f"chr{chrom}_normalized.parquet"
            if p.exists():
                self.file_paths.append(p)

    def __iter__(self):
        paths = list(self.file_paths)
        if self.shuffle:
            random.shuffle(paths)

        for path in paths:
            df = pd.read_parquet(path)

            if self.shuffle:
                df = df.sample(frac=1.0).reset_index(drop=True)

            # Extract tensors
            numeric = torch.tensor(
                df[config.NUMERIC_FEATURES].values, dtype=torch.float32
            )
            chr_idx = torch.tensor(df["chr_idx"].values, dtype=torch.long)
            a1_idx = torch.tensor(df["a1_idx"].values, dtype=torch.long)
            a2_idx = torch.tensor(df["a2_idx"].values, dtype=torch.long)
            targets = torch.tensor(df["logOR"].values, dtype=torch.float32)
            se = torch.tensor(df["SE"].values, dtype=torch.float32)

            n = len(df)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                yield {
                    "numeric": numeric[start:end],
                    "chr_idx": chr_idx[start:end],
                    "a1_idx": a1_idx[start:end],
                    "a2_idx": a2_idx[start:end],
                    "target": targets[start:end],
                    "se": se[start:end],
                }


def get_dataloader(chromosomes: list[int], shuffle: bool = True) -> DataLoader:
    """Create a DataLoader wrapping the GWASDataset.

    Uses num_workers=0 for MPS compatibility. Batching is handled
    inside the dataset, so batch_size=None here.
    """
    dataset = GWASDataset(chromosomes, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=None,  # dataset yields pre-batched tensors
        num_workers=0,
        pin_memory=False,  # not needed for MPS
    )


def count_batches(chromosomes: list[int]) -> int:
    """Estimate total batches for a set of chromosomes."""
    total_rows = 0
    for chrom in chromosomes:
        p = config.DATA_PROCESSED_DIR / f"chr{chrom}_normalized.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=["logOR"])
            total_rows += len(df)
    return (total_rows + config.BATCH_SIZE - 1) // config.BATCH_SIZE
