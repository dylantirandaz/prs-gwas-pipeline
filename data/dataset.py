import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

TENSOR_COLUMNS = (
    config.NUMERIC_FEATURES
    + ["chr_idx", "a1_idx", "a2_idx", "logOR", "SE"]
)


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
    ds = load_from_disk(str(config.DATASET_PATH))
    split_ds = ds[split].with_format("torch", columns=TENSOR_COLUMNS)
    return DataLoader(
        split_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=gwas_collate,
        drop_last=False,
    )


def count_batches(split: str) -> int:
    ds = load_from_disk(str(config.DATASET_PATH))
    n = len(ds[split])
    return (n + config.BATCH_SIZE - 1) // config.BATCH_SIZE
