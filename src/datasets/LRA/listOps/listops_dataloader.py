"""LRA ListOps dataloader factory."""
from __future__ import annotations
from typing import Tuple, Optional

from torch.utils.data import DataLoader

from .listops_config import ListOpsConfig
from .listops_dataset import ListOpsDataset


def make_listops_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 0,
    max_length: int = 2000,
    subset_size: Optional[int] = None,
    vocab_path: Optional[str] = None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for LRA ListOps.

    Assumes files exist under:
        <root>/data/basic_train.tsv
        <root>/data/basic_val.tsv
        <root>/data/basic_test.tsv

    Returns:
        train_loader, val_loader, test_loader
    """

    train_cfg = ListOpsConfig(
        root=root,
        split="train",
        max_length=max_length,
        subset_size=subset_size,
        vocab_path=vocab_path,
    )

    val_cfg = ListOpsConfig(
        root=root,
        split="val",
        max_length=max_length,
        subset_size=None,
        vocab_path=vocab_path,
    )

    test_cfg = ListOpsConfig(
        root=root,
        split="test",
        max_length=max_length,
        subset_size=None,
        vocab_path=vocab_path,
    )

    train_dataset = ListOpsDataset(train_cfg)
    val_dataset = ListOpsDataset(val_cfg)
    test_dataset = ListOpsDataset(test_cfg)

    pw = persistent_workers if num_workers > 0 else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        drop_last=False,
    )

    print("ListOps Loaders Created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Max length: {max_length}")
    print(f"  Vocab size: {train_dataset.vocab_size}")
    print("  Classes: 10 (0-9)")

    return train_loader, val_loader, test_loader
