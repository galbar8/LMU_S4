"""Sequential CIFAR-10 dataloader factory."""
from __future__ import annotations

from typing import Tuple, Optional

from torch.utils.data import DataLoader

from .cifar10_config import CIFAR10Config
from .cifar10_dataset import CIFAR10Dataset


def make_cifar10_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 0,
    normalize: str = "standard",
    subset_size: Optional[int] = None,
    download: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    permute: bool = False,
    permutation_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for Sequential CIFAR-10.

    Args:
        root: Directory to download/store CIFAR-10 data
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        normalize: Normalization method ("standard", "minmax", "none")
        subset_size: Optional subset size for faster experimentation (train only)
        download: Whether to download CIFAR-10 if not present
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        permute: Whether to apply a fixed permutation over spatial positions (1024)
        permutation_seed: Seed for generating fixed permutation when permute=True

    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
    """

    train_cfg = CIFAR10Config(
        root=root,
        split="train",
        normalize=normalize,  # type: ignore[arg-type]
        subset_size=subset_size,
        download=download,
        permute=permute,
        permutation_seed=permutation_seed,
    )

    test_cfg = CIFAR10Config(
        root=root,
        split="test",
        normalize=normalize,  # type: ignore[arg-type]
        subset_size=None,
        download=download,
        permute=permute,  # same permutation behavior as train
        permutation_seed=permutation_seed,
    )

    train_dataset = CIFAR10Dataset(train_cfg)
    test_dataset = CIFAR10Dataset(test_cfg)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        drop_last=False,
    )

    print("CIFAR-10 Loaders Created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Sequence length: 1024 (32×32)")
    print(f"  Input dim: 3 (RGB)")
    print(f"  Classes: 10")
    if permute:
        print(f"  Permutation: enabled (seed={permutation_seed})")
    else:
        print("  Permutation: disabled")

    return train_loader, test_loader
