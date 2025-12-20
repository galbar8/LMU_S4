"""Permuted Sequential MNIST dataloader factory."""
from __future__ import annotations
from typing import Tuple

from torch.utils.data import DataLoader

from .psmnist_config import PSMNISTConfig
from .psmnist_dataset import PSMNISTDataset


def make_psmnist_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 0,
    permutation_seed: int = 42,
    normalize: str = "standard",
    subset_size: int = None,
    download: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for Permuted Sequential MNIST.

    Args:
        root: Directory to download/store MNIST data
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        permutation_seed: Seed for generating fixed pixel permutation
        normalize: Normalization method ("standard", "minmax", "none")
        subset_size: Optional subset size for faster experimentation
        download: Whether to download MNIST if not present
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep workers alive between epochs

    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
    """

    # Create train config
    train_cfg = PSMNISTConfig(
        root=root,
        permutation_seed=permutation_seed,
        normalize=normalize,
        split="train",
        subset_size=subset_size,
        download=download,
    )

    # Create test config (same permutation!)
    test_cfg = PSMNISTConfig(
        root=root,
        permutation_seed=permutation_seed,
        normalize=normalize,
        split="test",
        subset_size=None,
        download=download,
    )

    # Create datasets
    train_dataset = PSMNISTDataset(train_cfg)
    test_dataset = PSMNISTDataset(test_cfg)

    # Handle persistent_workers (only if num_workers > 0)
    pw = persistent_workers if num_workers > 0 else False

    # Create dataloaders
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
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        drop_last=False,
    )

    print(f"PS-MNIST Loaders Created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Permutation seed: {permutation_seed}")
    print(f"  Sequence length: 784 (28×28)")
    print(f"  Classes: 10 (digits 0-9)")

    return train_loader, test_loader