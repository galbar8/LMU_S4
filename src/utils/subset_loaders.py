from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def reduce_loader_size(
    loader: DataLoader,
    fraction: float,
    seed: int = 42,
    stratified: bool = True,
) -> DataLoader:
    """
    Create a new DataLoader with a subset of the original dataset.

    Args:
        loader: Original DataLoader
        fraction: Fraction of data to keep (0.0 to 1.0)
        seed: Random seed for reproducibility
        stratified: Whether to use stratified sampling (maintains class balance)

    Returns:
        New DataLoader with reduced dataset size
    """
    if fraction >= 1.0:
        return loader

    if fraction <= 0.0:
        raise ValueError(f"fraction must be > 0, got {fraction}")

    dataset = loader.dataset
    original_size = len(dataset)
    subset_size = max(1, int(original_size * fraction))

    if stratified:
        # Extract labels for stratification
        labels = []
        for i in range(original_size):
            try:
                # Try different ways to get labels
                item = dataset[i]
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    label = item[1]
                else:
                    label = item

                if isinstance(label, torch.Tensor):
                    label = label.item() if label.numel() == 1 else label.argmax().item()
                labels.append(label)
            except Exception:
                # Fallback to random sampling if we can't extract labels
                stratified = False
                break

        if stratified:
            labels = np.array(labels)
            unique_classes = np.unique(labels)

            # Check if this is a regression task (too many unique values)
            # If more than 20% of samples have unique labels, treat as regression
            if len(unique_classes) > original_size * 0.2:
                # Regression task - use simple random sampling
                stratified = False
            else:
                # Classification task - use stratified sampling
                rng = np.random.RandomState(seed)
                subset_indices = []

                for cls in unique_classes:
                    cls_indices = np.where(labels == cls)[0]
                    cls_subset_size = max(1, int(len(cls_indices) * fraction))
                    cls_subset_indices = rng.choice(cls_indices, size=cls_subset_size, replace=False)
                    subset_indices.extend(cls_subset_indices)

                # Shuffle the combined indices
                rng.shuffle(subset_indices)
                subset_indices = list(subset_indices)

    if not stratified:
        # Random sampling (for regression or when stratification fails)
        rng = np.random.RandomState(seed)
        subset_indices = rng.choice(original_size, size=subset_size, replace=False).tolist()

    # Create subset dataset
    subset_dataset = Subset(dataset, subset_indices)

    # Create new loader with same parameters
    new_loader = DataLoader(
        subset_dataset,
        batch_size=loader.batch_size,
        shuffle=isinstance(loader.sampler, torch.utils.data.RandomSampler) if hasattr(loader, 'sampler') else False,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=loader.collate_fn,
    )

    return new_loader


def apply_fraction_to_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    fraction: float,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Apply fraction reduction to train and validation loaders.

    Note: Test loader is typically kept at full size for proper evaluation.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        fraction: Fraction of data to keep (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (reduced_train, reduced_val, full_test)
    """
    if fraction >= 1.0:
        return train_loader, val_loader, test_loader

    print(f"   Applying data fraction: {fraction:.1%}")
    print(f"   Original train size: {len(train_loader.dataset)}")
    print(f"   Original val size: {len(val_loader.dataset)}")

    # Reduce train and val loaders
    reduced_train = reduce_loader_size(train_loader, fraction, seed=seed, stratified=True)
    reduced_val = val_loader #reduce_loader_size(val_loader, fraction, seed=seed + 1, stratified=True)

    print(f"   Reduced train size: {len(reduced_train.dataset)}")
    print(f"   Reduced val size: {len(reduced_val.dataset)}")
    print(f"   Test size: {len(test_loader.dataset)} (unchanged)")

    return reduced_train, reduced_val, test_loader

