"""Sequential CIFAR-10 configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class CIFAR10Config:
    """
    Configuration for Sequential CIFAR-10 dataset.

    CIFAR-10 images (32×32×3) are treated as sequences:
    - Each spatial position (pixel) is a timestep (seq_len = 32*32 = 1024)
    - Feature dim is 3 (RGB)
    - Optional fixed permutation over spatial positions for all samples
    - Task: classify image into 10 classes

    Notes:
    - Normalization can be "standard" (dataset mean/std), "minmax", or "none".
    - Permutation is applied on the flattened spatial dimension (1024 positions),
      while keeping RGB channels as features.
    """

    # Data location
    root: str  # folder to download/store CIFAR-10 data

    # Split
    split: Literal["train", "test"] = "train"

    # Preprocessing
    normalize: Literal["standard", "minmax", "none"] = "standard"
    # standard: use CIFAR-10 channel-wise mean/std
    # minmax: scale to [0, 1]
    # none: keep original [0, 255] values (as float)

    # Sequence formatting
    sequence_format: Literal["pixel"] = "pixel"
    # currently only "pixel": seq_len = 1024, input_dim = 3

    # Optional: permutation over spatial positions (like PS-MNIST conceptually)
    permute: bool = False
    permutation_seed: int = 42  # seed for generating fixed permutation when permute=True

    # Optional: use a subset for faster experimentation
    subset_size: Optional[int] = None  # if set, use only first N samples

    # Download
    download: bool = True  # whether to download CIFAR-10 if not present
