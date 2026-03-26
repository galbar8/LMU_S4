"""Permuted Sequential MNIST configuration."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class PSMNISTConfig:
    """
    Configuration for Permuted Sequential MNIST dataset.

    PS-MNIST is a classic sequence modeling benchmark where:
    - MNIST digits (28×28 = 784 pixels) are treated as sequences
    - A fixed random permutation is applied to pixel order
    - Task: classify digit (0-9) after seeing the entire sequence
    """

    # Data location
    root: str  # folder to download/store MNIST data

    # Permutation
    use_permutation: bool = True  # whether to apply permutation (False = SMNIST, True = PSMNIST)
    permutation_seed: int = 42  # seed for generating fixed permutation (only used if use_permutation=True)

    # Preprocessing
    normalize: Literal["standard", "minmax", "none"] = "standard"
    # standard: mean=0.1307, std=0.3081 (MNIST stats)
    # minmax: scale to [0, 1]
    # none: keep original [0, 255] values

    # Split
    split: Literal["train", "test"] = "train"

    # Optional: use a subset for faster experimentation
    subset_size: Optional[int] = None  # if set, use only first N samples

    # Download
    download: bool = True  # whether to download MNIST if not present

