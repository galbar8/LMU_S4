"""Permuted Sequential MNIST dataset."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

from .psmnist_config import PSMNISTConfig


class PSMNISTDataset(Dataset):
    """
    Permuted Sequential MNIST dataset.

    Each MNIST image (28×28) is:
    1. Flattened to a sequence of 784 pixels
    2. Permuted using a fixed random permutation
    3. Normalized according to config

    Returns:
        x: FloatTensor (784, 1) - permuted pixel sequence
        y: LongTensor - digit label (0-9)

    This is a classic benchmark for testing sequence models' ability to:
    - Handle long-range dependencies (784 timesteps)
    - Learn from scrambled/unstructured sequences
    """

    def __init__(self, cfg: PSMNISTConfig):
        super().__init__()
        self.cfg = cfg

        # Load MNIST dataset
        root = Path(cfg.root)
        is_train = (cfg.split == "train")

        self.mnist = datasets.MNIST(
            root=str(root),
            train=is_train,
            download=cfg.download,
            transform=None  # We'll apply transforms manually
        )

        if cfg.use_permutation:
            rng = np.random.RandomState(cfg.permutation_seed)
            self.permutation = rng.permutation(784)
        else:
            self.permutation = np.arange(784)

        # Normalization parameters (MNIST statistics)
        if cfg.normalize == "standard":
            self.mean = 0.1307
            self.std = 0.3081
        elif cfg.normalize == "minmax":
            self.mean = 0.0
            self.std = 255.0
        else:  # none
            self.mean = 0.0
            self.std = 1.0

        # Handle subset
        self.length = len(self.mnist)
        if cfg.subset_size is not None and cfg.subset_size < self.length:
            self.length = cfg.subset_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        """
        Returns:
            x: (784, 1) - permuted pixel sequence as (seq_len, features)
            y: scalar - digit label (0-9)
        """
        # Get MNIST image and label
        img, label = self.mnist[idx]

        # Convert PIL image to numpy array
        img_np = np.array(img, dtype=np.float32)  # (28, 28)

        # Flatten to sequence
        seq = img_np.reshape(-1)  # (784,)

        # Apply permutation
        seq_permuted = seq[self.permutation]  # (784,)

        # Normalize
        seq_normalized = (seq_permuted - self.mean) / self.std

        # Convert to tensor and add feature dimension
        x = torch.from_numpy(seq_normalized).unsqueeze(-1)  # (784, 1)
        y = torch.tensor(label, dtype=torch.long)

        return x, y

    @property
    def num_classes(self) -> int:
        """Number of digit classes."""
        return 10

    @property
    def seq_len(self) -> int:
        """Sequence length (28 * 28 = 784)."""
        return 784

    @property
    def input_dim(self) -> int:
        """Input feature dimension (1 for grayscale pixels)."""
        return 1

