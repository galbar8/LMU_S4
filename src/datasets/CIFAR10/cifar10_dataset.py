"""Sequential CIFAR-10 dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets

from .cifar10_config import CIFAR10Config


class CIFAR10Dataset(Dataset):
    """
    Sequential CIFAR-10 dataset.

    Each CIFAR-10 image (32×32×3) is:
    1. Converted to numpy float32 array (H, W, C)
    2. Optionally scaled/normalized
    3. Reshaped into sequence of length 1024 with feature dim 3: (1024, 3)
    4. Optionally permuted using a fixed permutation over the 1024 spatial positions

    Returns:
        x: FloatTensor (1024, 3) - pixel sequence
        y: LongTensor - class label (0-9)
    """

    def __init__(self, cfg: CIFAR10Config):
        super().__init__()
        self.cfg = cfg

        root = Path(cfg.root)
        is_train = (cfg.split == "train")

        self.ds = datasets.CIFAR10(
            root=str(root),
            train=is_train,
            download=cfg.download,
            transform=None,  # we apply transforms manually
        )

        # Fixed permutation over spatial positions (1024)
        self.permutation = None
        if cfg.permute:
            rng = np.random.RandomState(cfg.permutation_seed)
            self.permutation = rng.permutation(32 * 32)

        # CIFAR-10 standard normalization stats (channel-wise)
        # Values are common defaults used in many repos.
        # Mean/std are in [0,1] scale (after dividing by 255).
        if cfg.normalize == "standard":
            self.mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
            self.std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
            self._expects_unit_range = True
        elif cfg.normalize == "minmax":
            self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            self._expects_unit_range = True
        else:  # "none"
            self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            self._expects_unit_range = False

        # Handle subset
        self.length = len(self.ds)
        if cfg.subset_size is not None and cfg.subset_size < self.length:
            self.length = cfg.subset_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        """
        Returns:
            x: (1024, 3) - pixel sequence as (seq_len, features)
            y: scalar - class label (0-9)
        """
        img, label = self.ds[idx]  # img is PIL.Image

        # PIL -> numpy float32, shape (H, W, C)
        img_np = np.array(img, dtype=np.float32)  # (32, 32, 3), values in [0,255]

        # If normalization expects [0,1], scale first
        if self._expects_unit_range:
            img_np = img_np / 255.0

        # Reshape to sequence: (H*W, C) = (1024, 3)
        seq = img_np.reshape(-1, 3)

        if self.permutation is not None:
            seq = seq[self.permutation]

        # Normalize (channel-wise)
        # Broadcasting: (1024,3) - (3,) / (3,)
        seq = (seq - self.mean) / self.std

        x = torch.from_numpy(seq)  # (1024, 3)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def seq_len(self) -> int:
        return 32 * 32  # 1024

    @property
    def input_dim(self) -> int:
        return 3  # RGB
