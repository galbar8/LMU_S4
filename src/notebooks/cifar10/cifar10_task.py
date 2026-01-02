from __future__ import annotations
from typing import Tuple, Dict, Any

from torch.utils.data import DataLoader

from src.datasets.CIFAR10.cifar10_dataloader import make_cifar10_loaders
from src.types.task_protocol import TaskProtocol

class CIFAR10Task(TaskProtocol):
    """
    Sequential CIFAR-10 classification task.

    - 10-class image classification
    - Input: sequences derived from 32×32 RGB images
      Default sequence representation:
        * seq_len = 1024 (32×32 spatial positions)
        * input_dim = 3 (RGB channels as features)
    - Optional: fixed permutation over spatial positions (like PS-MNIST conceptually)

    Note:
    CIFAR-10 typically does not ship with an official validation split.
    We follow the PS-MNIST convention here and return the test loader twice
    (val and test are the same), unless you later add an explicit val split.
    """
    problem_type: str = "multiclass"

    def make_loaders(
        self,
        data_root: str,
        batch_size: int = 64,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns: (train_loader, test_loader, test_loader)

        Note:
        CIFAR-10 has no standard validation set by default,
        so we return test_loader twice (val and test are the same).
        """
        normalize = kwargs.get("normalize", "standard")
        subset_size = kwargs.get("subset_size", None)
        download = kwargs.get("download", True)
        pin_memory = kwargs.get("pin_memory", False)
        persistent_workers = kwargs.get("persistent_workers", False)
        num_workers = kwargs.get("num_workers", 0)

        # Optional permutation support (mirrors PS-MNIST "fixed permutation" concept)
        permute = kwargs.get("permute", False)
        permutation_seed = kwargs.get("permutation_seed", 42)

        train_loader, test_loader = make_cifar10_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            subset_size=subset_size,
            download=download,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            permute=permute,
            permutation_seed=permutation_seed,
        )

        return train_loader, test_loader, test_loader

    def infer_input_dim(self, args: Dict[str, Any]) -> int:
        """
        Input feature dimension:
        - Default sequential CIFAR-10 representation uses RGB features per timestep => 3.
        """
        return 3

    def infer_num_classes(self, args: Dict[str, Any]) -> int:
        """Number of CIFAR-10 classes."""
        return 10

    def infer_theta(self, args: Dict[str, Any]) -> int:
        """
        Sequence length:
        - Default sequential CIFAR-10 representation => 32×32 = 1024 timesteps.
        """
        return 32 * 32
