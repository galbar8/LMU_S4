from __future__ import annotations
from typing import Tuple, Dict, Any

from torch.utils.data import DataLoader

from src.types.task_protocol import TaskProtocol
from src.datasets.psmnist.psmnist_dataloader import make_psmnist_loaders

class PSMNISTTask(TaskProtocol):
    """
    Permuted Sequential MNIST classification task.

    - 10-class digit classification (0-9)
    - Input: Permuted sequences of 784 pixels (28×28 MNIST images)
    - Challenge: Long-range dependencies with scrambled pixel order

    This is a classic benchmark for sequence models, testing:
    - Ability to handle very long sequences (784 timesteps)
    - Learning from unstructured/scrambled inputs
    - Memory and long-range dependency modeling
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

        Note: PS-MNIST traditionally doesn't have a separate validation set,
        so we return test_loader twice (val and test are the same).

        For proper validation during training, you could split train set
        or use test set for both validation and final evaluation.
        """
        permutation_seed = kwargs.get("permutation_seed", 42)
        normalize = kwargs.get("normalize", "standard")
        subset_size = kwargs.get("subset_size", None)
        fraction = kwargs.get("fraction", 1.0)
        download = kwargs.get("download", True)
        pin_memory = kwargs.get("pin_memory", False)
        persistent_workers = kwargs.get("persistent_workers", False)
        num_workers = kwargs.get("num_workers", 0)

        # Convert fraction to subset_size if specified
        if fraction < 1.0 and subset_size is None:
            # MNIST train has 60000 samples
            subset_size = int(60000 * fraction)

        train_loader, test_loader = make_psmnist_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            permutation_seed=permutation_seed,
            normalize=normalize,
            subset_size=subset_size,
            download=download,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        # Return test_loader as both val and test
        # (PS-MNIST traditionally uses test set for monitoring)
        return train_loader, test_loader, test_loader

    def infer_input_dim(self, args: Dict[str, Any]) -> int:
        """Input feature dimension (1 for grayscale pixels)"""
        return 1

    def infer_num_classes(self, args: Dict[str, Any]) -> int:
        """Number of digit classes"""
        return 10

    def infer_theta(self, args: Dict[str, Any]) -> int:
        """Sequence length (784 = 28×28 flattened MNIST)"""
        return 784

