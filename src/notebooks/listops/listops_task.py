"""ListOps Task for LMU Training."""
from __future__ import annotations
from typing import Tuple, Dict, Any

from torch.utils.data import DataLoader

from src.types.task_protocol import TaskProtocol
from src.datasets.LRA.listOps.listops_dataloader import make_listops_loaders
from src.datasets.LRA.listOps.listops_dataset import ListOpsDataset
from src.datasets.LRA.listOps.listops_config import ListOpsConfig

class ListOpsTask(TaskProtocol):
    """
    ListOps classification task from Long Range Arena (LRA).

    - 10-class classification (0-9) based on nested arithmetic expressions
    - Input: Character-level sequences of up to 2000 tokens
    - Challenge: Long-range dependencies in hierarchical expressions

    This benchmark tests:
    - Ability to handle very long sequences (up to 2000 timesteps)
    - Understanding of hierarchical structure
    - Long-range dependency modeling across nested expressions
    """
    problem_type: str = "multiclass"

    def make_loaders(
        self,
        data_root: str,
        batch_size: int = 50,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test dataloaders for ListOps.

        Returns: (train_loader, val_loader, test_loader)

        Args:
            data_root: Path to the dataset root directory
            batch_size: Batch size for training
            **kwargs: Additional arguments including:
                - max_length: Maximum sequence length (default: 2000)
                - subset_size: Optional subset size for faster experimentation
                - vocab_path: Optional path to vocabulary file
                - num_workers: Number of data loading workers (default: 0)
                - pin_memory: Whether to pin memory (default: True)
                - persistent_workers: Whether to use persistent workers (default: False)
        """
        max_length = kwargs.get("max_length", 2000)
        subset_size = kwargs.get("subset_size", None)
        vocab_path = kwargs.get("vocab_path", None)
        num_workers = kwargs.get("num_workers", 0)
        pin_memory = kwargs.get("pin_memory", True)
        persistent_workers = kwargs.get("persistent_workers", False)

        train_loader, val_loader, test_loader = make_listops_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
            subset_size=subset_size,
            vocab_path=vocab_path,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        return train_loader, val_loader, test_loader

    def infer_input_dim(self, args: Dict[str, Any]) -> int:
        """
        Input feature dimension.

        ListOps uses character-level tokenization with embedding layer.
        This is handled via get_vocab_size() method.
        For compatibility, return 1 but model will use embedding.
        """
        return 1

    @staticmethod
    def get_vocab_size(data_root: str, **kwargs) -> int:
        """
        Get the vocabulary size for the ListOps dataset.
        This is needed for the embedding layer.
        """

        max_length = kwargs.get("max_length", 2000)
        vocab_path = kwargs.get("vocab_path", None)

        temp_cfg = ListOpsConfig(
            root=data_root,
            split="train",
            max_length=max_length,
            vocab_path=vocab_path,
        )
        temp_dataset = ListOpsDataset(temp_cfg)
        return temp_dataset.vocab_size

    def infer_num_classes(self, args: Dict[str, Any]) -> int:
        """Number of output classes (digits 0-9)"""
        return 10

    def infer_theta(self, args: Dict[str, Any]) -> int:
        """
        Sequence length for ListOps.

        Default is 2000, which is the standard LRA ListOps sequence length.
        Can be overridden via max_length argument.
        """
        return args.get("max_length", 2000)

