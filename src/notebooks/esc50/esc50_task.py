"""ESC-50 environmental sound classification task."""
from __future__ import annotations
from typing import Tuple, Dict, Any

from torch.utils.data import DataLoader

from src.types.task_protocol import TaskProtocol
from src.datasets.esc50.esc50_dataset import make_esc50_loaders


class ESC50Task(TaskProtocol):
    """
    ESC-50 environmental sound classification task.
    - 50-class classification
    - Input: Audio spectrograms (mel-spectrograms)
    - 5-fold cross-validation
    """
    problem_type: str = "multiclass"

    def make_loaders(
        self,
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns: (train_loader, val_loader, test_loader)
        with x: (B, T, D), y: (B,) for classification
        """
        fold_val = kwargs.get("fold_val", 1)
        feature = kwargs.get("feature", "melspec")
        sample_rate = kwargs.get("sample_rate", 16000)
        n_mels = kwargs.get("n_mels", 128)
        hop_length = kwargs.get("hop_length", 160)
        n_fft = kwargs.get("n_fft", 512)
        target_num_frames = kwargs.get("target_num_frames", 500)
        augment = kwargs.get("augment", True)

        loaders = make_esc50_loaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            fold_val=fold_val,
            feature=feature,
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            target_num_frames=target_num_frames,
            augment=augment,
            download=False
        )

        # Handle both 2-tuple and 3-tuple returns
        if isinstance(loaders, (list, tuple)) and len(loaders) == 3:
            return loaders
        else:
            train_loader, val_loader = loaders
            # Use val_loader as test_loader for ESC-50 (single validation fold)
            return train_loader, val_loader, val_loader

    def infer_input_dim(self, args: Dict[str, Any]) -> int:
        """Number of mel bins or other features"""
        feature = args.get("feature", "melspec")
        if feature == "melspec":
            return args.get("n_mels", 128)
        return 1

    def infer_num_classes(self, args: Dict[str, Any]) -> int:
        """Number of classes for ESC-50"""
        return 50

    def infer_theta(self, args: Dict[str, Any]) -> int:
        """Sequence length (number of time frames)"""
        return args.get("target_num_frames", 500)

