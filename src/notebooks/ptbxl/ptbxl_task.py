"""PTB-XL ECG Classification Task."""
from __future__ import annotations
from typing import Tuple, Dict, Any

from torch.utils.data import DataLoader

from src.types.task_protocol import TaskProtocol
from src.datasets.ptbxl.ptbxl_dataloader import make_ptbxl_loaders


class PTBXLTask(TaskProtocol):
    """
    PTB-XL ECG classification task.
    - Multi-label classification (5 superclasses: NORM, MI, STTC, HYP, CD)
    - Input: 12-lead ECG signals
    - Sampling rate: 100Hz (lr100) or 500Hz (hr500)
    """
    problem_type: str = "multilabel"

    def make_loaders(
        self,
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns: (train_loader, val_loader, test_loader)
        with x: (B, T, D), y: (B, 5) for multi-label classification
        """
        sampling = kwargs.get("sampling", "lr100")
        length = kwargs.get("length", 1000)
        leads = kwargs.get("leads", None)
        folds_train = kwargs.get("folds_train", (1, 2, 3, 4, 5, 6, 7, 8))
        fold_val = kwargs.get("fold_val", 9)
        fold_test = kwargs.get("fold_test", 10)

        return make_ptbxl_loaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            sampling=sampling,
            length=length,
            leads=leads,
            folds_train=folds_train,
            fold_val=fold_val,
            fold_test=fold_test,
        )

    def infer_input_dim(self, args: Dict[str, Any]) -> int:
        """Number of leads (channels)"""
        leads = args.get("leads", None)
        return len(leads) if leads is not None else 12

    def infer_num_classes(self, args: Dict[str, Any]) -> int:
        """Number of superclasses for multi-label classification"""
        return 5  # NORM, MI, STTC, HYP, CD

    def infer_theta(self, args: Dict[str, Any]) -> int:
        """Sequence length (number of time steps)"""
        return args.get("length", 1000)

