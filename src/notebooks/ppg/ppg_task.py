"""PPG Heart Rate Estimation Task."""
from __future__ import annotations
from typing import Tuple, Dict, Any

from torch.utils.data import DataLoader

from src.types.task_protocol import TaskProtocol
from src.datasets.ppg.ppg_config import PPGDaliaConfig
from src.datasets.ppg.ppg_dataloader import make_ppgdalia_loaders


class PPGTask(TaskProtocol):
    """PPG-based heart rate estimation (regression)."""
    problem_type: str = "regression"

    def make_loaders(
        self,
        data_root: str,
        batch_size: int = 96,
        num_workers: int = 0,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders using PPGDaliaConfig."""
        pin_memory = kwargs.pop("pin_memory", False)
        persistent_workers = kwargs.pop("persistent_workers", False)

        cfg = PPGDaliaConfig(root=data_root, **kwargs)

        return make_ppgdalia_loaders(
            cfg,
            batch=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def infer_input_dim(self, args: Dict[str, Any]) -> int:
        """Single PPG channel + X, Y, Z acc position ."""
        return 4

    def infer_num_classes(self, args: Dict[str, Any]) -> int:
        """Single HR output."""
        return 1

    def infer_theta(self, args: Dict[str, Any]) -> int:
        """Sequence length = window size in samples."""
        win_sec = args.get("win_sec", 8)
        fs = args.get("fs", 100.0)
        return int(win_sec * fs)

