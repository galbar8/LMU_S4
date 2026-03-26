from __future__ import annotations
from typing import Tuple, Dict, Any

from torch.utils.data import DataLoader

from src.datasets.etts.etts_dataloader import make_etts_loaders
from src.types.task_protocol import TaskProtocol


class ETTSTask(TaskProtocol):
    """ETTS forecasting task (regression)."""
    problem_type: str = "regression"

    def make_loaders(
        self,
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for the ETTS dataset."""
        return make_etts_loaders(
            data_root=data_root,
            which=kwargs.get("which", "ETTh1"),
            batch_size=batch_size,
            num_workers=num_workers,
            seq_len=kwargs.get("seq_len", 96),
            pred_len=kwargs.get("pred_len", 24),
            feature_mode=kwargs.get("feature_mode", "target"),
            target_col=kwargs.get("target_col", "OT"),
            split_ratio=kwargs.get("split_ratio", (0.7, 0.1, 0.2)),
            normalize=kwargs.get("normalize", "zscore"),
            pin_memory=kwargs.get("pin_memory", True),
            persistent_workers=kwargs.get("persistent_workers", False),
        )

    def infer_input_dim(self, args: Dict[str, Any]) -> int:
        """Infer input dimension based on feature mode."""
        fm = args.get("feature_mode", "target")
        return 1 if fm == "target_only" else 7

    def infer_num_classes(self, args: Dict[str, Any]) -> int:
        """Infer number of classes based on feature mode."""
        return 7 if args.get("feature_mode", "target") == "multivariate" else 1

    def infer_theta(self, args: Dict[str, Any]) -> int:
        """Infer sequence length."""
        return args.get("seq_len", 96)

