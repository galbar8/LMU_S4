from __future__ import annotations
from typing import Tuple, Dict, Any, Callable

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.amp import autocast as amp_autocast

from src.datasets.etts.etts_dataloader import make_etts_loaders
from src.models.v2.build_model import BlockConfig, build_model
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


def make_block_cfg_ctor(
    kind: str,
    *,
    # Common
    dropout: float,
    mlp_ratio: float,
    droppath_final: float,
    layerscale_init: float,
    residual_gain: float,
    pool: str,
    # LMU-specific
    memory_size: int = 256,
    # S4-specific
    d_state: int = 64,
    channels: int = 1,
    bidirectional: bool = False,
    mode: str = "s4d",
    dt_min: float = 0.001,
    dt_max: float = 0.1,
):
    """
    Create a block config constructor. Pass kind="lmu" or "s4".
    This allows for easy comparison between LMU and S4 models by changing a single argument.
    """
    def block_cfg_ctor(theta: int) -> BlockConfig:
        if kind.lower() == "lmu":
            return BlockConfig(
                kind="lmu",
                memory_size=memory_size,
                theta=theta,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                droppath_final=droppath_final,
                layerscale_init=layerscale_init,
                residual_gain=residual_gain,
                pool=pool,
            )
        elif kind.lower() == "s4":
            return BlockConfig(
                kind="s4",
                d_state=d_state,
                channels=channels,
                bidirectional=bidirectional,
                mode=mode,
                dt_min=dt_min,
                dt_max=dt_max,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                droppath_final=droppath_final,
                layerscale_init=layerscale_init,
                residual_gain=residual_gain,
                pool=pool,
            )
        else:
            raise ValueError(f"Unknown block kind: {kind}")
    return block_cfg_ctor