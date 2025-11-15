"""Shared utilities for ETTS notebooks."""
from __future__ import annotations
from typing import Tuple, Dict, Any, Callable

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.amp import autocast as amp_autocast

from src.datasets.etts.etts_dataloader import make_etts_loaders
from src.models.v2.build_model import BlockConfig
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


def evaluate_best_model(
    args: dict,
    task: TaskProtocol,
    model_builder: Callable,
    best_model_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the best checkpoint and evaluate on the test set.

    Args:
        args: Configuration dictionary with data_root, batch, device, etc.
        task: TaskProtocol instance for creating data loaders and inferring dimensions
        model_builder: Function to build the model (e.g., build_model)
        best_model_path: Path to the best checkpoint file

    Returns:
        Tuple of (predictions, targets) as numpy arrays
    """
    print("Evaluating best model on the test set...")

    # 1. Create test data loader
    _, _, test_loader = task.make_loaders(
        data_root=args["data_root"],
        batch_size=args["batch"],
        **args["data_loader_kwargs"]
    )

    # 2. Load checkpoint and rebuild model
    device = args.get("device", torch.device("cpu"))
    checkpoint = torch.load(best_model_path, map_location=device)

    # Infer model dimensions from task and args
    flat_args = dict(args)
    flat_args.update(args.get("data_loader_kwargs", {}))
    d_in = task.infer_input_dim(flat_args)
    d_out = task.infer_num_classes(flat_args)
    theta = task.infer_theta(flat_args)
    pred_len = flat_args.get("pred_len", 24)

    # Re-create model architecture
    block_cfg = args["block_cfg_ctor"](theta)
    model = model_builder(
        d_in=d_in,
        n_classes=d_out * pred_len,
        d_model=args["d_model"],
        depth=args["depth"],
        block_cfg=block_cfg
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    val_metrics = checkpoint.get('val', {})
    print(f"Validation MSE: {val_metrics.get('mse', 'N/A'):.6f}, MAE: {val_metrics.get('mae', 'N/A'):.6f}")

    # 3. Evaluation loop
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            # Disable AMP for evaluation to prevent FFT/precision issues
            with amp_autocast(device_type=device.type, enabled=False):
                out = model(x).view(x.size(0), pred_len, d_out)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # 4. Calculate and print metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    test_mse = np.mean((all_preds - all_targets) ** 2)
    test_mae = np.mean(np.abs(all_preds - all_targets))
    test_rmse = np.sqrt(test_mse)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS:")
    print("=" * 50)
    print(f"MSE:  {test_mse:.6f}")
    print(f"MAE:  {test_mae:.6f}")
    print(f"RMSE: {test_rmse:.6f}")

    return all_preds, all_targets



