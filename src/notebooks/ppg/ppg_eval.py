"""PPG evaluation utilities."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.amp import autocast as amp_autocast

from src.models.v2.build_model import build_model
from src.types.task_protocol import TaskProtocol

def load_best_model(
    args: dict, task: TaskProtocol, model_builder: callable, best_model_path: str) -> torch.nn.Module:
    """
    Load the best checkpoint and rebuild model.
    Args:
        args: Training arguments dictionary
        task: Task protocol instance
        model_builder: Function to build the model
        best_model_path: Path to the best model checkpoint
    Returns:
        model: Loaded model
    """
    device = args.get("device", torch.device("cpu"))
    checkpoint = torch.load(best_model_path, map_location=device)
    flat_args = dict(args)
    flat_args.update(args.get("data_loader_kwargs", {}))
    d_in = task.infer_input_dim(flat_args)
    n_classes = task.infer_num_classes(flat_args)
    theta = task.infer_theta(flat_args)

    block_cfg = args["block_cfg_ctor"](theta)
    model = model_builder(
        d_in=d_in,
        n_classes=n_classes,
        d_model=args["d_model"],
        depth=args["depth"],
        block_cfg=block_cfg
    ).to(device)

    model.load_state_dict(checkpoint["model"])

    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")

    return model



def evaluate_best_model(
    args: dict,
    task: TaskProtocol,
    best_model_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the best checkpoint and evaluate on test set.

    Args:
        args: Training arguments dictionary
        task: Task protocol instance
        best_model_path: Path to the best model checkpoint

    Returns:
        (predictions, targets) as numpy arrays
    """
    print("📊 Evaluating best model on the test set...")

    device = args.get("device", torch.device("cpu"))
    checkpoint = torch.load(best_model_path, map_location=device)

    # 1. Test loader
    _, _, test_loader = task.make_loaders(
        data_root=args["data_root"],
        batch_size=args["batch"],
        **args["data_loader_kwargs"]
    )

    model = load_best_model(args, task, build_model, best_model_path)
    model.eval()

    val_metrics = checkpoint.get('val', {})
    print(f"📈 Val MSE: {val_metrics.get('mse', 'N/A'):.4f}, Val MAE: {val_metrics.get('mae', 'N/A'):.4f}")

    # 3. Evaluation loop
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y, _ in tqdm(test_loader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            # Disable AMP for evaluation
            with amp_autocast(device_type=device.type, enabled=False):
                out = model(x)
            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # 4. Metrics
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    test_mse = np.mean((all_preds - all_targets) ** 2)
    test_mae = np.mean(np.abs(all_preds - all_targets))
    test_rmse = np.sqrt(test_mse)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS:")
    print("=" * 50)
    print(f"MSE:  {test_mse:.4f}")
    print(f"MAE:  {test_mae:.4f} bpm")
    print(f"RMSE: {test_rmse:.4f} bpm")

    return all_preds, all_targets

