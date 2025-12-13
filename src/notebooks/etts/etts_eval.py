from __future__ import annotations
from typing import Tuple, Dict

import torch
import numpy as np

from tqdm.auto import tqdm
from torch.amp import autocast as amp_autocast

from src.models.v2.build_model import build_model
from src.types.task_protocol import TaskProtocol
from src.utils.checkpoint import save_test_results


def evaluate_best_model(
    args: dict,
    task: TaskProtocol,
    best_model_path: str,
    save_results: bool = True
) -> Tuple[any, any, any]:
    """
    Load the best checkpoint and evaluate on the test set.

    Args:
        args: Configuration dictionary with data_root, batch, device, etc.
        task: TaskProtocol instance for creating data loaders and inferring dimensions
        best_model_path: Path to the best checkpoint file
        save_results: Whether to save test results to JSON file (default: True)

    Returns:
        Tuple of test_mae, test_mse, test_rmse
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
    model = build_model(
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

    # Save test results to file
    if save_results:
        test_results = {
            'mse': float(test_mse),
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'num_samples': int(all_preds.shape[0])
        }
        results_path = save_test_results(best_model_path, test_results)
        print("=" * 50)
        print(f"✅ Test results saved to: {results_path}")

    return test_mae, test_mse, test_rmse



