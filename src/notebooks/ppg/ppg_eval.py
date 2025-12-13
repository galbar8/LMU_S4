"""PPG evaluation utilities."""
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from torch.amp import autocast as amp_autocast

from src.models.v2.build_model import build_model
from src.types.task_protocol import TaskProtocol
from src.utils.checkpoint import save_test_results


def eval_results(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute evaluation results for PPG regression task.

    Args:
        predictions: Model predictions (heart rate values)
        targets: Ground truth targets (heart rate values)

    Returns:
        Dictionary containing test metrics (MSE, MAE, RMSE)
    """
    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(mse))

    results = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'num_samples': len(targets),
    }

    return results


def load_best_model(
    args: dict, task: TaskProtocol, model_builder, best_model_path: str) -> torch.nn.Module:
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
    best_model_path: str,
    save_results: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the best checkpoint and evaluate on test set.

    Args:
        args: Training arguments dictionary
        task: Task protocol instance
        best_model_path: Path to the best model checkpoint
        save_results: Whether to save test results to the run folder (default: True)

    Returns:
        (predictions, targets) as numpy arrays
    """
    print("Evaluating best model on the test set...")

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

    # Compute test metrics
    test_results = eval_results(all_preds, all_targets)
    test_mse = test_results['mse']
    test_mae = test_results['mae']
    test_rmse = test_results['rmse']

    # Print results
    print("\n" + "=" * 50)
    print("TEST SET RESULTS:")
    print("=" * 50)
    print(f"MSE:  {test_mse:.4f}")
    print(f"MAE:  {test_mae:.4f} bpm")
    print(f"RMSE: {test_rmse:.4f} bpm")

    # Save to file if requested
    if save_results:
        results_path = save_test_results(best_model_path, test_results)
        print("=" * 50)
        print(f"✅ Test results saved to: {results_path}")

    return all_preds, all_targets

def plot_train_history(history):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train_loss", linewidth=2)
    plt.plot(history["val_loss"], label="val_loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.title("Training & Validation Loss (S4)", fontsize=14)
    plt.grid(True, alpha=0.3)

    # MAE
    plt.subplot(1, 2, 2)
    plt.plot(history["train_mae"], label="train_mae", linewidth=2)
    plt.plot(history["val_mae"], label="val_mae", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MAE (bpm)", fontsize=12)
    plt.legend(fontsize=11)
    plt.title("Mean Absolute Error (S4)", fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def prediction_visualization(predictions, targets):
    """Visualize predictions vs targets and error distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(targets, predictions, alpha=0.5, s=10)
    lims = [min(targets.min(), predictions.min()) - 5, max(targets.max(), predictions.max()) + 5]
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True HR (bpm)', fontsize=12)
    ax.set_ylabel('Predicted HR (bpm)', fontsize=12)
    ax.set_title('S4: Predictions vs Ground Truth', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Error distribution
    ax = axes[1]
    errors = predictions - targets
    ax.hist(errors, bins=50, alpha=0.75, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Prediction Error (bpm)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('S4: Error Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print(f"\nError Statistics:")
    print(f"  Mean Error: {errors.mean():.4f} bpm (bias)")
    print(f"  Std Error:  {errors.std():.4f} bpm")
    print(f"  95% of predictions within: ±{1.96 * errors.std():.2f} bpm")
