"""Unified evaluation utilities for all tasks."""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.amp import autocast as amp_autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix as sklearn_cm

from src.types.task_protocol import TaskProtocol
from src.utils.checkpoint import save_test_results, load_trainer_from_checkpoint
from src.eval.infer import predict_loader
from src.eval.metrics import confusion_matrix, per_class_accuracy, multilabel_metrics_fn
from src.eval.report import print_basic_report, plot_confusion, print_per_class
from src.utils.common import amp_autocast as amp_ctx
from src.utils.visualization import (
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix_binary,
    plot_threshold_analysis,
    plot_metrics_summary,
    plot_class_distribution,
)

def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for classification tasks.

    Args:
        logits: Model predictions (logits)
        labels: Ground truth labels
        num_classes: Number of classes

    Returns:
        Dictionary containing metrics
    """
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()

    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels).item()

    # Compute per-class accuracy
    cm = confusion_matrix(logits, labels, num_classes=num_classes)
    class_acc = per_class_accuracy(cm)

    results = {
        'accuracy': acc,
        'loss': loss,
        'per_class_accuracy': {
            f'class_{cls:02d}': class_acc[cls].item() if cm[cls].sum() > 0 else None
            for cls in range(num_classes)
        },
        'num_samples': len(labels),
    }

    return results


def compute_regression_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for regression tasks.

    Args:
        predictions: Model predictions
        targets: Ground truth targets

    Returns:
        Dictionary containing metrics (MSE, MAE, RMSE)
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


@torch.inference_mode()
def evaluate_classification_model(
    args: Dict[str, Any],
    task: TaskProtocol,
    best_model_path: str,
    num_classes: int,
    save_results: bool = True,
    use_test_set: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate classification model on validation or test set.

    Args:
        args: Training arguments dictionary
        task: Task protocol instance
        best_model_path: Path to the best model checkpoint
        num_classes: Number of classes
        save_results: Whether to save results to JSON file
        use_test_set: If True, use test set; if False, use validation set

    Returns:
        Tuple of (logits, labels)
    """
    # Load trainer and model from checkpoint
    trainer = load_trainer_from_checkpoint(
        checkpoint_path=best_model_path,
        args=args,
        task=task,
    )
    model = trainer.model.to(args["device"])
    model.eval()

    # Get data loader
    train_loader, val_loader, test_loader = task.make_loaders(
        data_root=args["data_root"],
        batch_size=args["batch"],
        **args["data_loader_kwargs"]
    )

    loader = test_loader if use_test_set else val_loader
    set_name = "test" if use_test_set else "validation"

    # Evaluate
    all_logits, all_labels = predict_loader(
        model,
        loader,
        args["device"],
        amp_ctx,
        args["amp"]
    )

    # Compute metrics
    results = compute_classification_metrics(all_logits, all_labels, num_classes)

    # Save results if requested
    if save_results and use_test_set:
        results_path = save_test_results(best_model_path, results)
        print(f"\n {set_name.capitalize()} results saved to: {results_path}")

    acc = results['accuracy']
    loss = results['loss']
    print(f"\n   {set_name.capitalize()} Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"   {set_name.capitalize()} Loss: {loss:.4f}")

    # Print basic metrics
    print("\n" + "=" * 50)
    print(f"{set_name.upper()} SET RESULTS")
    print("=" * 50)
    print_basic_report(acc, num_classes=num_classes)

    if num_classes == 2:
        # Binary classification detailed results
        plot_binary_classification_results(
            all_logits,
            all_labels,
            class_names=None,
            set_name=set_name.capitalize()
        )

        return all_logits, all_labels

    # Confusion matrix
    cm = confusion_matrix(all_logits, all_labels, num_classes=num_classes)
    plot_confusion(cm, class_names=None, normalize=True, figsize=(6, 6))

    # Per-class statistics
    print("\nTop-10 and Bottom-10 Classes by Performance:")
    print_per_class(cm, class_names=None, top_k=10)

    return all_logits, all_labels


@torch.inference_mode()
def evaluate_regression_model(
    args: Dict[str, Any],
    task: TaskProtocol,
    best_model_path: str,
    save_results: bool = True,
    output_reshape: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate regression model on test set.

    Args:
        args: Training arguments dictionary
        task: Task protocol instance
        best_model_path: Path to the best model checkpoint
        save_results: Whether to save results to JSON file
        output_reshape: Optional tuple for reshaping output (e.g., for time series forecasting)

    Returns:
        Tuple of (predictions, targets) as numpy arrays
    """
    print("Evaluating best model on the test set...")

    device = args.get("device", torch.device("cpu"))
    checkpoint = torch.load(best_model_path, map_location=device)

    # Get test loader
    _, _, test_loader = task.make_loaders(
        data_root=args["data_root"],
        batch_size=args["batch"],
        **args["data_loader_kwargs"]
    )

    trainer = load_trainer_from_checkpoint(
        checkpoint_path=best_model_path,
        args=args,
        task=task,
    )
    model = trainer.model.to(args["device"])
    model.eval()

    # Print validation metrics from checkpoint
    val_metrics = checkpoint.get('val', {})
    if 'mse' in val_metrics:
        print(f"📈 Val MSE: {val_metrics.get('mse', 'N/A'):.6f}, Val MAE: {val_metrics.get('mae', 'N/A'):.6f}")

    # Evaluation loop
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, y, _ = batch  # PPG case with metadata
                else:
                    x, y = batch[:2]
            else:
                x, y = batch["x"], batch["y"]

            x, y = x.to(device), y.to(device)

            # Disable AMP for evaluation to prevent precision issues
            with amp_autocast(device_type=device.type, enabled=False):
                out = model(x)

                # Reshape output if needed (e.g., for time series forecasting)
                if output_reshape:
                    out = out.view(*output_reshape)

            all_preds.append(out.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # Concatenate and flatten
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Flatten for scalar regression tasks
    if all_preds.ndim > 1 and all_preds.shape[1] == 1:
        all_preds = all_preds.flatten()
        all_targets = all_targets.flatten()

    # Compute metrics
    test_results = compute_regression_metrics(all_preds, all_targets)

    # Print results
    print("\n" + "=" * 50)
    print("TEST SET RESULTS:")
    print("=" * 50)
    print(f"MSE:  {test_results['mse']:.6f}")
    print(f"MAE:  {test_results['mae']:.6f}")
    print(f"RMSE: {test_results['rmse']:.6f}")

    # Save to file if requested
    if save_results:
        results_path = save_test_results(best_model_path, test_results)
        print("=" * 50)
        print(f"✅ Test results saved to: {results_path}")

    return all_preds, all_targets


@torch.inference_mode()
def evaluate_multilabel_model(
    args: Dict[str, Any],
    task: TaskProtocol,
    best_model_path: str,
    threshold: float = 0.5,
    save_results: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate multi-label classification model on validation and test sets.

    Args:
        args: Training arguments dictionary
        task: Task protocol instance
        best_model_path: Path to the best model checkpoint
        threshold: Decision threshold for binary predictions
        save_results: Whether to save test results to JSON file

    Returns:
        Tuple of (logits_val, labels_val, logits_test, labels_test)
    """
    print("Evaluating best model on validation and test sets...")

    # Recreate data loaders
    _, val_loader, test_loader = task.make_loaders(
        data_root=args["data_root"],
        batch_size=args["batch"],
        **args.get("data_loader_kwargs", {})
    )

    trainer = load_trainer_from_checkpoint(
        checkpoint_path=best_model_path,
        args=args,
        task=task,
    )
    model = trainer.model.to(args["device"])
    model.eval()

    device = args.get("device", torch.device("cpu"))
    amp = args.get("amp", False) and device.type in {"cuda", "mps"}

    # Get predictions on validation set
    logits_val, labels_val = predict_loader(model, val_loader, device, amp_ctx, amp)

    # Get predictions on test set
    logits_test, labels_test = predict_loader(model, test_loader, device, amp_ctx, amp)

    # Compute and save test metrics
    if save_results:
        metrics_fn = multilabel_metrics_fn(threshold=threshold)
        test_metrics = metrics_fn(logits_test, labels_test)

        test_results = {
            'f1_micro': float(test_metrics['f1_micro']),
            'num_samples': int(labels_test.shape[0]),
            'threshold': threshold
        }

        results_path = save_test_results(best_model_path, test_results)
        print(f"✅ Test results saved to: {results_path}")

    return logits_val, labels_val, logits_test, labels_test


def print_multilabel_evaluation_results(
    logits_val: torch.Tensor,
    labels_val: torch.Tensor,
    logits_test: torch.Tensor,
    labels_test: torch.Tensor,
    threshold: float = 0.5,
    class_names: Optional[list] = None
):
    """
    Print comprehensive evaluation results for multi-label classification.

    Args:
        logits_val: Validation set logits [N_val, num_classes]
        labels_val: Validation set labels [N_val, num_classes]
        logits_test: Test set logits [N_test, num_classes]
        labels_test: Test set labels [N_test, num_classes]
        threshold: Decision threshold for binary predictions
        class_names: List of class names
    """
    num_classes = logits_val.shape[-1]

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    # Overall metrics
    print("\n" + "=" * 60)
    print("VALIDATION SET RESULTS:")
    print("=" * 60)
    metrics_fn = multilabel_metrics_fn(threshold=threshold)
    val_metrics = metrics_fn(logits_val, labels_val)
    for k, v in val_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("TEST SET RESULTS:")
    print("=" * 60)
    test_metrics = metrics_fn(logits_test, labels_test)
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    # Per-class analysis
    preds_val = (torch.sigmoid(logits_val) > threshold).float()
    preds_test = (torch.sigmoid(logits_test) > threshold).float()

    print("\n" + "=" * 60)
    print("PER-CLASS F1 SCORES (Validation):")
    print("=" * 60)
    for i, name in enumerate(class_names):
        tp = (preds_val[:, i] * labels_val[:, i]).sum().item()
        fp = (preds_val[:, i] * (1 - labels_val[:, i])).sum().item()
        fn = ((1 - preds_val[:, i]) * labels_val[:, i]).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"{name:10s}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")

    print("\n" + "=" * 60)
    print("PER-CLASS F1 SCORES (Test):")
    print("=" * 60)
    for i, name in enumerate(class_names):
        tp = (preds_test[:, i] * labels_test[:, i]).sum().item()
        fp = (preds_test[:, i] * (1 - labels_test[:, i])).sum().item()
        fn = ((1 - preds_test[:, i]) * labels_test[:, i]).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"{name:10s}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")


def plot_binary_classification_results(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[Tuple[str, str]] = None,
    set_name: str = "Test"
) -> Dict[str, Any]:
    """
    Create comprehensive visualizations for binary classification.

    This function generates 8 separate plots (each in its own figure) for analyzing
    binary classification performance.

    Args:
        logits: Model predictions (logits) [N, 2]
        labels: Ground truth labels [N]
        class_names: Tuple of (negative_class_name, positive_class_name)
        set_name: Name of the dataset (e.g., "Test", "Validation")

    Returns:
        Dictionary containing computed metrics
    """
    if class_names is None:
        class_names = ("Not Duplicate", "Duplicate")

    # Convert to numpy
    labels_np = labels.cpu().numpy()
    probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=-1).cpu().numpy()

    # Compute metrics
    accuracy = accuracy_score(labels_np, preds)
    precision = precision_score(labels_np, preds, zero_division=0)
    recall = recall_score(labels_np, preds, zero_division=0)
    f1 = f1_score(labels_np, preds, zero_division=0)

    # Compute AUC scores
    fpr, tpr, _ = roc_curve(labels_np, probs)
    roc_auc = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(labels_np, probs)
    pr_auc = auc(recall_curve, precision_curve)

    # Get confusion matrix
    cm = sklearn_cm(labels_np, preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n1. ROC Curve...")
    plot_roc_curve(labels_np, probs, set_name)

    print("2. Precision-Recall Curve...")
    plot_precision_recall_curve(labels_np, probs, set_name)

    print("3. Confusion Matrix...")
    plot_confusion_matrix_binary(labels_np, preds, class_names, set_name)

    print("4. Threshold Analysis...")
    plot_threshold_analysis(labels_np, probs)

    print("5. Metrics Summary...")
    plot_metrics_summary(accuracy, precision, recall, f1)

    print("6. Class Distribution...")
    plot_class_distribution(labels_np, class_names, set_name)

    print("\n" + "=" * 60)
    print("All visualizations generated!")
    print("=" * 60)

    # Return metrics dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }

