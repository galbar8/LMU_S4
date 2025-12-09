"""PTB-XL evaluation utilities."""
from __future__ import annotations
from typing import Tuple, Dict, Any
import os
import torch

from src.models.v2.build_model import build_model
from src.types.task_protocol import TaskProtocol
from src.eval.infer import predict_loader
from src.utils.metrics import multilabel_metrics_fn
from src.utils.common import amp_autocast


def evaluate_best_model(
    args: Dict[str, Any],
    task: TaskProtocol,
    best_model_path: str,
    data_root: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load the best checkpoint and evaluate on validation and test sets.

    Args:
        args: Training arguments dictionary
        task: Task protocol instance
        best_model_path: Path to the best model checkpoint
        data_root: Root directory of the dataset

    Returns:
        (logits_val, labels_val, logits_test, labels_test) as torch tensors
    """
    print("📊 Evaluating best model on validation and test sets...")

    # Check if checkpoint exists
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Checkpoint not found at {best_model_path}")

    # Recreate data loaders
    _, val_loader, test_loader = task.make_loaders(
        data_root=data_root,
        batch_size=args["batch"],
        **args.get("data_loader_kwargs", {})
    )

    # Load best checkpoint
    device = args.get("device", torch.device("cpu"))
    best_ckpt = torch.load(best_model_path, map_location=device)

    # Recreate model
    flat_args = dict(args)
    flat_args.update(args.get("data_loader_kwargs", {}))
    d_in = task.infer_input_dim(flat_args)
    n_classes = task.infer_num_classes(flat_args)
    theta = task.infer_theta(flat_args)

    block_cfg = args["block_cfg_ctor"](theta)

    model = build_model(
        d_in=d_in,
        n_classes=n_classes,
        d_model=args["d_model"],
        depth=args["depth"],
        block_cfg=block_cfg,
    )

    # Load weights
    model.load_state_dict(best_ckpt["model"])
    model.to(device)
    model.eval()

    amp = args.get("amp", False) and device.type in {"cuda", "mps"}

    print(f"✅ Loaded checkpoint from epoch {best_ckpt.get('epoch', 'N/A')}")
    print(f"📈 Val metrics: {best_ckpt.get('val', {})}")

    # Get predictions on validation set
    logits_val, labels_val = predict_loader(model, val_loader, device, amp_autocast, amp)

    # Get predictions on test set
    logits_test, labels_test = predict_loader(model, test_loader, device, amp_autocast, amp)

    return logits_val, labels_val, logits_test, labels_test


def print_evaluation_results(
    logits_val: torch.Tensor,
    labels_val: torch.Tensor,
    logits_test: torch.Tensor,
    labels_test: torch.Tensor,
    threshold: float = 0.5,
    class_names: list = None
):
    """
    Print comprehensive evaluation results for multi-label classification.

    Args:
        logits_val: Validation set logits [N_val, num_classes]
        labels_val: Validation set labels [N_val, num_classes]
        logits_test: Test set logits [N_test, num_classes]
        labels_test: Test set labels [N_test, num_classes]
        threshold: Decision threshold for binary predictions
        class_names: List of class names (default: ["NORM", "MI", "STTC", "HYP", "CD"])
    """
    if class_names is None:
        class_names = ["NORM", "MI", "STTC", "HYP", "CD"]

    metrics_fn = multilabel_metrics_fn(threshold=threshold)

    # Overall metrics
    print("\n" + "=" * 60)
    print("VALIDATION SET RESULTS:")
    print("=" * 60)
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
        tp = ((preds_val[:, i] == 1) & (labels_val[:, i] == 1)).sum().item()
        fp = ((preds_val[:, i] == 1) & (labels_val[:, i] == 0)).sum().item()
        fn = ((preds_val[:, i] == 0) & (labels_val[:, i] == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        support = labels_val[:, i].sum().item()
        print(f"{name:8s}: F1={f1:.4f} (P={precision:.4f}, R={recall:.4f}, support={int(support)})")

    print("\n" + "=" * 60)
    print("PER-CLASS F1 SCORES (Test):")
    print("=" * 60)
    for i, name in enumerate(class_names):
        tp = ((preds_test[:, i] == 1) & (labels_test[:, i] == 1)).sum().item()
        fp = ((preds_test[:, i] == 1) & (labels_test[:, i] == 0)).sum().item()
        fn = ((preds_test[:, i] == 0) & (labels_test[:, i] == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        support = labels_test[:, i].sum().item()
        print(f"{name:8s}: F1={f1:.4f} (P={precision:.4f}, R={recall:.4f}, support={int(support)})")

    print("=" * 60)

