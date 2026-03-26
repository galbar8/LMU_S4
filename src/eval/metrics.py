"""
Consolidated metrics module using sklearn where possible.

This module provides metric computation functions for classification, regression,
and multi-label tasks. Where possible, we use sklearn's built-in implementations
to avoid code duplication and leverage well-tested implementations.
"""
from __future__ import annotations
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==============================================================================
# Classification Metrics
# ==============================================================================

@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute accuracy using sklearn.

    Args:
        logits: Model predictions (logits) [N, num_classes]
        y: Ground truth labels [N]

    Returns:
        Accuracy score (0-1)
    """
    pred = logits.argmax(dim=-1).cpu().numpy()
    y_np = y.cpu().numpy()
    return float(accuracy_score(y_np, pred))


@torch.no_grad()
def top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Alias for accuracy (top-1 accuracy).

    Args:
        logits: Model predictions (logits) [N, num_classes]
        y: Ground truth labels [N]

    Returns:
        Top-1 accuracy score (0-1)
    """
    return accuracy(logits, y)


def confusion_matrix(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute confusion matrix using sklearn.

    Args:
        logits: Model predictions (logits) [N, num_classes]
        y: Ground truth labels [N]
        num_classes: Number of classes

    Returns:
        Confusion matrix as torch.Tensor [num_classes, num_classes]
    """
    pred = logits.argmax(dim=-1).cpu().numpy()
    y_np = y.cpu().numpy()

    # Use sklearn's confusion_matrix which handles edge cases better
    cm = sklearn_confusion_matrix(y_np, pred, labels=np.arange(num_classes))
    return torch.from_numpy(cm).long()


def per_class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class accuracy from confusion matrix.

    This is a custom metric that can't be directly replaced by sklearn.
    It computes recall (sensitivity) for each class.

    Args:
        cm: Confusion matrix [num_classes, num_classes]

    Returns:
        Per-class accuracy [num_classes]
    """
    correct = cm.diag()
    totals = cm.sum(dim=1).clamp_min(1)
    return correct.float() / totals.float()


@torch.no_grad()
def compute_f1_score(logits: torch.Tensor, y: torch.Tensor, average: str = 'binary') -> float:
    """
    Compute F1 score using sklearn.

    Args:
        logits: Model predictions (logits) [N, num_classes] or [N]
        y: Ground truth labels [N]
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')

    Returns:
        F1 score (0-1)
    """
    pred = logits.argmax(dim=-1).cpu().numpy()
    y_np = y.cpu().numpy()
    return float(f1_score(y_np, pred, average=average, zero_division=0))


@torch.no_grad()
def compute_precision(logits: torch.Tensor, y: torch.Tensor, average: str = 'binary') -> float:
    """
    Compute precision using sklearn.

    Args:
        logits: Model predictions (logits) [N, num_classes] or [N]
        y: Ground truth labels [N]
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')

    Returns:
        Precision score (0-1)
    """
    pred = logits.argmax(dim=-1).cpu().numpy()
    y_np = y.cpu().numpy()
    return float(precision_score(y_np, pred, average=average, zero_division=0))


@torch.no_grad()
def compute_recall(logits: torch.Tensor, y: torch.Tensor, average: str = 'binary') -> float:
    """
    Compute recall using sklearn.

    Args:
        logits: Model predictions (logits) [N, num_classes] or [N]
        y: Ground truth labels [N]
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')

    Returns:
        Recall score (0-1)
    """
    pred = logits.argmax(dim=-1).cpu().numpy()
    y_np = y.cpu().numpy()
    return float(recall_score(y_np, pred, average=average, zero_division=0))


@torch.no_grad()
def compute_roc_auc(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute ROC-AUC score using sklearn.
    Only applicable for binary classification.

    Args:
        logits: Model predictions (logits) [N, 2]
        y: Ground truth labels [N] (values should be 0 or 1)

    Returns:
        ROC-AUC score (0-1), or 0.0 if computation fails
    """
    # Extract probability for positive class
    if logits.dim() == 2 and logits.size(1) == 2:
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    elif logits.dim() == 2 and logits.size(1) == 1:
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    else:
        probs = torch.sigmoid(logits).cpu().numpy()

    y_np = y.cpu().numpy()

    try:
        return float(roc_auc_score(y_np, probs))
    except ValueError:
        # Happens when only one class is present in y_true
        return 0.0


@torch.no_grad()
def compute_pr_auc(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Precision-Recall AUC score using sklearn.
    Only applicable for binary classification.

    Args:
        logits: Model predictions (logits) [N, 2]
        y: Ground truth labels [N] (values should be 0 or 1)

    Returns:
        PR-AUC score (0-1), or 0.0 if computation fails
    """
    # Extract probability for positive class
    if logits.dim() == 2 and logits.size(1) == 2:
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    elif logits.dim() == 2 and logits.size(1) == 1:
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    else:
        probs = torch.sigmoid(logits).cpu().numpy()

    y_np = y.cpu().numpy()

    try:
        return float(average_precision_score(y_np, probs))
    except ValueError:
        # Happens when only one class is present in y_true
        return 0.0


# ==============================================================================
# Multi-label Classification Metrics
# ==============================================================================

def multilabel_metrics_fn(threshold: float = 0.5):
    """
    Create a multi-label metrics function using sklearn.

    Args:
        threshold: Decision threshold for binary predictions

    Returns:
        Function that computes multi-label metrics
    """
    @torch.no_grad()
    def _fn(logits: torch.Tensor, y: torch.Tensor):
        """
        Compute multi-label metrics (micro-averaged F1).

        Args:
            logits: Model predictions (logits) [B, C]
            y: Ground truth labels [B, C] in {0,1}

        Returns:
            Dictionary with 'f1_micro' score
        """
        # Convert to binary predictions
        p = (logits.sigmoid() >= threshold).float().cpu().numpy()
        y_np = y.cpu().numpy()

        # Use sklearn's f1_score with micro averaging
        # This computes: 2 * (precision * recall) / (precision + recall)
        # where precision and recall are computed globally across all samples and classes
        f1_micro = float(f1_score(y_np.flatten(), p.flatten(), average='micro', zero_division=0))

        return {"f1_micro": f1_micro}

    return _fn


# ==============================================================================
# Regression Metrics
# ==============================================================================

@torch.no_grad()
def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute regression metrics using sklearn.

    Args:
        y_true: Ground truth values [N]
        y_pred: Predicted values [N]

    Returns:
        Dictionary with 'mae', 'rmse', and 'r2' scores
    """
    y_true_np = y_true.view(-1).cpu().numpy()
    y_pred_np = y_pred.view(-1).cpu().numpy()

    # Use sklearn's built-in implementations
    mae = float(mean_absolute_error(y_true_np, y_pred_np))
    mse = float(mean_squared_error(y_true_np, y_pred_np))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_np, y_pred_np))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

