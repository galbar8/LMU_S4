"""Unified visualization utilities for training history and results."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix as sklearn_cm,
    precision_score,
    recall_score,
    f1_score,
)

def plot_classification_history(history: Dict[str, List[float]], model_name: str = "Model"):
    """
    Plot training history for classification tasks (accuracy and loss curves).

    Args:
        history: Training history dictionary with keys like 'train_acc', 'val_acc', 'train_loss', 'val_loss'
        model_name: Name to display in plot titles (e.g., "LMU", "S4")
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if "train_acc" in history and "val_acc" in history:
        axes[0].plot(history["train_acc"], label="train_acc", linewidth=2.5)
        axes[0].plot(history["val_acc"], label="val_acc", linewidth=2.5)
        axes[0].set_xlabel("Epoch", fontsize=14)
        axes[0].set_ylabel("Accuracy", fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].set_title(f"{model_name} Model - Accuracy", fontsize=16)
        axes[0].grid(True, alpha=0.3)

    if "train_loss" in history and "val_loss" in history:
        axes[1].plot(history["train_loss"], label="train_loss", linewidth=2.5)
        axes[1].plot(history["val_loss"], label="val_loss", linewidth=2.5)
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].set_ylabel("Loss", fontsize=14)
        axes[1].legend(fontsize=12)
        axes[1].set_title(f"{model_name} Model - Loss", fontsize=16)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_regression_history(history: Dict[str, List[float]], model_name: str = "Model"):
    """
    Plot training history for regression tasks (loss and MAE curves).

    Args:
        history: Training history dictionary with keys like 'train_loss', 'val_loss', 'train_mae', 'val_mae'
        model_name: Name to display in plot titles (e.g., "LMU", "S4")
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss (MSE)
    if "train_loss" in history and "val_loss" in history:
        axes[0].plot(history["train_loss"], label="train_loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="val_loss", linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("MSE Loss", fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].set_title(f"{model_name} - Training & Validation Loss", fontsize=14)
        axes[0].grid(True, alpha=0.3)

    # Plot MAE
    if "train_mae" in history and "val_mae" in history:
        axes[1].plot(history["train_mae"], label="train_mae", linewidth=2)
        axes[1].plot(history["val_mae"], label="val_mae", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("MAE", fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].set_title(f"{model_name} - Mean Absolute Error", fontsize=14)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_multilabel_history(history: Dict[str, List[float]], model_name: str = "Model"):
    """
    Plot training history for multi-label classification tasks.

    Args:
        history: Training history dictionary with keys like 'train_loss', 'val_loss', 'train_f1_micro', 'val_f1_micro'
        model_name: Name to display in plot titles
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    if "train_loss" in history and "val_loss" in history:
        axes[0].plot(history["train_loss"], label="train_loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="val_loss", linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].set_title(f"{model_name} - Training and Validation Loss", fontsize=14)
        axes[0].grid(True, alpha=0.3)

    if "train_f1_micro" in history and "val_f1_micro" in history:
        axes[1].plot(history["train_f1_micro"], label="train_f1_micro", linewidth=2)
        axes[1].plot(history["val_f1_micro"], label="val_f1_micro", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("F1-Micro", fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].set_title(f"{model_name} - Training and Validation F1-Micro Score", fontsize=14)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_regression_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_name: str = "Model",
    unit: str = "bpm"
):
    """
    Visualize regression predictions vs targets and error distribution.

    Args:
        predictions: Model predictions
        targets: Ground truth targets
        model_name: Name to display in plot titles
        unit: Unit of measurement (e.g., "bpm" for heart rate)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(targets, predictions, alpha=0.5, s=10)
    lims = [
        min(targets.min(), predictions.min()) - 5,
        max(targets.max(), predictions.max()) + 5
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, label='Perfect prediction')
    ax.set_xlabel(f'True Value ({unit})', fontsize=12)
    ax.set_ylabel(f'Predicted Value ({unit})', fontsize=12)
    ax.set_title(f'{model_name}: Predictions vs Ground Truth', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    errors = predictions - targets
    ax.hist(errors, bins=50, alpha=0.75, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel(f'Prediction Error ({unit})', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{model_name}: Error Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print(f"\nError Statistics:")
    print(f"  Mean Error: {errors.mean():.4f} {unit} (bias)")
    print(f"  Std Error:  {errors.std():.4f} {unit}")
    print(f"  95% of predictions within: ±{1.96 * errors.std():.2f} {unit}")


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    set_name: str = "Test",
    figsize: Tuple[int, int] = (8, 6),
) -> float:
    """
    Plot ROC curve for binary classification.

    Args:
        labels: Ground truth labels [N]
        probs: Predicted probabilities for positive class [N]
        set_name: Name of the dataset (e.g., "Test", "Validation")
        figsize: Figure size (width, height)

    Returns:
        ROC AUC score
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{set_name} Set - ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return roc_auc


def plot_precision_recall_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    set_name: str = "Test",
    figsize: Tuple[int, int] = (8, 6),
) -> float:
    """
    Plot Precision-Recall curve for binary classification.

    Args:
        labels: Ground truth labels [N]
        probs: Predicted probabilities for positive class [N]
        set_name: Name of the dataset (e.g., "Test", "Validation")
        figsize: Figure size (width, height)

    Returns:
        PR AUC score
    """
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall_curve, precision_curve)

    plt.figure(figsize=figsize)
    plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{set_name} Set - Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return pr_auc


def plot_confusion_matrix_binary(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: Tuple[str, str] = ("Negative", "Positive"),
    set_name: str = "Test",
    figsize: Tuple[int, int] = (8, 6),
) -> np.ndarray:
    """
    Plot enhanced confusion matrix with counts and percentages.

    Args:
        labels: Ground truth labels [N]
        preds: Predicted labels [N]
        class_names: Tuple of (negative_class_name, positive_class_name)
        set_name: Name of the dataset (e.g., "Test", "Validation")
        figsize: Figure size (width, height)

    Returns:
        Confusion matrix as numpy array [[TN, FP], [FN, TP]]
    """
    cm = sklearn_cm(labels, preds)

    plt.figure(figsize=figsize)
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percentage = count / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            plt.text(j, i, f'{count}\n({percentage:.1f}%)',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'{set_name} Set - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    plt.tight_layout()
    plt.show()

    return cm


def plot_threshold_analysis(
    labels: np.ndarray,
    probs: np.ndarray,
    default_threshold: float = 0.5,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot how metrics change with different decision thresholds.

    Args:
        labels: Ground truth labels [N]
        probs: Predicted probabilities for positive class [N]
        default_threshold: Default threshold to mark on plot
        figsize: Figure size (width, height)
    """
    thresholds = np.linspace(0, 1, 100)
    precisions, recalls, f1s = [], [], []

    for thresh in thresholds:
        preds_thresh = (probs >= thresh).astype(int)
        p = precision_score(labels, preds_thresh, zero_division=0)
        r = recall_score(labels, preds_thresh, zero_division=0)
        f = f1_score(labels, preds_thresh, zero_division=0)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    plt.figure(figsize=figsize)
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, label='F1-Score', linewidth=2)
    plt.axvline(x=default_threshold, color='black', linestyle='--', alpha=0.5, label=f'Default={default_threshold}')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs. Decision Threshold', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_metrics_summary(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot bar chart summarizing key metrics.

    Args:
        accuracy: Accuracy score
        precision: Precision score
        recall: Recall score
        f1: F1 score
        figsize: Figure size (width, height)
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    plt.figure(figsize=figsize)
    bars = plt.barh(metrics, values, color=colors, alpha=0.7)
    plt.xlim([0, 1])
    plt.xlabel('Score', fontsize=12)
    plt.title('Performance Metrics Summary', fontsize=14, fontweight='bold')
    plt.grid(True, axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(val + 0.02, i, f'{val:.4f}', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: Tuple[str, str] = ("Negative", "Positive"),
    set_name: str = "Test",
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """
    Plot distribution of classes in the dataset.

    Args:
        labels: Ground truth labels [N]
        class_names: Tuple of (negative_class_name, positive_class_name)
        set_name: Name of the dataset (e.g., "Test", "Validation")
        figsize: Figure size (width, height)
    """
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=figsize)
    plt.bar([class_names[i] for i in unique], counts, color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'{set_name} Set - Class Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)

    for i, (label, count) in enumerate(zip(unique, counts)):
        percentage = count / len(labels) * 100
        plt.text(i, count + max(counts) * 0.02, f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()