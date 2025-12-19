"""Unified visualization utilities for training history and results."""
from __future__ import annotations
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def plot_classification_history(history: Dict[str, List[float]], model_name: str = "Model"):
    """
    Plot training history for classification tasks (accuracy and loss curves).

    Args:
        history: Training history dictionary with keys like 'train_acc', 'val_acc', 'train_loss', 'val_loss'
        model_name: Name to display in plot titles (e.g., "LMU", "S4")
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Accuracy
    if "train_acc" in history and "val_acc" in history:
        axes[0].plot(history["train_acc"], label="train_acc", linewidth=2.5)
        axes[0].plot(history["val_acc"], label="val_acc", linewidth=2.5)
        axes[0].set_xlabel("Epoch", fontsize=14)
        axes[0].set_ylabel("Accuracy", fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].set_title(f"{model_name} Model - Accuracy", fontsize=16)
        axes[0].grid(True, alpha=0.3)

    # Plot Loss
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

    # Plot Loss
    if "train_loss" in history and "val_loss" in history:
        axes[0].plot(history["train_loss"], label="train_loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="val_loss", linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Loss", fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].set_title(f"{model_name} - Training and Validation Loss", fontsize=14)
        axes[0].grid(True, alpha=0.3)

    # Plot F1-Micro
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

    # Error distribution
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

    # Print error statistics
    print(f"\nError Statistics:")
    print(f"  Mean Error: {errors.mean():.4f} {unit} (bias)")
    print(f"  Std Error:  {errors.std():.4f} {unit}")
    print(f"  95% of predictions within: ±{1.96 * errors.std():.2f} {unit}")


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str = "Model",
    task_type: str = "classification"
):
    """
    Auto-detect and plot appropriate training history based on task type.

    Args:
        history: Training history dictionary
        model_name: Name to display in plot titles
        task_type: Type of task - "classification", "regression", or "multilabel"
    """
    if task_type == "classification":
        plot_classification_history(history, model_name)
    elif task_type == "regression":
        plot_regression_history(history, model_name)
    elif task_type == "multilabel":
        plot_multilabel_history(history, model_name)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

