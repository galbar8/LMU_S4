from __future__ import annotations
from typing import Optional
import torch

from src.eval.eval_utils import print_multilabel_evaluation_results
from src.utils.visualization import plot_multilabel_history

def print_evaluation_results(
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
        class_names: List of class names (default: ["NORM", "MI", "STTC", "HYP", "CD"])
    """
    if class_names is None:
        class_names = ["NORM", "MI", "STTC", "HYP", "CD"]

    print_multilabel_evaluation_results(
        logits_val=logits_val,
        labels_val=labels_val,
        logits_test=logits_test,
        labels_test=labels_test,
        threshold=threshold,
        class_names=class_names
    )


def plot_train_history(history, model_name: str = "PTB-XL Model"):
    """Plot training history for PTB-XL multi-label classification."""
    plot_multilabel_history(history, model_name)
