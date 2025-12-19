from __future__ import annotations
from typing import Dict, Any, Tuple

import torch

from src.types.task_protocol import TaskProtocol
from src.eval.eval_utils import evaluate_classification_model


def evaluate_best_model(
    args: Dict[str, Any],
    task: TaskProtocol,
    best_model_path: str,
    save_results: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the best model on the test set.

    Args:
        args: Arguments dictionary containing device, data_root, batch size, etc.
        task: Task protocol instance
        best_model_path: Path to the best model checkpoint
        save_results: Whether to save test results to the run folder (default: True)

    Returns:
        - A tuple of (logits, labels) for the test set.
    """
    return evaluate_classification_model(
        args=args,
        task=task,
        best_model_path=best_model_path,
        num_classes=10,  # PS-MNIST has 10 digit classes
        save_results=save_results,
        use_test_set=True,
    )