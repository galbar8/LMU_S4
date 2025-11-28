from __future__ import annotations
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.eval.report import plot_confusion
from src.types.task_protocol import TaskProtocol
from src.utils.checkpoint import load_trainer_from_checkpoint
from src.eval.metrics import confusion_matrix, per_class_accuracy


@torch.inference_mode()
def evaluate_best_model(
    args: Dict[str, Any],
    task: TaskProtocol,
    best_model_path: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the best model on the test set.

    Returns:
        - A tuple of (logits, labels) for the test set.
    """
    # Load trainer and model from checkpoint
    trainer = load_trainer_from_checkpoint(
        checkpoint_path=best_model_path,
        args=args,
        task=task,
    )
    model = trainer.model.to(args["device"])
    model.eval()

    # Get test loader
    _, _, test_loader = task.make_loaders(
        data_root=args["data_root"],
        batch_size=args["batch"],
        # num_workers is expected to be passed via data_loader_kwargs, avoid duplicates
        **args["data_loader_kwargs"],
    )

    # Evaluate on test set
    all_logits = []
    all_labels = []
    for batch in tqdm(test_loader, desc="Evaluating on test set"):
        x, y = batch
        x, y = x.to(args["device"]), y.to(args["device"])

        with torch.autocast(
            device_type=str(args["device"].type),
            dtype=torch.bfloat16 if args["device"].type == "cpu" else torch.float16,
            enabled=args["amp"],
        ):
            logits = model(x)

        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


def print_evaluation_results(
    logits_test: torch.Tensor,
    labels_test: torch.Tensor,
) -> None:
    """Prints evaluation results."""
    preds = logits_test.argmax(dim=-1)
    accuracy = (preds == labels_test).float().mean()

    print("\n==================================================")
    print("TEST SET RESULTS:")
    print("==================================================")
    print(f"Accuracy: {accuracy:.4f}")

    num_classes = logits_test.shape[-1]
    cm = confusion_matrix(logits_test, labels_test, num_classes=num_classes)
    class_acc = per_class_accuracy(cm)

    for cls in range(num_classes):
        if cm[cls].sum() > 0:
            print(f"Class {cls:02d}: {class_acc[cls]:.4f}")
        else:
            print(f"Class {cls:02d}: n/a (no samples)")

    plot_confusion(cm, class_names=None, normalize=True, figsize=(6, 6))