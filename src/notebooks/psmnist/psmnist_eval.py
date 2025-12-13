from __future__ import annotations
from typing import Dict, Any, Tuple

import torch
from tqdm import tqdm

from src.eval.report import plot_confusion
from src.types.task_protocol import TaskProtocol
from src.utils.checkpoint import load_trainer_from_checkpoint, save_test_results
from src.eval.metrics import confusion_matrix, per_class_accuracy

def eval_results(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, Any]:
    """Compute evaluation results given logits and labels."""
    preds = logits.argmax(dim=-1)
    accuracy = (preds == labels).float().mean().item()

    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels).item()

    # Compute per-class accuracy
    num_classes = logits.shape[-1]
    cm = confusion_matrix(logits, labels, num_classes=num_classes)
    class_acc = per_class_accuracy(cm)

    results = {
        'accuracy': accuracy,
        'loss': loss,
        'per_class_accuracy': {
            f'class_{cls:02d}': class_acc[cls].item() if cm[cls].sum() > 0 else None
            for cls in range(num_classes)
        },
        'num_samples': len(labels),
    }

    return results

@torch.inference_mode()
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

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Compute and save test metrics if requested
    if save_results:
        test_results = eval_results(all_logits, all_labels)
        accuracy = test_results['accuracy']
        test_loss = test_results['loss']
        # Save using the checkpoint utility
        results_path = save_test_results(best_model_path, test_results)

        print(f"\n✅ Test results saved to: {results_path}")
        print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Test Loss: {test_loss:.4f}")

    return all_logits, all_labels


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