"""ESC-50 evaluation utilities."""
from __future__ import annotations
from typing import Dict, Any

import torch
import matplotlib.pyplot as plt

from src.types.task_protocol import TaskProtocol
from src.eval.infer import predict_loader
from src.eval.metrics import accuracy, confusion_matrix, per_class_accuracy
from src.eval.report import print_basic_report, plot_confusion, print_per_class
from src.utils.common import amp_autocast
from src.utils.checkpoint import load_trainer_from_checkpoint, save_test_results

@torch.inference_mode()
def evaluate_best_model(
    args: Dict[str, Any],
    task: TaskProtocol,
    best_model_path: str,
    num_classes: int = 50,
    save_results: bool = True,
):
    """
    Evaluate the trained model on validation set with detailed metrics.

    Args:
        args: Training arguments dictionary
        task: Task protocol instance
        best_model_path: Path to the best model checkpoint
        num_classes: Number of classes (default: 50 for ESC-50)
        save_results: Whether to save validation results to the run folder (default: True)
    """
    # Load trainer and model from checkpoint
    trainer = load_trainer_from_checkpoint(
        checkpoint_path=best_model_path,
        args=args,
        task=task,
    )
    model = trainer.model.to(args["device"])
    model.eval()

    # Get validation loader
    _, val_loader, _ = task.make_loaders(
        data_root=args["data_root"],
        batch_size=args["batch"],
        **args["data_loader_kwargs"]
    )

    # Evaluate on validation set using predict_loader
    all_logits, all_labels = predict_loader(
        model,
        val_loader,
        args["device"],
        amp_autocast,
        args["amp"]
    )

    # Compute accuracy using the metrics function
    acc = accuracy(all_logits, all_labels)

    # Compute confusion matrix
    cm = confusion_matrix(all_logits, all_labels, num_classes=num_classes)

    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(all_logits, all_labels).item()

    # Compute per-class accuracy
    class_acc = per_class_accuracy(cm)

    # Prepare validation results
    val_results = {
        'accuracy': acc,
        'loss': loss,
        'per_class_accuracy': {
            f'class_{cls:02d}': class_acc[cls].item() if cm[cls].sum() > 0 else None
            for cls in range(num_classes)
        },
        'num_samples': len(all_labels),
    }

    # Save validation results if requested
    if save_results:
        results_path = save_test_results(best_model_path, val_results)
        print(f"\n✅ Validation results saved to: {results_path}")
        print(f"   Validation Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"   Validation Loss: {loss:.4f}")

    # Print basic metrics
    print("\n" + "=" * 50)
    print("VALIDATION SET RESULTS")
    print("=" * 50)
    print_basic_report(acc, num_classes=num_classes)

    # Confusion matrix
    plot_confusion(cm, class_names=None, normalize=True, figsize=(6, 6))

    # Per-class statistics
    print("\nTop-10 and Bottom-10 Classes by Performance:")
    print_per_class(cm, class_names=None, top_k=10)

    return all_logits, all_labels


def plot_training_history(trainer, model_name: str = "Model"):
    """
    Plot training history (accuracy and loss curves).

    Args:
        trainer: Trained Trainer instance with history
        model_name: Name to display in plot titles (e.g., "LMU", "S4")
    """
    history = trainer.history

    # Plot Accuracy
    plt.figure(figsize=(16, 6))
    plt.plot(history["train_acc"], label="train_acc", linewidth=2.5)
    plt.plot(history["val_acc"], label="val_acc", linewidth=2.5)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(fontsize=12)
    plt.title(f"{model_name} Model - Accuracy", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot Loss
    plt.figure(figsize=(16, 6))
    plt.plot(history["train_loss"], label="train_loss", linewidth=2.5)
    plt.plot(history["val_loss"], label="val_loss", linewidth=2.5)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.title(f"{model_name} Model - Loss", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

