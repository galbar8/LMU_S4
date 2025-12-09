"""ESC-50 evaluation utilities."""
from __future__ import annotations
from typing import Dict, Any

from src.types.task_protocol import TaskProtocol
from src.eval.infer import predict_loader
from src.eval.metrics import confusion_matrix
from src.eval.report import print_basic_report, plot_confusion, print_per_class
from src.utils.common import amp_autocast
import matplotlib.pyplot as plt

def evaluate_best_model(
    trainer,
    args: Dict[str, Any],
    task: TaskProtocol,
    num_classes: int = 50
):
    """
    Evaluate the trained model on validation set with detailed metrics.

    Args:
        trainer: Trained Trainer instance
        args: Training arguments dictionary
        task: Task protocol instance
        num_classes: Number of classes (default: 50 for ESC-50)
    """
    # Get validation loader
    _, val_loader, _ = task.make_loaders(
        data_root=args["data_root"],
        batch_size=args["batch"],
        **args["data_loader_kwargs"]
    )

    # Get predictions
    logits, labels = predict_loader(
        trainer.model,
        val_loader,
        trainer.device,
        amp_autocast,
        trainer.amp
    )

    # Print basic metrics
    print("\n" + "=" * 50)
    print("VALIDATION SET RESULTS")
    print("=" * 50)
    print_basic_report(logits, labels)

    # Confusion matrix
    cm = confusion_matrix(logits, labels, num_classes=num_classes)
    plot_confusion(cm, class_names=None, normalize=True, figsize=(6, 6))

    # Per-class statistics
    print("\nTop-10 and Bottom-10 Classes by Performance:")
    print_per_class(cm, class_names=None, top_k=10)


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

