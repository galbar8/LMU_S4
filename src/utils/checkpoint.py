"""Trainer checkpoint utilities."""
from __future__ import annotations
from typing import Dict, Any, Optional
import json
from pathlib import Path
import torch

from src.types.task_protocol import TaskProtocol
from src.train_utils.trainer import Trainer


def load_trainer_from_checkpoint(
    checkpoint_path: str,
    args: Dict[str, Any],
    task: TaskProtocol,
) -> Trainer:
    """
    Load trainer from checkpoint with full state restoration.

    Args:
        checkpoint_path: Path to the checkpoint file
        args: Training arguments dictionary
        task: Task protocol instance

    Returns:
        Trainer instance with restored state
    """
    # Load checkpoint
    device = args.get("device", torch.device("cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create new trainer instance with same args
    trainer = Trainer(args=args, task=task)

    # Restore state
    trainer.model.load_state_dict(checkpoint["model"])

    if "optimizer" in checkpoint:
        trainer.opt.load_state_dict(checkpoint["optimizer"])

    if "history" in checkpoint:
        trainer.history = checkpoint["history"]

    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"📊 Val metrics: {checkpoint.get('val', {})}")

    return trainer


def save_test_results(
    checkpoint_path: str,
    test_results: Dict[str, Any],
    filename: str = "test_results.json"
) -> Path:
    """
    Save test results to the same directory as the checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        test_results: Dictionary containing test metrics
        filename: Name of the results file (default: "test_results.json")

    Returns:
        Path to the saved results file
    """
    checkpoint_dir = Path(checkpoint_path).parent
    results_path = checkpoint_dir / filename

    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    return results_path


def load_test_results(
    checkpoint_path: str,
    filename: str = "test_results.json"
) -> Optional[Dict[str, Any]]:
    """
    Load test results from the checkpoint directory if they exist.

    Args:
        checkpoint_path: Path to the checkpoint file
        filename: Name of the results file (default: "test_results.json")

    Returns:
        Dictionary containing test metrics, or None if file doesn't exist
    """
    checkpoint_dir = Path(checkpoint_path).parent
    results_path = checkpoint_dir / filename

    if not results_path.exists():
        return None

    with open(results_path, 'r') as f:
        return json.load(f)
