
"""Trainer checkpoint utilities."""
from __future__ import annotations
from typing import Dict, Any
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

    if "scheduler" in checkpoint:
        trainer.scheduler.load_state_dict(checkpoint["scheduler"])

    if "history" in checkpoint:
        trainer.history = checkpoint["history"]

    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"📊 Val metrics: {checkpoint.get('val', {})}")

    return trainer

