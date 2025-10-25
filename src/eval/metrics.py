from __future__ import annotations
import torch

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()

def confusion_matrix(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    pred = logits.argmax(dim=-1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y.view(-1), pred.view(-1)):
        cm[t, p] += 1
    return cm

def per_class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    correct = cm.diag()
    totals = cm.sum(dim=1).clamp_min(1)
    return correct.float() / totals.float()
