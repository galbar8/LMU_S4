from __future__ import annotations
import torch
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np

from .metrics import accuracy, per_class_accuracy

def print_basic_report(logits: torch.Tensor, y: torch.Tensor, class_names: Optional[List[str]] = None):
    acc = accuracy(logits, y)
    print(f"Overall accuracy: {acc:.4f}")
    if class_names is not None:
        print(f"Num classes: {len(class_names)}")

def plot_confusion(cm: torch.Tensor, class_names: Optional[List[str]] = None, normalize: bool = True, figsize=(10,10)):
    if normalize:
        cm = cm.float()
        cm = cm / cm.sum(dim=1, keepdim=True).clamp_min(1)
    cm_np = cm.cpu().numpy()

    plt.figure(figsize=figsize)
    plt.imshow(cm_np, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

def print_per_class(cm: torch.Tensor, class_names: Optional[List[str]] = None, top_k: int = 10):
    accs = per_class_accuracy(cm)
    vals, idxs = torch.sort(accs, descending=True)
    print("Top per-class accuracies:")
    for i in range(min(top_k, len(idxs))):
        name = class_names[idxs[i]] if class_names is not None else f"class_{idxs[i].item()}"
        print(f"{i+1:02d}. {name:25s} : {vals[i].item():.3f}")
    print("Bottom per-class accuracies:")
    vals_b, idxs_b = torch.sort(accs, descending=False)
    for i in range(min(top_k, len(idxs_b))):
        name = class_names[idxs_b[i]] if class_names is not None else f"class_{idxs_b[i].item()}"
        print(f"{i+1:02d}. {name:25s} : {vals_b[i].item():.3f}")
