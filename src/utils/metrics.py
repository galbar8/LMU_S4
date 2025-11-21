# src/utils/metrics.py
import math

import torch


@torch.no_grad()
def top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


def multilabel_metrics_fn(threshold: float = 0.5):
    @torch.no_grad()
    def _fn(logits: torch.Tensor, y: torch.Tensor):
        # logits: [B, C]; y: [B, C] in {0,1}
        p = (logits.sigmoid() >= threshold).float()
        # micro-F1
        tp = (p * y).sum().item()
        fp = (p * (1 - y)).sum().item()
        fn = ((1 - p) * y).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_micro = 2 * precision * recall / (precision + recall + 1e-8)
        return {"f1_micro": f1_micro}

    return _fn


@torch.no_grad()
def regression_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    y_true, y_pred: shape [N] (float)
    Returns dict with MAE, RMSE, and R2.
    """
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse = math.sqrt(torch.mean((y_pred - y_true) ** 2).item())
    ybar = torch.mean(y_true)
    ss_tot = torch.sum((y_true - ybar) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    return {"mae": mae, "rmse": rmse, "r2": r2.item()}
