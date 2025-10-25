# src/utils/metrics.py
import torch
@torch.no_grad()
def top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()

def multilabel_metrics_fn(threshold: float = 0.5):
    def _fn(logits: torch.Tensor, y: torch.Tensor):
        # logits: [B, C]; y: [B, C] in {0,1}
        p = (logits.sigmoid() >= threshold).float()
        # micro-F1
        tp = (p * y).sum().item()
        fp = (p * (1 - y)).sum().item()
        fn = ((1 - p) * y).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1_micro  = 2 * precision * recall / (precision + recall + 1e-8)
        return {"f1_micro": f1_micro}
    return _fn