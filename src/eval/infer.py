from __future__ import annotations
import torch
from typing import Tuple, List

@torch.no_grad()
def predict_loader(model, loader, device, amp_autocast, amp: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      logits: (N, C)
      labels: (N,)
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    # Get device type string for amp_autocast
    device_type = device.type if hasattr(device, 'type') else str(device)

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            xb, yb = batch[:2]
        else:
            xb, yb = batch["x"], batch["y"]
        xb = xb.to(device, non_blocking=True)
        with amp_autocast(device_type, amp):
            logits = model(xb)
        all_logits.append(logits.cpu())
        all_labels.append(yb.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)
