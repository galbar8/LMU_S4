from __future__ import annotations
import torch
import torch.nn as nn

class EMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module):
        self.backup = {}
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.backup[k] = v.detach().clone()
                v.copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if not self.backup:
            return
        for k, v in model.state_dict().items():
            if k in self.backup:
                v.copy_(self.backup[k])
        self.backup = {}
