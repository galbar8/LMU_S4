from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable


class SeqClassifier(nn.Module):
    """
    Projection -> N×MemoryBlock -> LN -> GAP(time) -> Linear Head
    """

    def __init__(
            self,
            d_in: int,
            n_classes: int,
            d_model: int = 256,
            depth: int = 4,
            block_factory: Callable[[int], nn.Module] = None,
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([block_factory(d_model) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B,T,D)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
