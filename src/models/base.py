# src/models/blocks/base.py
from __future__ import annotations
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class MemoryBlock(nn.Module, ABC):
    """
    Unified memory module: takes (B,T,D) and returns (B,T,D).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
