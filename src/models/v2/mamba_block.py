from __future__ import annotations

from typing import Optional
import torch

from src.models.v2.base import BaseSeqCore
from mambapy.mamba import Mamba, MambaConfig

class MambaCoreAdapter(BaseSeqCore):
    """
    Mamba adapter (mambapy) with (B, T, D) -> (B, T, D).

    We intentionally expose only the main architectural knobs:
      - d_state: SSM state size
      - expand_factor: inner expansion factor
      - d_conv: local convolution width
      - dt_min/dt_max: timescale range (optional but useful for comparability)
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        expand_factor: int = 2,
        d_conv: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        cfg = MambaConfig(
            d_model=d_model,
            n_layers=1,              # one "core" per ResidualSeqBlock
            d_state=d_state,
            expand_factor=expand_factor,
            d_conv=d_conv,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.mamba = Mamba(cfg)

        # Learnable output scaling for numerical stability
        # Initialized very small (0.01) to prevent exploding activations
        # This is more conservative than 0.1 and helps after initial epochs
        self.output_scale = torch.nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
    ) -> torch.Tensor:
        out = self.mamba(x)
        # Apply learnable scaling - prevents exploding activations
        # while allowing model to adapt the scale during training
        return out * self.output_scale

    @property
    def d_output(self) -> int:
        return self.d_model