from typing import Optional
import torch
from src.models.lmu import LMUFFT
from src.models.v2.base import BaseSeqCore


class LMUCoreAdapter(BaseSeqCore):
    """
    Thin adapter: keeps (B,T,D) API; threads seq_len; ignores mask/cache (not needed for ESC-50).
    """
    def __init__(self, d_model: int, memory_size: int, theta: int, seq_len_hint: Optional[int] = None):
        super().__init__()
        self.core = LMUFFT(
            input_size=d_model,
            hidden_size=d_model,
            memory_size=memory_size,
            theta=theta,
            seq_len=seq_len_hint or theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
    ) -> torch.Tensor:
        if seq_len is not None and hasattr(self.core, "seq_len"):
            self.core.seq_len = seq_len
        y, _state = self.core(x)   # (B,T,D)
        return y
