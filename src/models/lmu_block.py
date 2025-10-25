import torch
import torch.nn as nn
from .lmu import LMUFFT  # <-- comes from the file we just loaded

class LMUMemoryBlock(nn.Module):
    """
    Wraps the PyTorch LMU to (B,T,D)->(B,T,D), with residual + MLP scaffold.
    No custom LMU math here—just a thin adapter.
    """
    def __init__(self, d_model: int, memory_size: int, theta: int,
                 dropout: float = 0.1, mlp_ratio: float = 2.0):
        super().__init__()
        self.core = LMUFFT(
            input_size=d_model,
            hidden_size=d_model,
            memory_size=memory_size,
            theta=theta,
            seq_len=theta,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)       # (B,T,D)
        y, _ = self.core(h)     # (B,T,D), states ignored for this classifier
        x = x + self.drop1(y)
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x
