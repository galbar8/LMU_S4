import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep) / keep
        return x * rnd


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-2):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 2.0, p: float = 0.2):
        super().__init__()
        h = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, 2 * h)  # GEGLU
        self.fc2 = nn.Linear(h, d_model)
        self.drop = nn.Dropout(p)
        # Zero-init last projection → smoother, identity-ish start on small datasets like ESC-50
        nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc1(x).chunk(2, dim=-1)
        x = F.gelu(b) * a
        x = self.drop(x)
        return self.fc2(x)


class BaseSeqCore(nn.Module):
    """
    Protocol for core blocks (LMU/S4 adapters).
    Expects x: (B, T, D); may receive seq_len, mask, cache; returns y: (B, T, D).
    """
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class ResidualSeqBlock(nn.Module):
    """
    Pre-norm residual scaffold around a sequence core:
      LN -> Core -> Dropout -> LayerScale -> DropPath -> +residual (× residual_gain)
      LN -> GatedMLP -> Dropout -> LayerScale -> DropPath -> +residual (× residual_gain)
    """
    def __init__(
        self,
        core: BaseSeqCore,
        d_model: int,
        dropout: float = 0.2,
        mlp_ratio: float = 2.0,
        droppath: float = 0.1,
        layerscale_init: float = 1e-2,
        residual_gain: float = 1.0,
    ):
        super().__init__()
        self.core = core
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.dp1 = DropPath(droppath)
        self.ls1 = LayerScale(d_model, layerscale_init)

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = GatedMLP(d_model, mlp_ratio, dropout)
        self.drop2 = nn.Dropout(dropout)
        self.dp2 = DropPath(droppath)
        self.ls2 = LayerScale(d_model, layerscale_init)

        self.residual_gain = residual_gain

    def forward(
        self,
        x: torch.Tensor,                          # (B,T,D)
        mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
    ) -> torch.Tensor:
        _, T, _ = x.shape

        # Core branch
        h = self.norm1(x)
        y = self.core(h, seq_len=T, mask=mask, cache=cache)    # (B,T,D)
        x = x + self.residual_gain * self.dp1(self.ls1(self.drop1(y)))

        # MLP branch
        x = x + self.residual_gain * self.dp2(self.ls2(self.drop2(self.mlp(self.norm2(x)))))
        return x
