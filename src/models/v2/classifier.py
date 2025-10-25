from typing import Callable
import torch
from torch import nn
import torch.nn.functional as F


class AttentivePool(nn.Module):
    """
    Optional attention pooling over time.
    Keeps core-agnostic; enable via BlockConfig.pool='attn'.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,T,D)
        w = self.score(x).squeeze(-1)                    # (B,T)
        w = F.softmax(w, dim=1)
        return torch.einsum("bt,btd->bd", w, x)          # weighted mean


class SeqClassifier(nn.Module):
    """
    Proj -> N×ResidualSeqBlock(core=LMU/S4) -> LN -> Pool -> Linear
    """
    def __init__(
        self,
        d_in: int,
        n_classes: int,
        d_model: int = 256,
        depth: int = 4,
        block_factory: Callable[[int, float], nn.Module] = None,
        droppath_final: float = 0.1,
        pool: str = "mean",  # 'mean' (default) or 'attn'
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)

        dps = torch.linspace(0, droppath_final, steps=depth).tolist()
        self.blocks = nn.ModuleList([block_factory(d_model, dps[i]) for i in range(depth)])

        self.norm = nn.LayerNorm(d_model)
        self.pool = pool
        if pool == "attn":
            self.attnpool = AttentivePool(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,D_in)
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        if self.pool == "attn":
            x = self.attnpool(x)
        else:
            x = x.mean(dim=1)
        return self.head(x)
