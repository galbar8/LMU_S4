from dataclasses import dataclass
from src.models.v2.classifier import SeqClassifier
from src.models.v2.base import ResidualSeqBlock
from src.models.v2.lmu_block import LMUCoreAdapter


@dataclass
class BlockConfig:
    kind: str                     # "lmu" or "s4" (S4 adapter to be added later)
    # LMU params
    memory_size: int = 256
    theta: int = 500
    # Common params (used inside blocks/classifier)
    dropout: float = 0.2
    mlp_ratio: float = 2.0
    droppath_final: float = 0.1
    layerscale_init: float = 1e-2
    residual_gain: float = 1.0
    pool: str = "mean"            # 'mean' or 'attn'

def make_block_factory(cfg: BlockConfig):
    def factory(d_model: int, droppath: float):
        if cfg.kind.lower() == "lmu":
            core = LMUCoreAdapter(
                d_model,
                memory_size=cfg.memory_size,
                theta=cfg.theta,
                seq_len_hint=cfg.theta,
            )
        else:
            # S4 adapter to be implemented with the same BaseSeqCore signature
            raise ValueError(f"Unknown block kind: {cfg.kind}")

        return ResidualSeqBlock(
            core=core,
            d_model=d_model,
            dropout=cfg.dropout,
            mlp_ratio=cfg.mlp_ratio,
            droppath=droppath,
            layerscale_init=cfg.layerscale_init,
            residual_gain=cfg.residual_gain,
        )
    return factory

def build_model(
    d_in: int,
    n_classes: int,
    d_model: int = 256,
    depth: int = 4,
    block_cfg: BlockConfig = BlockConfig(kind="lmu"),
):
    block_factory = make_block_factory(block_cfg)
    return SeqClassifier(
        d_in=d_in,
        n_classes=n_classes,
        d_model=d_model,
        depth=depth,
        block_factory=block_factory,
        droppath_final=block_cfg.droppath_final,
        pool=block_cfg.pool,
    )
