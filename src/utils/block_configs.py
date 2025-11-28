"""Block configuration helpers for LMU and S4."""
from __future__ import annotations
from src.models.v2.build_model import BlockConfig


def create_lmu_block_cfg_ctor(
    dropout: float,
    mlp_ratio: float,
    droppath_final: float,
    layerscale_init: float,
    residual_gain: float,
    pool: str,
    memory_size: int = 256
):
    """
    Create LMU block config constructor.

    Args:
        dropout: Dropout probability
        mlp_ratio: MLP expansion ratio
        droppath_final: Final droppath probability
        layerscale_init: LayerScale initialization value
        residual_gain: Residual connection gain
        pool: Pooling method ('mean' or 'attn')
        memory_size: LMU memory size

    Returns:
        Function that takes theta and returns BlockConfig
    """
    def block_cfg_ctor(theta: int):
        return BlockConfig(
            kind="lmu",
            memory_size=memory_size,
            theta=theta,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            droppath_final=droppath_final,
            layerscale_init=layerscale_init,
            residual_gain=residual_gain,
            pool=pool
        )
    return block_cfg_ctor


def create_s4_block_cfg_ctor(
    dropout: float,
    mlp_ratio: float,
    droppath_final: float,
    layerscale_init: float,
    residual_gain: float,
    pool: str,
    d_state: int = 64,
    mode: str = "s4d",
    bidirectional: bool = False
):
    """
    Create S4 block config constructor.

    Args:
        dropout: Dropout probability
        mlp_ratio: MLP expansion ratio
        droppath_final: Final droppath probability
        layerscale_init: LayerScale initialization value
        residual_gain: Residual connection gain
        pool: Pooling method ('mean' or 'attn')
        d_state: S4 state dimensionality
        mode: S4 mode ('s4d', 's4', 'diag')
        bidirectional: Use bidirectional S4

    Returns:
        Function that takes theta and returns BlockConfig
    """
    def block_cfg_ctor(theta: int):
        return BlockConfig(
            kind="s4",
            d_state=d_state,
            channels=1,
            bidirectional=bidirectional,
            mode=mode,
            dt_min=0.001,
            dt_max=0.1,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            droppath_final=droppath_final,
            layerscale_init=layerscale_init,
            residual_gain=residual_gain,
            pool=pool,
        )
    return block_cfg_ctor

