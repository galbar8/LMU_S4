"""Unified block configuration factory for creating LMU and S4 block configs."""
from typing import Callable
from src.models.v2.build_model import BlockConfig


def make_block_cfg_ctor(
    kind: str,
    *,
    # Common parameters
    dropout: float = 0.2,
    mlp_ratio: float = 2.0,
    droppath_final: float = 0.1,
    layerscale_init: float = 1e-2,
    residual_gain: float = 1.0,
    pool: str = "mean",
    # LMU-specific parameters
    memory_size: int = 256,
    # S4-specific parameters
    d_state: int = 64,
    channels: int = 1,
    bidirectional: bool = False,
    mode: str = "s4d",
    dt_min: float = 0.001,
    dt_max: float = 0.1,
) -> Callable[[int], BlockConfig]:
    """
    Create a block configuration constructor function.

    This factory allows for easy comparison between LMU and S4 models by
    changing a single argument (kind="lmu" or kind="s4").

    Args:
        kind: Type of block - "lmu" or "s4"
        dropout: Dropout rate
        mlp_ratio: MLP expansion ratio
        droppath_final: Final drop path rate
        layerscale_init: Layer scale initialization value
        residual_gain: Residual connection gain
        pool: Pooling method - "mean" or "attn"
        memory_size: Memory size for LMU
        d_state: State dimension for S4
        channels: Number of channels for S4
        bidirectional: Whether to use bidirectional S4
        mode: S4 mode - "s4d", "s4", or "diag"
        dt_min: Minimum timestep for S4
        dt_max: Maximum timestep for S4

    Returns:
        Function that takes theta (sequence length) and returns BlockConfig

    Example:
        >>> lmu_cfg_ctor = make_block_cfg_ctor("lmu", memory_size=256)
        >>> block_cfg = lmu_cfg_ctor(theta=500)

        >>> s4_cfg_ctor = make_block_cfg_ctor("s4", d_state=64, bidirectional=True)
        >>> block_cfg = s4_cfg_ctor(theta=500)
    """
    def block_cfg_ctor(theta: int) -> BlockConfig:
        if kind.lower() == "lmu":
            return BlockConfig(
                kind="lmu",
                memory_size=memory_size,
                theta=theta,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                droppath_final=droppath_final,
                layerscale_init=layerscale_init,
                residual_gain=residual_gain,
                pool=pool,
            )
        elif kind.lower() == "s4":
            return BlockConfig(
                kind="s4",
                d_state=d_state,
                channels=channels,
                bidirectional=bidirectional,
                mode=mode,
                dt_min=dt_min,
                dt_max=dt_max,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                droppath_final=droppath_final,
                layerscale_init=layerscale_init,
                residual_gain=residual_gain,
                pool=pool,
            )
        else:
            raise ValueError(
                f"Unknown block kind: {kind}. Must be 'lmu' or 's4'."
            )

    return block_cfg_ctor


def make_lmu_block_cfg_ctor(
    memory_size: int = 256,
    dropout: float = 0.2,
    mlp_ratio: float = 2.0,
    droppath_final: float = 0.1,
    layerscale_init: float = 1e-2,
    residual_gain: float = 1.0,
    pool: str = "mean",
) -> Callable[[int], BlockConfig]:
    """
    Convenience function to create LMU block config constructor.

    Args:
        memory_size: Memory size for LMU
        dropout: Dropout rate
        mlp_ratio: MLP expansion ratio
        droppath_final: Final drop path rate
        layerscale_init: Layer scale initialization value
        residual_gain: Residual connection gain
        pool: Pooling method - "mean" or "attn"

    Returns:
        Function that takes theta (sequence length) and returns BlockConfig
    """
    return make_block_cfg_ctor(
        kind="lmu",
        memory_size=memory_size,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        droppath_final=droppath_final,
        layerscale_init=layerscale_init,
        residual_gain=residual_gain,
        pool=pool,
    )


def make_s4_block_cfg_ctor(
    d_state: int = 64,
    channels: int = 1,
    bidirectional: bool = False,
    mode: str = "s4d",
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    dropout: float = 0.2,
    mlp_ratio: float = 2.0,
    droppath_final: float = 0.1,
    layerscale_init: float = 1e-2,
    residual_gain: float = 1.0,
    pool: str = "mean",
) -> Callable[[int], BlockConfig]:
    """
    Convenience function to create S4 block config constructor.

    Args:
        d_state: State dimension for S4
        channels: Number of channels for S4
        bidirectional: Whether to use bidirectional S4
        mode: S4 mode - "s4d", "s4", or "diag"
        dt_min: Minimum timestep for S4
        dt_max: Maximum timestep for S4
        dropout: Dropout rate
        mlp_ratio: MLP expansion ratio
        droppath_final: Final drop path rate
        layerscale_init: Layer scale initialization value
        residual_gain: Residual connection gain
        pool: Pooling method - "mean" or "attn"

    Returns:
        Function that takes theta (sequence length) and returns BlockConfig
    """
    return make_block_cfg_ctor(
        kind="s4",
        d_state=d_state,
        channels=channels,
        bidirectional=bidirectional,
        mode=mode,
        dt_min=dt_min,
        dt_max=dt_max,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        droppath_final=droppath_final,
        layerscale_init=layerscale_init,
        residual_gain=residual_gain,
        pool=pool,
    )

