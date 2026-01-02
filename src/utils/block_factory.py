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
    # Mamba-specific parameters
    mamba_d_state: int = 16,
    mamba_expand_factor: int = 2,
    mamba_d_conv: int = 4,
    mamba_dt_min: float = 0.001,
    mamba_dt_max: float = 0.1,
    use_external_mlp: bool = True,
) -> Callable[[int], BlockConfig]:
    """
    Create a block configuration constructor function.

    This factory allows for easy comparison between LMU, S4, and Mamba models by
    changing a single argument (kind="lmu", kind="s4", or kind="mamba").

    Args:
        kind: Type of block - "lmu", "s4", or "mamba"
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
        mamba_d_state: State dimension for Mamba
        mamba_expand_factor: Expansion factor for Mamba
        mamba_d_conv: Convolution width for Mamba
        mamba_dt_min: Minimum timestep for Mamba
        mamba_dt_max: Maximum timestep for Mamba
        use_external_mlp: Whether to use external MLP (for Mamba)

    Returns:
        Function that takes theta (sequence length) and returns BlockConfig
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
        elif kind.lower() == "mamba":
            return BlockConfig(
                kind="mamba",
                mamba_d_state=mamba_d_state,
                mamba_expand_factor=mamba_expand_factor,
                mamba_d_conv=mamba_d_conv,
                mamba_dt_min=mamba_dt_min,
                mamba_dt_max=mamba_dt_max,
                use_external_mlp=use_external_mlp,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
                droppath_final=droppath_final,
                layerscale_init=layerscale_init,
                residual_gain=residual_gain,
                pool=pool,
            )
        else:
            raise ValueError(
                f"Unknown block kind: {kind}. Must be 'lmu', 's4', or 'mamba'."
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


def make_mamba_block_cfg_ctor(
    d_state: int = 16,
    expand_factor: int = 2,
    d_conv: int = 4,
    dt_min: float = 0.001,
    dt_max: float = 0.1,
    use_external_mlp: bool = True,
    dropout: float = 0.2,
    mlp_ratio: float = 2.0,
    droppath_final: float = 0.1,
    layerscale_init: float = 1e-2,
    residual_gain: float = 1.0,
    pool: str = "mean",
) -> Callable[[int], BlockConfig]:
    """
    Convenience function to create Mamba block config constructor.

    Args:
        d_state: State dimension for Mamba
        expand_factor: Expansion factor for Mamba
        d_conv: Convolution width for Mamba
        dt_min: Minimum timestep for Mamba
        dt_max: Maximum timestep for Mamba
        use_external_mlp: Whether to use external MLP
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
        kind="mamba",
        mamba_d_state=d_state,
        mamba_expand_factor=expand_factor,
        mamba_d_conv=d_conv,
        mamba_dt_min=dt_min,
        mamba_dt_max=dt_max,
        use_external_mlp=use_external_mlp,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        droppath_final=droppath_final,
        layerscale_init=layerscale_init,
        residual_gain=residual_gain,
        pool=pool,
    )
