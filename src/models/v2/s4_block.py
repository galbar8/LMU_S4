import torch
from typing import Optional, Tuple

from src.models.s4.s4_block import S4Block
from src.models.v2.base import BaseSeqCore


class S4CoreAdapter(BaseSeqCore):
    """
    Adapter for S4 (Structured State Space) model.
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 64,
            channels: int = 1,
            bidirectional: bool = False,
            activation: str = 'gelu',
            dropout: float = 0.0,
            tie_dropout: bool = False,
            mode: str = 's4d',
            init: str = 'legs',
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            **kwargs
    ):
        """Initialize S4 core adapter.

        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            channels: Number of channels/heads
            bidirectional: Whether to use bidirectional convolution
            activation: Activation function
            dropout: Dropout rate
            tie_dropout: Whether to tie dropout across sequence length
            mode: Kernel mode ('s4d', 's4', 'diag', etc.)
            init: Initialization method for SSM parameters
            dt_min: Minimum timescale
            dt_max: Maximum timescale
            **kwargs: Additional arguments for S4Block
        """
        super().__init__()
        self.d_model = d_model

        self.s4_block = S4Block(
            d_model=d_model,
            d_state=d_state,
            channels=channels,
            bidirectional=bidirectional,
            activation=activation,
            dropout=dropout,
            tie_dropout=tie_dropout,
            transposed=False,  # Use (B, L, D) format
            mode=mode,
            init=init,
            dt_min=dt_min,
            dt_max=dt_max,
            **kwargs
        )

        self._state = None

    def forward(
            self,
            x: torch.Tensor,
            seq_len: Optional[int] = None,
            mask: Optional[torch.Tensor] = None,
            cache: Optional[dict] = None,
    ) -> torch.Tensor:
        """Forward pass through S4 block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            seq_len: Optional sequence length (not used for S4)
            mask: Optional attention mask (not used for S4)
            cache: Optional cache dictionary for state management

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # S4Block expects (B, L, D) format
        batch_size, T, _ = x.shape

        # Handle state from cache if provided
        state = None
        if cache is not None and 's4_state' in cache:
            state = cache['s4_state']

        # Compute lengths from mask if provided
        lengths = None
        if mask is not None:
            # Assuming mask is (batch, seq_len) with 1s for valid positions
            lengths = mask.sum(dim=1).long()

        # Forward through S4
        output, next_state = self.s4_block(
            x,
            lengths=lengths,
            state=state
        )

        # Update cache with new state if provided
        if cache is not None and next_state is not None:
            cache['s4_state'] = next_state

        return output

    def step(
            self,
            x: torch.Tensor,
            state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single timestep (for autoregressive generation).

        Args:
            x: Input tensor of shape (batch, d_model)
            state: Current state tensor

        Returns:
            Tuple of:
                - Output tensor of shape (batch, d_model)
                - Updated state tensor
        """
        if state is None:
            state = self._state
            if state is None:
                # Initialize default state
                state = self.default_state(x.shape[0], device=x.device)

        output, next_state = self.s4_block.step(x, state)
        self._state = next_state

        return output, next_state

    def default_state(
            self,
            batch_size: int,
            device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Create default initial state.

        Args:
            batch_size: Batch size
            device: Device for state tensor

        Returns:
            Default state tensor
        """
        return self.s4_block.default_state(batch_size, device=device)

    def reset_state(self):
        """Reset internal state."""
        self._state = None

    def setup_step(self, **kwargs):
        """Setup for efficient stepping mode."""
        self.s4_block.setup_step(**kwargs)

    @property
    def d_output(self) -> int:
        """Output dimension."""
        return self.d_model

