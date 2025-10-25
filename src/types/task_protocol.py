from __future__ import annotations
from typing import Protocol, Tuple, Any, Optional
import torch


class TaskProtocol(Protocol):
    """
    Defines the interface required by Trainer for any dataset task.

    A task must provide:
    - A `problem_type` attribute: either "multiclass" or "multilabel".
    - A `make_loaders` method returning train/val/(test) DataLoaders.
    - Helper methods to infer input/output dimensions.
    """

    problem_type: str  # "multiclass" or "multilabel"

    def make_loaders(
        self,
        data_root: str,
        batch_size: int,
        **kwargs: Any,
    ) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        Optional[torch.utils.data.DataLoader],
    ]:
        """Return train, val, test loaders."""

    def infer_input_dim(self, args: dict) -> int:
        """Return model input dimension (e.g., number of features or channels)."""

    def infer_num_classes(self, args: dict) -> int:
        """Return number of classes (for multiclass or multilabel tasks)."""

    def infer_theta(self, args: dict) -> int:
        """Return the sequence length or other temporal parameter (theta)."""
