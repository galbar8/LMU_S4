"""LRA ListOps configuration."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ListOpsConfig:
    """
    Configuration for LRA ListOps dataset.

    ListOps is a long-range sequence benchmark where:
    - Each sample is a character-level expression (length up to ~2K)
    - Task: classify the result (0-9) based on the expression structure
    - Input is typically treated as a token sequence
    """

    # Data location
    root: str  # folder that contains a "data" folder with TSV files

    # Split
    split: Literal["train", "val", "test"] = "train"

    # Preprocessing
    max_length: int = 2000  # truncate / pad sequences to fixed length
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    # Tokenization style for ListOps TSV (sequence is char-level already)
    tokenize: Literal["char"] = "char"

    # Optional: use a subset for faster experimentation
    subset_size: Optional[int] = None  # if set, use only first N samples

    # Vocab handling
    vocab_path: Optional[str] = None
    # If None, default to "<root>/data/vocab_listops.json".
    # Vocab is created from *train* file and reused for val/test.
