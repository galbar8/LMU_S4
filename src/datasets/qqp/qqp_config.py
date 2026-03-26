"""
Quora Question Pairs (QQP) configuration.

Dataset file expected:
- questions.csv with columns: id, qid1, qid2, question1, question2, is_duplicate
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class QQPConfig:
    """
    Configuration for QQP dataset.

    Task:
    - Input: (question1, question2)
    - Output: is_duplicate ∈ {0,1}

    This implementation is "lightweight NLP":
    - whitespace/punctuation tokenization (regex-based)
    - train-only vocab build
    - pad/truncate to fixed max_len per question
    """

    # Data location
    root: str                   # folder containing questions.csv
    csv_filename: str = "questions.csv"

    # Split handling
    split: Literal["train", "val", "test"] = "train"
    seed: int = 42
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Optional: use a subset for faster experimentation (applied per split)
    subset_size: Optional[int] = None

    # Text preprocessing / tokenization
    lowercase: bool = True
    max_len: int = 64           # tokens per question (fixed sequence length)
    max_vocab: int = 50000      # top-K tokens from train split
    min_freq: int = 2           # drop rare tokens in train vocab

    # Special tokens
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    # Caching split indices to disk (recommended for reproducibility + speed)
    cache_splits: bool = True
