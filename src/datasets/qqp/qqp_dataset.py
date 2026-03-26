"""
Quora Question Pairs (QQP) dataset.

Reads questions.csv and returns:
    x: LongTensor (max_len * 2,) # concatenated token ids [q1, q2]
    y: LongTensor scalar         # 0/1 is_duplicate

Vocab is expected to be built from train and shared across splits.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import re
from torch.utils.data import Dataset

from .qqp_config import QQPConfig


_TOKEN_PATTERN = r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]"

def _tokenize(text: str, lowercase: bool) -> List[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        text = ""
    text = str(text)
    if lowercase:
        text = text.lower()
    return re.findall(_TOKEN_PATTERN, text)


@dataclass(frozen=True)
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    pad_id: int
    unk_id: int
    sep_id: int

    @property
    def size(self) -> int:
        return len(self.id_to_token)


def build_vocab_from_texts(
    texts: List[str],
    lowercase: bool,
    pad_token: str,
    unk_token: str,
    max_vocab: int,
    min_freq: int,
    sep_token: str = "<sep>",
) -> Vocab:
    """
    Build vocab from a list of texts (train split only).
    """
    from collections import Counter

    counter = Counter()
    for t in texts:
        counter.update(_tokenize(t, lowercase=lowercase))

    # Reserve special tokens at the beginning
    id_to_token: List[str] = [pad_token, unk_token, sep_token]
    token_to_id: Dict[str, int] = {pad_token: 0, unk_token: 1, sep_token: 2}

    # Most common tokens with min_freq, capped by max_vocab
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in token_to_id:
            continue
        if len(id_to_token) >= max_vocab:
            break
        token_to_id[tok] = len(id_to_token)
        id_to_token.append(tok)

    return Vocab(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_id=token_to_id[pad_token],
        unk_id=token_to_id[unk_token],
        sep_id=token_to_id[sep_token],
    )


def encode_text(
    text: str,
    vocab: Vocab,
    lowercase: bool,
    max_len: int,
) -> torch.Tensor:
    toks = _tokenize(text, lowercase=lowercase)
    ids = [vocab.token_to_id.get(tok, vocab.unk_id) for tok in toks]

    # pad/truncate to fixed length
    if len(ids) >= max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [vocab.pad_id] * (max_len - len(ids))

    return torch.tensor(ids, dtype=torch.long)  # (max_len,)


class QQPDataset(Dataset):
    """
    QQP Dataset.

    Returns:
        x: LongTensor (max_len * 2,) where x = [q1_ids, q2_ids] concatenated
        y: LongTensor scalar (0/1)
    """
    def __init__(
        self,
        cfg: QQPConfig,
        *,
        vocab: Vocab,
        indices: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.vocab = vocab

        csv_path = Path(cfg.root) / cfg.csv_filename
        if not csv_path.exists():
            raise FileNotFoundError(f"QQP CSV not found: {csv_path}")

        # Load once per dataset object (simple + reliable)
        df = pd.read_csv(csv_path)

        required = {"question1", "question2", "is_duplicate"}
        missing = required.difference(set(df.columns))
        if missing:
            raise ValueError(f"QQP CSV is missing required columns: {sorted(missing)}")

        # Restrict rows if indices provided (split selection)
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)

        # Optional subset (per split)
        if cfg.subset_size is not None and cfg.subset_size < len(df):
            df = df.iloc[: cfg.subset_size].reset_index(drop=True)

        # Store arrays for fast indexing
        self.q1 = df["question1"].astype("object").tolist()
        self.q2 = df["question2"].astype("object").tolist()
        self.y = df["is_duplicate"].astype("int64").to_numpy()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        q1_ids = encode_text(
            self.q1[idx],
            vocab=self.vocab,
            lowercase=self.cfg.lowercase,
            max_len=self.cfg.max_len,
        )
        q2_ids = encode_text(
            self.q2[idx],
            vocab=self.vocab,
            lowercase=self.cfg.lowercase,
            max_len=self.cfg.max_len,
        )

        # Insert separator token between questions: [q1, <sep>, q2]
        sep_token = torch.tensor([self.vocab.sep_id], dtype=torch.long)
        x = torch.cat([q1_ids, sep_token, q2_ids], dim=0)  # (max_len * 2 + 1,)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)

        return x, y

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def seq_len(self) -> int:
        # Concatenated sequence with separator: [q1, <sep>, q2]
        return self.cfg.max_len * 2 + 1

    @property
    def vocab_size(self) -> int:
        return self.vocab.size

    @property
    def pad_id(self) -> int:
        return self.vocab.pad_id

    @property
    def unk_id(self) -> int:
        return self.vocab.unk_id
