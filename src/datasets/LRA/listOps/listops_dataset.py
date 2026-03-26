"""LRA ListOps dataset."""
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from .listops_config import ListOpsConfig


def _default_vocab_path(root: Path) -> Path:
    return root / "data" / "vocab_listops.json"


def _tsv_path(root: Path, split: str) -> Path:
    if split == "train":
        return root / "data" / "basic_train.tsv"
    if split == "val":
        return root / "data" / "basic_val.tsv"
    if split == "test":
        return root / "data" / "basic_test.tsv"
    raise ValueError(f"Invalid split: {split}")


def _read_tsv(path: Path, limit: int | None = None) -> List[Tuple[int, str]]:
    samples: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # Skip header row (first line)
            if i == 0:
                continue

            if limit is not None and len(samples) >= limit:
                break

            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                # Warn about malformed line but continue
                print(f"Warning: Skipping malformed line {i+1} in {path.name}: expected 2 columns, got {len(parts)}")
                continue

            # NOTE: TSV format is "Source\tTarget" which means "sequence\tlabel"
            seq, y_str = parts

            # Try to parse the label
            try:
                label = int(y_str)
                if not (0 <= label <= 9):
                    print(f"Warning: Line {i+1} has label {label} outside expected range [0-9], skipping")
                    continue
                samples.append((label, seq))
            except ValueError as e:
                # This should only happen on header or corrupted data
                print(f"Warning: Line {i+1} has non-integer label '{y_str}', skipping (error: {e})")
                continue

    if not samples:
        raise ValueError(f"No valid samples found in {path}")

    return samples


def _build_vocab_from_samples(
    samples: List[Tuple[int, str]],
    pad_token: str,
    unk_token: str,
) -> Dict[str, int]:
    chars = set()
    for _, seq in samples:
        chars.update(seq)

    stoi: Dict[str, int] = {pad_token: 0, unk_token: 1}
    for ch in sorted(chars):
        if ch not in stoi:
            stoi[ch] = len(stoi)
    return stoi


def _save_vocab(path: Path, stoi: Dict[str, int], pad_token: str, unk_token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {"stoi": stoi, "pad_token": pad_token, "unk_token": unk_token},
            f,
            ensure_ascii=False,
            indent=2,
        )


def _load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["stoi"]


class ListOpsDataset(Dataset):
    """
    LRA ListOps dataset (TSV).

    Each sample is:
        label \\t sequence_string

    Returns:
        x: LongTensor (max_length,) token ids (char-level)
        y: LongTensor scalar label (0-9)
    """

    def __init__(self, cfg: ListOpsConfig):
        super().__init__()
        self.cfg = cfg
        root = Path(cfg.root)

        tsv = _tsv_path(root, cfg.split)
        if not tsv.exists():
            raise FileNotFoundError(f"ListOps TSV not found: {tsv}")

        # Load samples (optionally subset)
        self.samples = _read_tsv(tsv, limit=cfg.subset_size)

        # Vocab path resolution
        vocab_path = Path(cfg.vocab_path) if cfg.vocab_path is not None else _default_vocab_path(root)

        # Build vocab on train split if missing; else load
        if cfg.split == "train":
            if vocab_path.exists():
                self.stoi = _load_vocab(vocab_path)
            else:
                self.stoi = _build_vocab_from_samples(
                    self.samples,
                    pad_token=cfg.pad_token,
                    unk_token=cfg.unk_token,
                )
                _save_vocab(vocab_path, self.stoi, cfg.pad_token, cfg.unk_token)
        else:
            if not vocab_path.exists():
                raise FileNotFoundError(
                    f"Vocab not found for split={cfg.split}. "
                    f"Run train split once or provide vocab_path. Expected: {vocab_path}"
                )
            self.stoi = _load_vocab(vocab_path)

        self.pad_id = self.stoi[cfg.pad_token]
        self.unk_id = self.stoi[cfg.unk_token]

        # Fixed output properties
        self._seq_len = cfg.max_length

    def __len__(self) -> int:
        return len(self.samples)

    def _encode(self, seq: str) -> torch.Tensor:
        ids = [self.stoi.get(ch, self.unk_id) for ch in seq]

        # truncate/pad to fixed length
        ids = ids[: self.cfg.max_length]
        if len(ids) < self.cfg.max_length:
            ids.extend([self.pad_id] * (self.cfg.max_length - len(ids)))

        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int):
        y, seq = self.samples[idx]
        x = self._encode(seq)  # (max_length,)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def input_dim(self) -> int:
        """
        For token ids, the "input_dim" to the backbone is typically via an Embedding.
        So the raw dataset returns a 1D token id sequence.
        """
        return 1

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)
