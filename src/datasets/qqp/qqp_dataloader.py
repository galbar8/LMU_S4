"""
QQP dataloader factory.

Creates train/val/test splits deterministically and builds a train-only vocab.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .qqp_config import QQPConfig
from .qqp_dataset import QQPDataset, Vocab, build_vocab_from_texts


def _make_split_indices(
    n: int,
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0,1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0.")

    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return train_idx, val_idx, test_idx


def _splits_cache_path(root: str, seed: int, val_ratio: float, test_ratio: float) -> Path:
    safe = f"qqp_splits_seed{seed}_v{val_ratio:.3f}_t{test_ratio:.3f}.npz"
    return Path(root) / safe


def make_qqp_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    lowercase: bool = True,
    max_len: int = 64,
    max_vocab: int = 50000,
    min_freq: int = 2,
    subset_size: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    cache_splits: bool = True,
    csv_filename: str = "questions.csv",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders for QQP.

    - Deterministic split by seed
    - Vocab built only from train questions (question1+question2)
    - All splits use the same vocab
    - Samples return x=(2,max_len) token ids, y in {0,1}
    """
    csv_path = Path(root) / csv_filename
    if not csv_path.exists():
        raise FileNotFoundError(f"QQP CSV not found: {csv_path}")

    # Read just enough to determine N and build vocab later
    df = pd.read_csv(csv_path)
    n = len(df)

    # Load or create split indices
    cache_path = _splits_cache_path(root, seed, val_ratio, test_ratio)
    if cache_splits and cache_path.exists():
        npz = np.load(cache_path)
        train_idx = npz["train_idx"]
        val_idx = npz["val_idx"]
        test_idx = npz["test_idx"]
    else:
        train_idx, val_idx, test_idx = _make_split_indices(
            n=n, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio
        )
        if cache_splits:
            np.savez_compressed(cache_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # Build train-only vocab from train indices (question1 + question2)
    train_df = df.iloc[train_idx]
    train_texts = (
        train_df["question1"].astype("object").tolist()
        + train_df["question2"].astype("object").tolist()
    )

    vocab: Vocab = build_vocab_from_texts(
        texts=train_texts,
        lowercase=lowercase,
        pad_token="<pad>",
        unk_token="<unk>",
        max_vocab=max_vocab,
        min_freq=min_freq,
    )

    train_cfg = QQPConfig(
        root=root,
        csv_filename=csv_filename,
        split="train",
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        subset_size=subset_size,
        lowercase=lowercase,
        max_len=max_len,
        max_vocab=max_vocab,
        min_freq=min_freq,
        cache_splits=cache_splits,
    )
    val_cfg = QQPConfig(
        root=root,
        csv_filename=csv_filename,
        split="val",
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        subset_size=subset_size,
        lowercase=lowercase,
        max_len=max_len,
        max_vocab=max_vocab,
        min_freq=min_freq,
        cache_splits=cache_splits,
    )
    test_cfg = QQPConfig(
        root=root,
        csv_filename=csv_filename,
        split="test",
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        subset_size=None,
        lowercase=lowercase,
        max_len=max_len,
        max_vocab=max_vocab,
        min_freq=min_freq,
        cache_splits=cache_splits,
    )

    train_ds = QQPDataset(train_cfg, vocab=vocab, indices=train_idx)
    val_ds = QQPDataset(val_cfg, vocab=vocab, indices=val_idx)
    test_ds = QQPDataset(test_cfg, vocab=vocab, indices=test_idx)

    # Handle persistent_workers (only if num_workers > 0)
    pw = persistent_workers if num_workers > 0 else False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw,
        drop_last=False,
    )

    print("QQP Loaders Created:")
    print(f"  CSV: {csv_path}")
    print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_ds)} samples, {len(test_loader)} batches")
    print(f"  Seed: {seed} | val_ratio={val_ratio} | test_ratio={test_ratio}")
    print(f"  max_len per question: {max_len} tokens")
    print(f"  vocab_size: {train_ds.vocab_size} (max_vocab={max_vocab}, min_freq={min_freq})")
    print(f"  x shape: (max_len * 2 + 1,) = ({max_len * 2 + 1},) concatenated [q1, <seq>, q2]; y ∈ {{0,1}}")

    return train_loader, val_loader, test_loader
