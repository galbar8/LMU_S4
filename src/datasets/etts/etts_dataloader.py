from __future__ import annotations
from typing import Tuple, Literal
from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .etts_config import ETTSConfig
from .etts_dataset import ETTSLMDataset


def _compute_train_stats(
    root: str,
    which: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
    split_ratio: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read full CSV once, slice the train part, drop 'date',
    return (mean, std) over train only, per feature.
    """
    csv_path = Path(root) / f"{which}.csv"
    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError(f"{csv_path} missing 'date' column")

    data_df = df.drop(columns=["date"])
    data_np = data_df.to_numpy(dtype=np.float32)  # (T, D_all)

    T_all = data_np.shape[0]
    r_train, r_val, r_test = split_ratio
    n_train = int(math.floor(T_all * r_train))

    train_np = data_np[:n_train]  # only train slice

    mean = train_np.mean(axis=0)           # (D_all,)
    std = train_np.std(axis=0) + 1e-6      # (D_all,)

    return mean.astype(np.float32), std.astype(np.float32)


def make_etts_loaders(
    data_root: str,
    which: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"] = "ETTh1",
    batch_size: int = 64,
    num_workers: int = 4,
    seq_len: int = 96,
    pred_len: int = 24,
    feature_mode: Literal["multivariate", "target", "target_only"] = "target",
    target_col: str = "OT",
    split_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    normalize: Literal["zscore", "none"] = "zscore",
    pin_memory: bool = True,
    persistent_workers: bool = False,
):
    """
    Build train/val/test DataLoaders for ETTh1/ETTh2/ETTm1/ETTm2.

    Each batch:
      x: (B, seq_len, D_in)
      y: (B, pred_len, D_out)
    """

    # 1) compute train stats for normalization
    mean, std = _compute_train_stats(
        root=data_root,
        which=which,
        split_ratio=split_ratio,
    )

    # 2) create configs
    base_kwargs = dict(
        root=data_root,
        which=which,
        seq_len=seq_len,
        pred_len=pred_len,
        feature_mode=feature_mode,
        target_col=target_col,
        split_ratio=split_ratio,
        normalize=normalize,
        mean_=tuple(float(m) for m in mean),
        std_=tuple(float(s) for s in std),
    )

    cfg_train = ETTSConfig(**base_kwargs, split="train")
    cfg_val   = ETTSConfig(**base_kwargs, split="val")
    cfg_test  = ETTSConfig(**base_kwargs, split="test")

    # 3) datasets
    train_ds = ETTSLMDataset(cfg_train)
    val_ds   = ETTSLMDataset(cfg_val)
    test_ds  = ETTSLMDataset(cfg_test)

    # 4) reproducible shuffling for train
    g = torch.Generator().manual_seed(0)

    use_persistent = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        persistent_workers=use_persistent,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )

    return train_loader, val_loader, test_loader
