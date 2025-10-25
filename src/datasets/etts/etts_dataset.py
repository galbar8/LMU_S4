from __future__ import annotations
from typing import List
from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .etts_config import ETTSConfig


class ETTSLMDataset(Dataset):
    """
    Sliding-window ETT dataset.

    Returns:
      x: FloatTensor (seq_len, D_in)
      y: FloatTensor (pred_len, D_out)

    Notes:
    - We drop the 'date' column and work only on numeric columns.
    - We split by contiguous time segments: first 70% train, next 10% val, last 20% test
      (configurable via cfg.split_ratio).
    - Normalization ('zscore') is per-feature using TRAIN statistics only.
    - Windowing: for each index i in this split,
        x = data[i : i+seq_len]
        y = data[i+seq_len : i+seq_len+pred_len]
      so prediction starts *right after* x ends.
    """

    def __init__(self, cfg: ETTSConfig):
        super().__init__()
        self.cfg = cfg

        root = Path(cfg.root)
        csv_path = root / f"{cfg.which}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Could not find {csv_path}. Expected files like ETTh1.csv in {root}"
            )

        # load csv
        df = pd.read_csv(csv_path)

        # basic sanity
        if "date" not in df.columns:
            raise ValueError(f"{csv_path} missing 'date' column")
        if cfg.target_col not in df.columns:
            raise ValueError(f"{csv_path} missing target column {cfg.target_col}")

        # keep numeric columns only (drop 'date')
        # NOTE: df.dtypes might have 'object' leftovers; coerce to float
        data_df = df.drop(columns=["date"])
        data_np = data_df.to_numpy(dtype=np.float32)  # shape (T, D_all)

        self.feature_names: List[str] = list(data_df.columns)  # store names for debug
        self.target_idx = self.feature_names.index(cfg.target_col)

        # figure out split indices
        T_all = data_np.shape[0]
        r_train, r_val, r_test = cfg.split_ratio
        n_train = int(math.floor(T_all * r_train))
        n_val = int(math.floor(T_all * r_val))
        n_test = T_all - n_train - n_val
        # sanity
        assert n_train > 0 and n_val > 0 and n_test > 0, "bad split ratios"

        idx_train = (0, n_train)
        idx_val = (n_train, n_train + n_val)
        idx_test = (n_train + n_val, T_all)

        if cfg.split == "train":
            start, end = idx_train
        elif cfg.split == "val":
            start, end = idx_val
        elif cfg.split == "test":
            start, end = idx_test
        else:
            raise ValueError("split must be train/val/test")

        # slice the relevant split
        split_np = data_np[start:end]  # shape (T_split, D_all)

        # apply normalization: use cfg.mean_/cfg.std_ which should be computed on train
        if cfg.normalize == "zscore":
            if cfg.mean_ is None or cfg.std_ is None:
                raise ValueError("Expected mean_/std_ in cfg for zscore normalization")
            mean = np.asarray(cfg.mean_, dtype=np.float32)  # (D_all,)
            std = np.asarray(cfg.std_, dtype=np.float32) + 1e-6
            split_np = (split_np - mean) / std

        # choose what goes into x and y
        # multivariate:   x all feats, y all feats
        # target:         x all feats, y only target_col
        # target_only:    x only target_col, y only target_col
        if cfg.feature_mode == "multivariate":
            self.X_all = split_np  # (T_split, D_all)
            self.Y_all = split_np  # same
            self.x_dim = split_np.shape[1]
            self.y_dim = split_np.shape[1]
            self._x_selector = slice(None)
            self._y_selector = slice(None)
        elif cfg.feature_mode == "target":
            self.X_all = split_np
            self.Y_all = split_np[:, [self.target_idx]]  # (T_split, 1)
            self.x_dim = split_np.shape[1]
            self.y_dim = 1
            self._x_selector = slice(None)
            self._y_selector = [self.target_idx]
        elif cfg.feature_mode == "target_only":
            only_target = split_np[:, [self.target_idx]]  # (T_split,1)
            self.X_all = only_target
            self.Y_all = only_target
            self.x_dim = 1
            self.y_dim = 1
            self._x_selector = [self.target_idx]  # NOTE: not actually used in this mode
            self._y_selector = [self.target_idx]
        else:
            raise ValueError("feature_mode must be multivariate/target/target_only")

        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len

        # precompute max starting index for a full (seq_len + pred_len) window
        self.T_split = self.X_all.shape[0]
        self.max_start = self.T_split - (self.seq_len + self.pred_len) + 1
        if self.max_start <= 0:
            raise ValueError(
                f"Split too short ({self.T_split} timesteps) for seq_len={self.seq_len} "
                f"and pred_len={self.pred_len}"
            )

    def __len__(self) -> int:
        return self.max_start

    def __getitem__(self, idx: int):
        s = idx
        e_x = s + self.seq_len
        e_y = e_x + self.pred_len

        x_win = self.X_all[s:e_x]         # (seq_len, D_in)
        y_win = self.Y_all[e_x:e_y]       # (pred_len, D_out)

        x_t = torch.from_numpy(x_win.astype(np.float32))
        y_t = torch.from_numpy(y_win.astype(np.float32))

        return x_t, y_t
