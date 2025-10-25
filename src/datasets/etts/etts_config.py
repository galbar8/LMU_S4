from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass
class ETTSConfig:
    root: str  # folder that contains ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
    which: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"] = "ETTh1"

    # sequence slicing
    seq_len: int = 96          # how many past timesteps to feed the model
    pred_len: int = 24         # how many future timesteps to predict

    # feature modes:
    #   "multivariate": model sees ALL features and predicts ALL features
    #   "target":       model sees ALL features but predicts ONLY target_col
    #   "target_only":  model sees ONLY target_col and predicts ONLY target_col
    feature_mode: Literal["multivariate", "target", "target_only"] = "target"

    target_col: str = "OT"     # default benchmark target is 'OT'

    # time-based split ratios (train, val, test)
    # must sum to 1.0
    split_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    # normalization
    normalize: Literal["zscore", "none"] = "zscore"

    # internal: which split this dataset instance is for
    split: Literal["train", "val", "test"] = "train"

    # we inject scaler stats from the dataloader factory
    mean_: Optional[Tuple[float, ...]] = None
    std_: Optional[Tuple[float, ...]] = None
