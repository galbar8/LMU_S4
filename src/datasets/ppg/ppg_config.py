from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Literal


@dataclass
class PPGDaliaConfig:
    root: str                          # directory with per-subject CSVs
    split: Literal["train", "val", "test"] = "train"
    subjects_train: Tuple[str, ...] = tuple()  # subject IDs for train
    subject_val: Optional[str] = None          # single subject for val
    subject_test: Optional[str] = None         # single subject for test

    fs_in: float = 64.0                # sampling rate of raw files (typical E4=64 Hz)
    fs: float = 100.0                  # target sampling rate
    win_sec: int = 8                   # window length in seconds
    stride_sec: int = 2                # stride in seconds
    clip_min: Optional[float] = None   # optional amplitude clipping
    clip_max: Optional[float] = None
    # filters
    do_bandpass: bool = True
    low_hz: float = 0.5
    high_hz: float = 8.0

    # labels: use HR (beats/min) from file column
    ppg_col: str = "ppg"
    hr_col: str = "hr"                 # ground truth HR (from ECG), in bpm
    # optional additional columns (activity, etc.) are ignored

    # a tiny train-time augmentation (safe for PPG)
    aug_noise_std: float = 0.01        # Gaussian noise std after z-score (train only)
    aug_time_jitter_ms: int = 50       # random circular shift up to ±jitter (train only)

