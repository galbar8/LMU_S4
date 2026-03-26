from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal
import torch

@dataclass
class ESC50Config:
    root: str
    split: Literal["train_utils", "val"] = "train_utils"
    fold_val: int = 1
    feature: Literal["melspec", "waveform"] = "melspec"
    sample_rate: int = 16000
    to_mono: bool = True
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 128
    f_min: float = 20.0
    f_max: Optional[float] = None
    to_db: bool = True
    normalize: Literal["none", "instance", "global_cmvn"] = "global_cmvn"
    target_num_frames: Optional[int] = 250
    augment: bool = True  # train_utils only
    # SpecAugment strength
    freq_mask_param: Optional[int] = None
    time_mask_param: Optional[int] = None
    n_freq_masks: int = 2
    n_time_masks: int = 2
    # Auto download
    download: bool = False
    timeout_s: int = 60
    wav_time_shift_pct: float = 0.0
    wav_gain_db: float = 0.0
    cmvn_mean: Optional[torch.Tensor] = None
    cmvn_std: Optional[torch.Tensor] = None
