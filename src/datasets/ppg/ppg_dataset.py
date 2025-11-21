from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .ppg_config import PPGDaliaConfig
from .ppg_filters import bandpass, to_100hz, zscore


def _list_subject_files(root: Path) -> Dict[str, Path]:
    """
    Expecting structure:
        root/
          S01/S01.pkl
          S02/S02.pkl
          ...
    Each pickle must contain 'signal' dict with BVP data and 'label' array with HR.
    """
    files_by_subj: Dict[str, Path] = {}
    for subj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        subj = subj_dir.name
        pkl_file = subj_dir / f"{subj}.pkl"
        if pkl_file.exists():
            files_by_subj[subj] = pkl_file
    return files_by_subj


def _load_and_preprocess_file(path: Path, cfg: PPGDaliaConfig) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Load DALIA pickle file and return (ppg_signal, hr_labels).
    PPG is resampled, filtered, and normalized.
    HR labels are kept as-is (one per window).
    """
    import pickle

    try:
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception:
        return None

    # Extract BVP signal (64 Hz from DALIA)
    if 'signal' not in data or 'wrist' not in data['signal'] or 'BVP' not in data['signal']['wrist']:
        return None

    ppg = np.asarray(data['signal']['wrist']['BVP'], dtype=np.float32)

    # Extract HR labels
    if 'label' not in data:
        return None

    hr_labels = np.asarray(data['label'], dtype=np.float32)

    # Resample PPG to target frequency
    ppg = to_100hz(ppg, cfg.fs_in, cfg.fs)

    # Filter PPG
    if cfg.do_bandpass:
        ppg = bandpass(ppg, fs=cfg.fs, low=cfg.low_hz, high=cfg.high_hz, order=3)

    # Optional clipping
    if cfg.clip_min is not None or cfg.clip_max is not None:
        ppg = np.clip(ppg,
                     cfg.clip_min if cfg.clip_min is not None else -np.inf,
                     cfg.clip_max if cfg.clip_max is not None else np.inf)

    # Z-score normalize
    ppg = zscore(ppg)

    return ppg.astype(np.float32), hr_labels.astype(np.float32)


def _windowize(ppg: np.ndarray, hr: np.ndarray, cfg: PPGDaliaConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create fixed windows (X) and scalar HR targets (y).
    Target is the mean HR in the window (bpm).
    """
    L = int(cfg.win_sec * cfg.fs)
    S = int(cfg.stride_sec * cfg.fs)
    n = len(ppg)
    xs, ys = [], []
    for start in range(0, n - L + 1, S):
        end = start + L
        x = ppg[start:end]
        y = np.nanmean(hr[start:end])  # robust to a few NaNs
        if np.isnan(y):
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.empty((0, L), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X = np.stack(xs, axis=0)  # [N, L]
    y = np.asarray(ys, dtype=np.float32)  # [N]
    return X, y


def _windowize_with_labels(ppg: np.ndarray, hr_labels: np.ndarray, cfg: PPGDaliaConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create fixed windows from PPG signal using pre-computed HR labels.
    DALIA has pre-computed HR labels (one per window).
    Windows are 8s with 2s stride by default.
    """
    L = int(cfg.win_sec * cfg.fs)
    S = int(cfg.stride_sec * cfg.fs)

    # Number of windows matches number of labels
    n_windows = len(hr_labels)

    xs, ys = [], []
    for i in range(n_windows):
        start = i * S
        end = start + L

        if end > len(ppg):
            break

        x = ppg[start:end]
        y = hr_labels[i]

        if np.isnan(y) or np.isnan(x).any():
            continue

        xs.append(x)
        ys.append(y)

    if not xs:
        return np.empty((0, L), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.stack(xs, axis=0)  # [N, L]
    y = np.asarray(ys, dtype=np.float32)  # [N]
    return X, y


class PPGDaliaDataset(Dataset):
    """
    Returns:
        x: [L, 1] float32 (PPG window, z-scored, band-passed)
        y: scalar float32 (mean HR over the window, bpm)
        meta: dict(subject, file, idx)
    """

    def __init__(self, cfg: PPGDaliaConfig):
        self.cfg = cfg
        root = Path(cfg.root)
        files_by_subj = _list_subject_files(root)
        if cfg.split == "train":
            subjects = list(cfg.subjects_train)
        elif cfg.split == "val":
            assert cfg.subject_val is not None, "subject_val must be set for val split"
            subjects = [cfg.subject_val]
        else:
            assert cfg.subject_test is not None, "subject_test must be set for test split"
            subjects = [cfg.subject_test]
        # Gather windows
        Xs, Ys, metas = [], [], []
        for subj in subjects:
            pkl_file = files_by_subj.get(subj)
            if pkl_file is None:
                continue

            result = _load_and_preprocess_file(pkl_file, cfg)
            if result is None:
                continue

            ppg, hr_labels = result
            X, y = _windowize_with_labels(ppg, hr_labels, cfg)

            for i in range(len(y)):
                Xs.append(X[i])
                Ys.append(y[i])
                metas.append({"subject": subj, "idx": i})

        if len(Ys) == 0:
            raise RuntimeError(f"No samples found for split={cfg.split}. Check subjects: {subjects}")
        self.X = np.stack(Xs, axis=0).astype(np.float32)  # [N, L]
        self.y = np.asarray(Ys, dtype=np.float32)  # [N]
        self.metas = metas
        # Small train-time augmentation
        self.do_aug = cfg.split == "train" and (cfg.aug_noise_std > 0 or cfg.aug_time_jitter_ms > 0)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        x = self.X[i].copy()  # [L]
        y = self.y[i].item()  # Scalar

        if self.do_aug:
            # time jitter: circular shift up to ±aug_time_jitter_ms
            max_shift = int((self.cfg.aug_time_jitter_ms / 1000.0) * self.cfg.fs)
            if max_shift > 0:
                s = np.random.randint(-max_shift, max_shift + 1)
                if s != 0:
                    x = np.roll(x, s)
            if self.cfg.aug_noise_std > 0:
                x = x + np.random.normal(0.0, self.cfg.aug_noise_std, size=x.shape).astype(np.float32)

        # x is [L], we need [L, 1] for the model
        # BUT: torch.from_numpy creates the right shape
        x = torch.from_numpy(x.astype(np.float32))  # [L]
        x = x.unsqueeze(-1)  # [L, 1]

        y = torch.tensor(y, dtype=torch.float32)  # scalar tensor

        return x, y, self.metas[i]
