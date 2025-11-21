"""Signal processing utilities for PPG data."""
from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    """Apply Butterworth bandpass filter with safe guards for short signals.
    If the signal is too short for filtfilt padding, return it unfiltered.
    """
    nyq = 0.5 * fs
    low_norm = low / nyq
    high_norm = high / nyq

    # Clamp to valid range
    low_norm = max(0.001, min(low_norm, 0.999))
    high_norm = max(low_norm + 0.001, min(high_norm, 0.999))

    b, a = signal.butter(order, [low_norm, high_norm], btype='band')

    # Compute padlen consistent with scipy's filtfilt default and check strictly > padlen
    padlen = 3 * (max(len(b), len(a)) - 1)
    if x.shape[0] <= padlen:
        # Too short for filtfilt padding: return unfiltered
        return x.astype(np.float32)

    # Try filtering; if scipy still raises (edge case), fall back to unfiltered
    try:
        return signal.filtfilt(b, a, x).astype(np.float32)
    except ValueError:
        return x.astype(np.float32)


def to_100hz(x: np.ndarray, fs_in: float, fs_out: float = 100.0) -> np.ndarray:
    """Resample signal to target sampling rate (default 100Hz)."""
    if abs(fs_in - fs_out) < 0.1:
        return x
    num_samples = int(len(x) * fs_out / fs_in)
    return signal.resample(x, num_samples).astype(np.float32)


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalization."""
    mu = np.mean(x)
    std = np.std(x)
    return ((x - mu) / (std + eps)).astype(np.float32)
