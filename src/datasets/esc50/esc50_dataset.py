from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, List, Set
from pathlib import Path
import os
import zipfile
import requests
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import (
    Resample, MelSpectrogram, AmplitudeToDB,
    FrequencyMasking, TimeMasking
)

ESC50_GITHUB_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
ESC50_ZIP_NAME   = "ESC-50-master.zip"
ESC50_ZIP_DIR    = "ESC-50-master"

@dataclass
class ESC50Config:
    root: str
    split: Literal["train_utils", "val"] = "train_utils"
    fold_val: int = 1                     # 1..5
    # Audio → Features
    feature: Literal["melspec", "waveform"] = "melspec"
    sample_rate: int = 16000
    to_mono: bool = True
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 128
    f_min: float = 20.0
    f_max: Optional[float] = None         # If None → sr/2
    to_db: bool = True                    # AmplitudeToDB
    # Processing
    normalize: Literal["none", "instance"] = "instance"
    target_num_frames: Optional[int] = 250   # Set fixed length for frames (e.g. ~5 seconds @16k, hop=320 ≈ ~250)
    augment: bool = False                 # SpecAugment (train_utils only)
    freq_mask_param: int = 12
    time_mask_param: int = 24
    # Automatic ZIP download from GitHub (optional)
    download: bool = False
    timeout_s: int = 60

class ESC50Dataset(Dataset):
    """
    Returns:
      x: FloatTensor (T, D)  —  if waveform → D=1; if melspec → D=n_mels
      y: LongTensor ()        —  target [0..49]

    Input ready directly for sequential models (S4/LMU) via DataLoader: (B, T, D)
    """
    def __init__(self, cfg: ESC50Config):
        super().__init__()
        self.cfg = cfg
        data_dir = self._ensure_data(cfg.root, cfg.download, cfg.timeout_s)
        meta_csv = Path(data_dir) / "meta" / "esc50.csv"
        audio_dir = Path(data_dir) / "audio"
        if not (meta_csv.is_file() and audio_dir.is_dir()):
            raise FileNotFoundError(f"ESC-50 not found under {data_dir} "
                                    "(expected meta/esc50.csv and audio/*.wav)")

        df = pd.read_csv(meta_csv)
        # Split by fold
        if cfg.fold_val not in {1,2,3,4,5}:
            raise ValueError("fold_val must be in {1,2,3,4,5}")
        if cfg.split == "train_utils":
            df = df[df["fold"] != cfg.fold_val].reset_index(drop=True)
        elif cfg.split == "val":
            df = df[df["fold"] == cfg.fold_val].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train_utils' or 'val'")

        self._items: List[Tuple[str, int, int]] = []
        for _, row in df.iterrows():
            wav_path = audio_dir / row["filename"]
            self._items.append((str(wav_path), int(row["target"]), int(row["fold"])))

        # Transforms
        self._resampler = Resample(orig_freq=44100, new_freq=cfg.sample_rate)
        self._melspec = None
        if cfg.feature == "melspec":
            self._melspec = MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                n_mels=cfg.n_mels, f_min=cfg.f_min,
                f_max=(cfg.f_max if cfg.f_max is not None else cfg.sample_rate/2),
                power=2.0, center=True, pad_mode="reflect", mel_scale="slaney", norm="slaney"
            )
            self._to_db = AmplitudeToDB(stype="power", top_db=80.0) if cfg.to_db else None
            # SpecAugment (train_utils only)
            self._freq_mask = FrequencyMasking(cfg.freq_mask_param) if cfg.augment and cfg.split=="train_utils" else None
            self._time_mask = TimeMasking(cfg.time_mask_param)       if cfg.augment and cfg.split=="train_utils" else None

    @staticmethod
    def _ensure_data(root: str, download: bool, timeout_s: int) -> str:
        """
        Locates official structure (audio/ + meta/esc50.csv) or the downloaded ZIP folder.
        If not found — and if download=True — downloads the official ZIP from GitHub.
        """
        root = os.path.expanduser(root)

        # If download=True and root is a relative path, use a path relative to this dataset module
        if download and not os.path.isabs(root):
            # Get the directory where this dataset module is located
            module_dir = Path(__file__).parent
            root = str(module_dir / root)

        official = os.path.join(root)
        print(f"Downloading ESC-50 data from {official} to {root}")

        if os.path.isfile(os.path.join(official, "meta", "esc50.csv")) and \
           os.path.isdir(os.path.join(official, "audio")):
            return official

        gh_dir = os.path.join(root, ESC50_ZIP_DIR)
        if os.path.isfile(os.path.join(gh_dir, "meta", "esc50.csv")) and \
           os.path.isdir(os.path.join(gh_dir, "audio")):
            return gh_dir

        if not download:
            raise FileNotFoundError(
                "ESC-50 not found. Place 'audio/' and 'meta/esc50.csv' under root, "
                "or pass download=True to fetch the official GitHub ZIP."
            )

        Path(root).mkdir(parents=True, exist_ok=True)
        zip_path = Path(root) / ESC50_ZIP_NAME
        if not zip_path.exists():
            r = requests.get(ESC50_GITHUB_URL, stream=True, timeout=timeout_s)
            r.raise_for_status()
            tmp = zip_path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1<<20):
                    if chunk:
                        f.write(chunk)
            tmp.rename(zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(path=root)

        # Check again
        gh_dir = os.path.join(root, ESC50_ZIP_DIR)
        if not (os.path.isfile(os.path.join(gh_dir, "meta", "esc50.csv")) and
                os.path.isdir(os.path.join(gh_dir, "audio"))):
            raise RuntimeError("Downloaded ZIP does not contain expected files.")
        return gh_dir

    def __len__(self) -> int:
        return len(self._items)

    def _instance_normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D) or (T,)
        if self.cfg.normalize != "instance":
            return x
        if x.dim() == 1:
            m, s = x.mean(), x.std().clamp_min(1e-6)
            return (x - m) / s
        m = x.mean(dim=0)                     # per-feature mean over time
        s = x.std(dim=0).clamp_min(1e-6)
        return (x - m) / s

    def _pad_or_crop(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, D) or (T,)
        L = self.cfg.target_num_frames
        if L is None:
            return x
        T = x.shape[0]
        if T == L:
            return x
        if T > L:
            return x[:L]
        pad = L - T
        if x.dim() == 1:
            return torch.nn.functional.pad(x, (0, pad))
        return torch.nn.functional.pad(x, (0, 0, 0, pad))  # pad time axis

    def __getitem__(self, idx: int):
        path, y, fold = self._items[idx]
        # Stable reading (soundfile backend)
        wav, sr = torchaudio.load(path, backend="soundfile")  # (C, N)
        if self.cfg.to_mono and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to target SR (default ESC-50 is 44.1k)
        if sr != self.cfg.sample_rate:
            wav = self._resampler(wav)

        if self.cfg.feature == "waveform":
            x = wav.squeeze(0)  # (T,)
            x = self._pad_or_crop(x)
            x = self._instance_normalize(x)
            x = x.unsqueeze(-1)  # (T, 1) — to maintain (T, D) shape
        else:
            # MelSpectrogram -> (n_mels, T)
            S = self._melspec(wav)  # (n_mels, T)
            if self.cfg.to_db and self._to_db is not None:
                S = self._to_db(S)

            # SpecAugment (train_utils only)
            if self.cfg.split == "train_utils":
                if hasattr(self, "_freq_mask") and self._freq_mask is not None:
                    S = self._freq_mask(S)
                if hasattr(self, "_time_mask") and self._time_mask is not None:
                    S = self._time_mask(S)

            # Ensure S is 2D (n_mels, T) - squeeze any extra dimensions
            while S.dim() > 2:
                S = S.squeeze(0)

            # Transpose to (T, n_mels) format expected by the model
            x = S.transpose(0, 1).contiguous()  # (T, n_mels)
            x = self._pad_or_crop(x)
            x = self._instance_normalize(x)

        return x, torch.tensor(y, dtype=torch.long)

def make_esc50_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    fold_val: int = 1,
    feature: Literal["melspec","waveform"] = "melspec",
    # mel params
    sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 320,
    n_mels: int = 128, f_min: float = 20.0, f_max: Optional[float] = None,
    to_db: bool = True,
    # processing
    normalize: Literal["none","instance"] = "instance",
    target_num_frames: Optional[int] = 250,
    augment: bool = True,
    download: bool = False,
):
    """
    Returns: (train_loader, val_loader) with x: (B, T, D), y: (B,)
    Ready for S4/LMU without custom collate function (since T is fixed).
    """
    common = dict(
        root=data_root, fold_val=fold_val, feature=feature,
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, f_min=f_min, f_max=f_max, to_db=to_db,
        normalize=normalize, target_num_frames=target_num_frames,
        download=download,
    )
    train_ds = ESC50Dataset(ESC50Config(**common, split="train_utils", augment=augment))
    val_ds   = ESC50Dataset(ESC50Config(**common, split="val",   augment=False))

    g = torch.Generator().manual_seed(0)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=True, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader