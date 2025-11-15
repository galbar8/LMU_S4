from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, List
from pathlib import Path
import os, zipfile, requests, random
import pandas as pd
import torch
import torch.nn.functional as F
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
    # normalization: "none" | "instance" | "global_cmvn"
    normalize: Literal["none", "instance", "global_cmvn"] = "global_cmvn"
    target_num_frames: Optional[int] = 250
    augment: bool = True  # train_utils only
    # SpecAugment strength
    freq_mask_param: Optional[int] = None   # if None -> n_mels//5
    time_mask_param: Optional[int] = None   # if None -> T//10 (runtime)
    n_freq_masks: int = 2
    n_time_masks: int = 2
    # Auto download
    download: bool = False
    timeout_s: int = 60
    # Optional simple waveform augs (train_utils only)
    wav_time_shift_pct: float = 0.0  # e.g., 0.1 for ±10%
    wav_gain_db: float = 0.0         # e.g., 3.0 for ±3 dB
    # Precomputed CMVN (filled for val from train_utils stats)
    cmvn_mean: Optional[torch.Tensor] = None  # (n_mels,)
    cmvn_std: Optional[torch.Tensor] = None   # (n_mels,)

class ESC50Dataset(Dataset):
    """
    Returns:
      x: FloatTensor (T, D)  — waveform -> D=1, melspec -> D=n_mels
      y: LongTensor ()       — 0..49
      meta: dict             — {"filename": str, "fold": int, "category": str}
    """
    def __init__(self, cfg: ESC50Config):
        super().__init__()
        self.cfg = cfg
        data_dir = self._ensure_data(cfg.root, cfg.download, cfg.timeout_s)
        meta_csv = Path(data_dir) / "meta" / "esc50.csv"
        audio_dir = Path(data_dir) / "audio"
        if not (meta_csv.is_file() and audio_dir.is_dir()):
            raise FileNotFoundError(f"ESC-50 not found under {data_dir}")

        df = pd.read_csv(meta_csv)
        if cfg.fold_val not in {1,2,3,4,5}:
            raise ValueError("fold_val must be in {1,2,3,4,5}")
        if cfg.split == "train_utils":
            df = df[df["fold"] != cfg.fold_val].reset_index(drop=True)
        elif cfg.split == "val":
            df = df[df["fold"] == cfg.fold_val].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train_utils' or 'val'")

        self.audio_dir = audio_dir
        self.df = df
        self._items: List[Tuple[str,int,int,str]] = []
        for _, row in df.iterrows():
            self._items.append((str(audio_dir / row["filename"]),
                                int(row["target"]), int(row["fold"]),
                                str(row["category"])))

        # transforms
        self._resampler = Resample(orig_freq=44100, new_freq=cfg.sample_rate)
        self._melspec = None
        self._to_db = None
        if cfg.feature == "melspec":
            self._melspec = MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft, hop_length=cfg.hop_length,
                n_mels=cfg.n_mels, f_min=cfg.f_min,
                f_max=(cfg.f_max if cfg.f_max is not None else cfg.sample_rate/2),
                power=2.0, center=True, pad_mode="reflect",
                mel_scale="slaney", norm="slaney",
            )
            if cfg.to_db:
                self._to_db = AmplitudeToDB(stype="power", top_db=80.0)

        # SpecAugment modules (we’ll apply multiple times)
        self._freq_mask = FrequencyMasking(
            cfg.freq_mask_param if cfg.freq_mask_param is not None else max(1, cfg.n_mels // 5)
        ) if (cfg.augment and cfg.split == "train_utils" and cfg.feature == "melspec") else None

        # time mask param depends on T: handled in __getitem__

        # Precompute CMVN if requested and split==train_utils and none provided
        if cfg.feature == "melspec" and cfg.normalize == "global_cmvn":
            if cfg.split == "train_utils" and (cfg.cmvn_mean is None or cfg.cmvn_std is None):
                mean, std = self._compute_cmvn_stats()
                self.cmvn_mean = mean
                self.cmvn_std = std
            else:
                # val/test should receive stats from train_utils via cfg
                self.cmvn_mean = cfg.cmvn_mean
                self.cmvn_std = cfg.cmvn_std

    @staticmethod
    def _ensure_data(root: str, download: bool, timeout_s: int) -> str:
        root = os.path.expanduser(root)
        if download and not os.path.isabs(root):
            module_dir = Path(__file__).parent
            root = str(module_dir / root)

        official = os.path.join(root)
        if os.path.isfile(os.path.join(official, "meta", "esc50.csv")) and \
           os.path.isdir(os.path.join(official, "audio")):
            return official

        gh_dir = os.path.join(root, ESC50_ZIP_DIR)
        print(f"ESC-50 check under {official}. Checking {gh_dir} ...")
        if os.path.isfile(os.path.join(gh_dir, "meta", "esc50.csv")) and \
           os.path.isdir(os.path.join(gh_dir, "audio")):
            return gh_dir

        if not download:
            raise FileNotFoundError(
                "ESC-50 not found. Place 'audio/' and 'meta/esc50.csv' under root, "
                "or pass download=True."
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
        gh_dir = os.path.join(root, ESC50_ZIP_DIR)
        if not (os.path.isfile(os.path.join(gh_dir, "meta", "esc50.csv")) and
                os.path.isdir(os.path.join(gh_dir, "audio"))):
            raise RuntimeError("Downloaded ZIP missing expected files.")
        return gh_dir

    def __len__(self) -> int:
        return len(self._items)

    # --------- helpers ---------

    def _instance_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.normalize != "instance":
            return x
        if x.dim() == 1:
            m, s = x.mean(), x.std().clamp_min(1e-6)
            return (x - m) / s
        m = x.mean(dim=0)
        s = x.std(dim=0).clamp_min(1e-6)
        return (x - m) / s

    def _apply_cmvn(self, S: torch.Tensor) -> torch.Tensor:
        # S: (n_mels, T) BEFORE transpose
        if self.cfg.normalize != "global_cmvn":
            return S
        if self.cmvn_mean is None or self.cmvn_std is None:
            return S
        eps = 1e-6
        return (S - self.cmvn_mean[:, None]) / (self.cmvn_std[:, None] + eps)

    def _train_time_crop_or_center(self, x: torch.Tensor, T_out: int) -> torch.Tensor:
        # x: (T, D) or (T,)
        T = x.shape[0]
        if T == T_out:
            return x
        if T > T_out:
            if self.cfg.split == "train_utils":
                start = torch.randint(0, T - T_out + 1, (1,)).item()
            else:
                start = (T - T_out) // 2
            return x[start:start+T_out]
        # pad
        pad = T_out - T
        if self.cfg.split == "train_utils":
            left = torch.randint(0, pad + 1, (1,)).item()
        else:
            left = pad // 2
        right = pad - left
        if x.dim() == 1:
            return F.pad(x, (left, right))
        return F.pad(x, (0, 0, left, right))

    def _apply_specaugment(self, S: torch.Tensor) -> torch.Tensor:
        # S: (n_mels, T)
        if self._freq_mask is None or not self.cfg.augment:
            return S
        out = S
        for _ in range(self.cfg.n_freq_masks):
            out = self._freq_mask(out)
        # time mask param relative to T
        T = out.shape[-1]
        tparam = self.cfg.time_mask_param if self.cfg.time_mask_param is not None else max(1, T // 10)
        tmask = TimeMasking(time_mask_param=tparam)
        for _ in range(self.cfg.n_time_masks):
            out = tmask(out)
        return out

    def _compute_cmvn_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # one light pass on train_utils split
        sums = torch.zeros(self.cfg.n_mels, dtype=torch.float32)
        sqs  = torch.zeros(self.cfg.n_mels, dtype=torch.float32)
        count_frames = 0
        for path, _, _, _ in self._items:
            wav, sr = torchaudio.load(path, backend="soundfile")  # (C, N)
            if self.cfg.to_mono and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.cfg.sample_rate:
                wav = self._resampler(wav)
            S = self._melspec(wav)  # (n_mels, T)
            if self._to_db is not None:
                S = self._to_db(S)
            S = S.squeeze(0) if S.dim() > 2 else S
            sums += S.mean(dim=1)
            sqs  += (S**2).mean(dim=1)
            count_frames += 1
        mean = sums / count_frames
        var  = (sqs / count_frames) - (mean ** 2)
        std  = var.clamp_min(1e-6).sqrt()
        return mean, std

    def _wav_light_aug(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (1, N)
        if self.cfg.split != "train_utils":
            return wav
        # time shift
        if self.cfg.wav_time_shift_pct > 0:
            N = wav.shape[-1]
            shift = int((random.uniform(-self.cfg.wav_time_shift_pct, self.cfg.wav_time_shift_pct)) * N)
            if shift != 0:
                wav = torch.roll(wav, shifts=shift, dims=-1)
        # gain
        if self.cfg.wav_gain_db > 0:
            gain = random.uniform(-self.cfg.wav_gain_db, self.cfg.wav_gain_db)
            wav = wav * (10.0 ** (gain / 20.0))
        return wav

    # --------- main ---------

    def __getitem__(self, idx: int):
        path, y, fold, category = self._items[idx]
        wav, sr = torchaudio.load(path, backend="soundfile")  # (C, N)
        if self.cfg.to_mono and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.cfg.sample_rate:
            wav = self._resampler(wav)

        wav = wav.contiguous()

        if self.cfg.feature == "waveform":
            wav = self._wav_light_aug(wav)  # optional
            x = wav.squeeze(0).to(torch.float32)  # (T,)
            if self.cfg.target_num_frames is not None:
                x = self._train_time_crop_or_center(x, self.cfg.target_num_frames *  self.cfg.hop_length)
            x = self._instance_normalize(x)
            x = x.unsqueeze(-1)  # (T,1)
        else:
            wav = self._wav_light_aug(wav)
            S = self._melspec(wav)  # (n_mels, T)
            if self._to_db is not None:
                S = self._to_db(S)
            S = S.squeeze(0) if S.dim() > 2 else S  # (n_mels, T)
            # SpecAugment (train_utils only)
            if self.cfg.split == "train_utils" and self.cfg.augment:
                S = self._apply_specaugment(S)
            # Global CMVN (before transpose)
            S = self._apply_cmvn(S)
            x = S.transpose(0, 1).contiguous().to(torch.float32)  # (T, n_mels)
            # pad/crop to fixed T
            if self.cfg.target_num_frames is not None:
                x = self._train_time_crop_or_center(x, self.cfg.target_num_frames)
            # if instance norm requested (rare when CMVN used)
            x = self._instance_normalize(x)

        meta = {"filename": Path(path).name, "fold": fold, "category": category}
        return x, torch.tensor(y, dtype=torch.long), meta


# ----------------- Loader utility -----------------

def seed_everything(seed: int):
    import numpy as np, random as pyrand
    pyrand.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id: int):
    # derive worker-specific RNG seeds for reproducible augs
    import numpy as np, random as pyrand, os
    base_seed = int(os.environ.get("PYTHONHASHSEED", "0")) or 0
    s = base_seed + worker_id
    np.random.seed(s); pyrand.seed(s)

def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def make_esc50_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    fold_val: int = 1,
    feature: Literal["melspec","waveform"] = "melspec",
    sample_rate: int = 16000, n_fft: int = 1024, hop_length: int = 320,
    n_mels: int = 128, f_min: float = 20.0, f_max: Optional[float] = None,
    to_db: bool = True,
    normalize: Literal["none","instance","global_cmvn"] = "global_cmvn",
    target_num_frames: Optional[int] = 250,
    augment: bool = True,
    download: bool = False,
    seed: int = 42,
    wav_time_shift_pct: float = 0.0,
    wav_gain_db: float = 0.0,
):
    """
    Returns: (train_loader, val_loader, cmvn_stats) with:
      - x: (B, T, D), y: (B,), meta: dict
      - cmvn_stats: dict with 'mean' and 'std' (for logging)
    """
    seed_everything(seed)

    # Only use pin_memory on CUDA devices
    pin_memory = torch.cuda.is_available()

    common = dict(
        root=data_root, fold_val=fold_val, feature=feature,
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, f_min=f_min, f_max=f_max, to_db=to_db,
        normalize=normalize, target_num_frames=target_num_frames,
        download=download, augment=augment,
        wav_time_shift_pct=wav_time_shift_pct, wav_gain_db=wav_gain_db,
    )

    # First create a train_utils dataset to compute CMVN if needed
    train_ds = ESC50Dataset(ESC50Config(**common, split="train_utils"))
    cmvn_stats = None
    if feature == "melspec" and normalize == "global_cmvn":
        # pass stats to val
        cmvn_stats = {
            "mean": train_ds.cmvn_mean.detach().clone(),
            "std":  train_ds.cmvn_std.detach().clone(),
        }
        val_ds = ESC50Dataset(ESC50Config(**common, split="val",
                             cmvn_mean=cmvn_stats["mean"], cmvn_std=cmvn_stats["std"]))
    else:
        val_ds = ESC50Dataset(ESC50Config(**common, split="val"))

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, generator=g,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, cmvn_stats
