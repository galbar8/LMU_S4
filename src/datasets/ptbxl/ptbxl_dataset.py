from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import wfdb

# 5 diagnostic superclasses (כמו במאמר/רפרנסים הרשמיים)
PTBXL_SUPERCLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]


@dataclass
class PTBXLConfig:
    root: str  # נתיב לתיקייה עם ה-CSV והתיקיות records100/records500
    split: Literal["train_utils", "val", "test"] = "train_utils"
    folds_train: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    fold_val: int = 9
    fold_test: int = 10
    sampling: Literal["lr100", "hr500"] = "lr100"  # 100Hz או 500Hz
    leads: Optional[Iterable[int]] = None  # ברירת מחדל: כל 12 הלידים
    length: Optional[int] = None  # אורך יעד (T); אם None – לא חותך/מרפד
    normalize: Literal["per_lead_z", "none"] = "per_lead_z"
    return_path: bool = False  # להחזיר גם את path יחסי (ל-debug)


def _parse_scp_codes(s: str) -> Dict[str, float]:
    # עמודת scp_codes ב-CSV היא dict כמחרוזת
    return ast.literal_eval(s)


def _labels_superclasses(row, scp_df: pd.DataFrame) -> np.ndarray:
    """
    מחזיר וקטור multi-label בגודל 5 ל-superclasses.
    scp_df (scp_statements.csv) מגדיר אילו קודים diagnostic ולאיזה superclass הם שייכים.
    """
    codes = _parse_scp_codes(row["scp_codes"])
    classes = set()
    for code in codes.keys():
        if code in scp_df.index and scp_df.loc[code, "diagnostic"] == 1:
            cls = scp_df.loc[code, "diagnostic_class"]
            if isinstance(cls, str):
                classes.add(cls)
    y = np.zeros(len(PTBXL_SUPERCLASSES), dtype=np.float32)
    for i, name in enumerate(PTBXL_SUPERCLASSES):
        if name in classes:
            y[i] = 1.0
    return y


class PTBXLDataset(Dataset):
    """
    מחזיר:
      x: torch.FloatTensor בצורה (T, D)  עם D=מספר לידים (ברירת מחדל 12)
      y: torch.FloatTensor בצורה (5,)    multi-label (NORM, MI, STTC, HYP, CD)

    הערות:
    - קריאת אותות ב-WFDB (קבצי .dat/.hea) דרך wfdb.
    - שימוש ב-filenames יחסי מתוך ה-CSV: filename_lr (100Hz) או filename_hr (500Hz).
    - נורמליזציה z-score לכל ליד בנפרד.
    - length: אם הוגדר, חותך/מרפד באפסים לאורך קבוע (נוח ל-batching).
    """

    def __init__(self, cfg: PTBXLConfig):
        super().__init__()
        self.cfg = cfg
        self.root = Path(cfg.root)
        db_csv = self.root / "ptbxl_database.csv"
        scp_csv = self.root / "scp_statements.csv"
        if not db_csv.exists() or not scp_csv.exists():
            raise FileNotFoundError(
                f"Missing CSV files. Expected {db_csv} and {scp_csv}. "
                f"See PhysioNet: PTB-XL download & extraction."
            )

        df = pd.read_csv(db_csv)
        scp_df = pd.read_csv(scp_csv, index_col=0)

        # פיצול לפי strat_fold (סטנדרטי ב-PTB-XL)
        if cfg.split == "train_utils":
            df = df[df["strat_fold"].isin(cfg.folds_train)]
        elif cfg.split == "val":
            df = df[df["strat_fold"] == cfg.fold_val]
        elif cfg.split == "test":
            df = df[df["strat_fold"] == cfg.fold_test]
        else:
            raise ValueError("split must be one of: train_utils/val/test")

        # בחירת עמודת הנתיב לפי קצב הדגימה
        sig_col = "filename_lr" if cfg.sampling == "lr100" else "filename_hr"

        # בונים רשימת פריטים: (path_rel, onehot_superclasses)
        self.items: List[Tuple[str, np.ndarray]] = []
        for _, row in df.iterrows():
            y = _labels_superclasses(row, scp_df)
            path_rel = row[sig_col]
            self.items.append((path_rel, y))

        # בחירת לידים (ברירת מחדל: כל 12)
        self.leads = list(cfg.leads) if cfg.leads is not None else list(range(12))

        # מאפיינים אוטומטיים
        self.D = len(self.leads)

    def __len__(self) -> int:
        return len(self.items)

    def _read_signal(self, path_rel: str) -> np.ndarray:
        """
        קורא קובץ WFDB לפי נתיב יחסי (כמו ב-CSV). מחזיר np.ndarray בצורת (T, 12).
        """
        rec_base = (self.root / path_rel).with_suffix("")  # wfdb.rdsamp מצפה לבסיס בלי סיומת
        sig, _ = wfdb.rdsamp(str(rec_base))  # (T, 12)
        sig = sig.astype(np.float32)
        # בחירת לידים
        sig = sig[:, self.leads]  # (T, D)
        # נורמליזציה
        if self.cfg.normalize == "per_lead_z":
            mean = sig.mean(axis=0, keepdims=True)
            std = sig.std(axis=0, keepdims=True) + 1e-6
            sig = (sig - mean) / std
        return sig

    def _fix_length(self, x: np.ndarray) -> np.ndarray:
        """
        חותך/מרפד x לאורך קבוע self.cfg.length אם נדרש.
        """
        if self.cfg.length is None:
            return x
        T, D = x.shape
        L = self.cfg.length
        if T == L:
            return x
        if T > L:
            return x[:L]
        # pad
        out = np.zeros((L, D), dtype=x.dtype)
        out[:T] = x
        return out

    def __getitem__(self, idx: int):
        path_rel, y = self.items[idx]
        x = self._read_signal(path_rel)  # (T, D)
        x = self._fix_length(x)  # (L or T, D)
        x_t = torch.from_numpy(x)  # float32
        y_t = torch.from_numpy(y)  # float32 multi-label (5,)
        if self.cfg.return_path:
            return x_t, y_t, path_rel
        return x_t, y_t
