from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, Tuple, Optional

@dataclass
class PTBXLConfig:
    root: str
    split: Literal["train_utils", "val", "test"] = "train_utils"
    folds_train: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    fold_val: int = 9
    fold_test: int = 10
    sampling: Literal["lr100", "hr500"] = "lr100"
    leads: Optional[Iterable[int]] = None
    length: Optional[int] = None
    normalize: Literal["per_lead_z", "none"] = "per_lead_z"
    return_path: bool = False