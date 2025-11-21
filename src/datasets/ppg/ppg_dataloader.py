from typing import Tuple, Optional
from torch.utils.data import DataLoader
import torch

from src.datasets.ppg.ppg_config import PPGDaliaConfig
from src.datasets.ppg.ppg_dataset import PPGDaliaDataset


def ppg_collate(batch):
    xs, ys, metas = zip(*batch)
    xs = torch.stack(xs, 0)  # [B, L, 1]
    ys = torch.stack(ys, 0)  # [B] - scalar targets for regression

    return xs, ys, metas

def make_ppgdalia_loaders(
    cfg: PPGDaliaConfig,
    batch: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    def mk(ds):
        if ds is None: return None
        return DataLoader(
            ds, batch_size=batch, shuffle=(ds.cfg.split=="train"),
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=persistent_workers, collate_fn=ppg_collate, drop_last=False
        )
    ds_train = PPGDaliaDataset(PPGDaliaConfig(**{**cfg.__dict__, "split": "train"})) if cfg.subjects_train else None
    ds_val   = PPGDaliaDataset(PPGDaliaConfig(**{**cfg.__dict__, "split": "val"}))   if cfg.subject_val   else None
    ds_test  = PPGDaliaDataset(PPGDaliaConfig(**{**cfg.__dict__, "split": "test"}))  if cfg.subject_test  else None
    return mk(ds_train), mk(ds_val), mk(ds_test)
