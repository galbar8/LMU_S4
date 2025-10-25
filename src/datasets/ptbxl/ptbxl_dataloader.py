from typing import Optional, Tuple, Literal, Iterable

import torch
from torch.utils.data import DataLoader

from src.datasets.ptbxl.ptbxl_dataset import PTBXLConfig, PTBXLDataset


def make_ptbxl_loaders(
        data_root: str,
        batch_size: int = 64,
        num_workers: int = 4,
        sampling: Literal["lr100", "hr500"] = "lr100",
        length: Optional[int] = 1000,
        leads: Optional[Iterable[int]] = None,
        folds_train: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
        fold_val: int = 9,
        fold_test: int = 10,
        pin_memory: bool = True,
        persistent_workers: bool = False,
):
    cfg_train = PTBXLConfig(
        root=data_root, split="train_utils", sampling=sampling, length=length, leads=leads,
        folds_train=folds_train, fold_val=fold_val, fold_test=fold_test
    )
    cfg_val = PTBXLConfig(
        root=data_root, split="val", sampling=sampling, length=length, leads=leads,
        folds_train=folds_train, fold_val=fold_val, fold_test=fold_test
    )
    cfg_test = PTBXLConfig(
        root=data_root, split="test", sampling=sampling, length=length, leads=leads,
        folds_train=folds_train, fold_val=fold_val, fold_test=fold_test
    )

    train_ds = PTBXLDataset(cfg_train)
    val_ds = PTBXLDataset(cfg_val)
    test_ds = PTBXLDataset(cfg_test)

    g = torch.Generator().manual_seed(0)

    # Configure persistent_workers (only works if num_workers > 0)
    use_persistent = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, generator=g,
        persistent_workers=use_persistent
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=use_persistent
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=use_persistent
    )
    return train_loader, val_loader, test_loader
