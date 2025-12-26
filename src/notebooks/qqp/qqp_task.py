from __future__ import annotations

from typing import Tuple, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from src.types.task_protocol import TaskProtocol
from src.datasets.qqp.qqp_dataloader import make_qqp_loaders
from src.datasets.qqp.qqp_config import QQPConfig
from src.datasets.qqp.qqp_dataset import QQPDataset, build_vocab_from_texts


class QQPTask(TaskProtocol):
    """
    Quora Question Pairs (QQP) binary classification task.

    - Input: a pair of questions (question1, question2)
    - Output: is_duplicate ∈ {0,1}
    - Form: token-ID sequences with fixed length per question (max_len)

    Dataset output (per sample):
        x: LongTensor (max_len * 2 + 1,)   # concatenated token ids [q1, <sep>, q2]
        y: LongTensor scalar               # 0/1 label

    Notes:
    - Questions are concatenated into a single sequence [q1, <sep>, q2]
    - Vocabulary is built from TRAIN only and shared across val/test.
    - Splits are deterministic by seed.
    - This task typically uses an nn.Embedding in the model (token IDs → vectors).
    """
    problem_type: str = "multiclass"

    def make_loaders(
        self,
        data_root: str,
        batch_size: int = 64,
        **kwargs,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Returns: (train_loader, val_loader, test_loader)
        """
        # Split / reproducibility
        seed = kwargs.get("seed", 42)
        val_ratio = kwargs.get("val_ratio", 0.1)
        test_ratio = kwargs.get("test_ratio", 0.1)

        # Tokenization / sequence settings
        lowercase = kwargs.get("lowercase", True)
        max_len = kwargs.get("max_len", 64)
        max_vocab = kwargs.get("max_vocab", 50_000)
        min_freq = kwargs.get("min_freq", 2)

        # DataLoader settings
        pin_memory = kwargs.get("pin_memory", False)
        persistent_workers = kwargs.get("persistent_workers", False)
        num_workers = kwargs.get("num_workers", 0)

        subset_size = kwargs.get("subset_size", None)
        csv_filename = kwargs.get("csv_filename", "questions.csv")

        train_loader, val_loader, test_loader = make_qqp_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            lowercase=lowercase,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
            subset_size=subset_size,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            cache_splits=kwargs.get("cache_splits", True),
            csv_filename=csv_filename,
        )

        return train_loader, val_loader, test_loader

    def infer_input_dim(self, args: Dict[str, Any]) -> int:
        """
        For QQP, the dataset returns token IDs, so the model should embed them.

        In most setups, `input_dim` for the backbone equals the embedding dimension.
        We infer it from args to keep the task generic.

        Expected keys (first match wins):
            - d_model
            - embed_dim
            - embedding_dim
        """
        for k in ("d_model", "embed_dim", "embedding_dim"):
            if k in args and args[k] is not None:
                return int(args[k])
        raise ValueError(
            "QQPTask.infer_input_dim: QQP uses token IDs and requires an embedding dimension. "
            "Please provide args['d_model'] (or 'embed_dim' / 'embedding_dim')."
        )

    @staticmethod
    def get_vocab_size(data_root: str, **kwargs) -> int:
        """
        Get the vocabulary size for the QQP dataset.
        This is needed for the embedding layer.
        """
        csv_filename = kwargs.get("csv_filename", "questions.csv")
        seed = kwargs.get("seed", 42)
        val_ratio = kwargs.get("val_ratio", 0.1)
        test_ratio = kwargs.get("test_ratio", 0.1)
        lowercase = kwargs.get("lowercase", True)
        max_len = kwargs.get("max_len", 64)
        max_vocab = kwargs.get("max_vocab", 50_000)
        min_freq = kwargs.get("min_freq", 2)

        # Load CSV and create train split
        csv_path = Path(data_root) / csv_filename
        df = pd.read_csv(csv_path)
        n = len(df)

        # Same split logic as make_qqp_loaders
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)

        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        n_train = n - n_val - n_test
        train_idx = idx[:n_train]

        # Build vocab from train split
        train_df = df.iloc[train_idx]
        train_texts = (
            train_df["question1"].astype("object").tolist()
            + train_df["question2"].astype("object").tolist()
        )

        vocab = build_vocab_from_texts(
            texts=train_texts,
            lowercase=lowercase,
            pad_token="<pad>",
            unk_token="<unk>",
            max_vocab=max_vocab,
            min_freq=min_freq,
        )

        # Create minimal config and dataset instance to get vocab_size
        temp_cfg = QQPConfig(
            root=data_root,
            csv_filename=csv_filename,
            split="train",
            seed=seed,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            subset_size=1,  # Minimal for efficiency
            lowercase=lowercase,
            max_len=max_len,
            max_vocab=max_vocab,
            min_freq=min_freq,
            cache_splits=kwargs.get("cache_splits", True),
        )

        temp_dataset = QQPDataset(temp_cfg, vocab=vocab, indices=train_idx[:1])
        return temp_dataset.vocab_size

    def infer_num_classes(self, args: Dict[str, Any]) -> int:
        """Binary classification: {not-duplicate, duplicate}."""
        return 2

    def infer_theta(self, args: Dict[str, Any]) -> int:
        """
        Sequence length for QQP.
        We concatenate two questions, each of length max_len, with a separator token.
        Total sequence length = max_len * 2 + 1 (for <sep> token)
        """
        max_len = int(args.get("max_len", 64))
        return max_len * 2 + 1
