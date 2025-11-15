"""
Trainer module.

This module defines a ``Trainer`` class that encapsulates the
common training and evaluation loops for various tasks.  It
automatically constructs the model, optimizer and scheduler, and
handles early stopping based on a specified metric.  The class
is designed to be task‑agnostic: any dataset that provides a
``make_loaders`` function and associated metadata can be used.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from src.train_utils.loops import train_one_epoch, evaluate_one_epoch
from src.types.task_protocol import TaskProtocol
from src.utils.metrics import multilabel_metrics_fn


class Trainer:
    """Generic training helper class.

    Parameters
    ----------
    args: dict
        Configuration parameters for training (data directories, hyperparameters, etc.).
    task: TaskProtocol
        A task specification with ``make_loaders`` and ``infer_*`` methods and a
        ``problem_type`` attribute (``"multiclass"`` or ``"multilabel"``).
    model_builder: callable
        A function that constructs the neural network given input/output sizes and
        other model hyperparameters.
    """

    def __init__(self, args: Dict[str, Any], task: TaskProtocol, model_builder: Any) -> None:
        self.args = args
        self.task = task
        self.model_builder = model_builder

        # Device selection
        self.device: torch.device = args.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp: bool = bool(args.get("amp", False) and self.device.type in {"cuda", "mps"})

        # Data loaders
        self.train_loader, self.val_loader, _ = self.task.make_loaders(
            data_root=args["data_root"],
            batch_size=args["batch"],
            **args.get("data_loader_kwargs", {})
        )

        flat_args: Dict[str, Any] = dict(self.args)
        flat_args.update(self.args.get("data_loader_kwargs", {}))

        # Model construction
        d_in = self.task.infer_input_dim(flat_args)
        theta = self.task.infer_theta(flat_args)
        n_classes = self.task.infer_num_classes(flat_args)

        # For regression tasks, the output dimension might need to be calculated differently
        if self.task.problem_type == "regression":
            pred_len = flat_args.get("pred_len", 1)  # Default to 1 if not specified
            n_classes = n_classes * pred_len

        block_cfg = args["block_cfg_ctor"](theta)
        self.model: nn.Module = self.model_builder(
            d_in=d_in,
            n_classes=n_classes,
            d_model=args["d_model"],
            depth=args["depth"],
            block_cfg=block_cfg,
        ).to(self.device)

        # Optimiser and scheduler
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=args["lr"],
            weight_decay=args["wd"],
            betas=(0.9, 0.95),
        )

        warmup_epochs = max(0, int(args.get("warmup_epochs", 0)))
        if warmup_epochs > 0:
            self.sch = SequentialLR(
                self.opt,
                schedulers=[
                    LinearLR(self.opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs),
                    CosineAnnealingLR(self.opt, T_max=args["epochs"] - warmup_epochs),
                ],
                milestones=[warmup_epochs],
            )
        else:
            self.sch = CosineAnnealingLR(self.opt, T_max=args["epochs"])

        # AMP scaler
        if self.amp and self.device.type == "cuda":
            try:
                self.scaler: Optional[torch.cuda.amp.GradScaler] = torch.amp.GradScaler(device="cuda", enabled=True)
            except AttributeError:  # older PyTorch
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None

        # Exponential moving average (optional)
        self.ema = None  # can be replaced with an EMA object if desired
        # Logging (stubbed out; replace with your own logger if needed)
        self.tb = None

        # Loss function and metrics based on task type
        if self.task.problem_type == "multiclass":
            ls = float(args.get("label_smoothing", 0.0))
            self.criterion = nn.CrossEntropyLoss(label_smoothing=ls) if ls > 0 else nn.CrossEntropyLoss()
            self.metrics_fn = None
            self.early_key = "acc"
            self.best_metric = float("-inf")
            self.history: Dict[str, list] = {
                "train_loss": [], "train_acc": [], "train_f1_micro": [],
                "val_loss": [], "val_acc": [], "val_f1_micro": [],
            }
        elif self.task.problem_type == "multilabel":
            # Multi‑label tasks expect binary targets per class

            pos_weight = None
            if args.get("pos_weight", None) is not None:
                pos_weight = self.compute_pos_weights().to(self.device)
                print(f"📊 Computed pos_weight for imbalanced classes: {pos_weight.cpu().numpy()}")

            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            thr = float(args.get("threshold", 0.5))
            self.metrics_fn = multilabel_metrics_fn(threshold=thr)
            self.early_key = args.get("early_key", "f1_micro")
            self.best_metric = float("-inf")
            self.history: Dict[str, list] = {
                "train_loss": [], "train_acc": [], "train_f1_micro": [],
                "val_loss": [], "val_acc": [], "val_f1_micro": [],
            }
        elif self.task.problem_type == "regression":
            self.criterion = nn.MSELoss()
            self.metrics_fn = lambda p, t: {"mse": nn.MSELoss()(p, t).item(), "mae": nn.L1Loss()(p, t).item()}
            self.early_key = args.get("early_key", "mse")
            self.best_metric = float("inf")  # Lower is better for regression
            self.history: Dict[str, list] = {
                "train_loss": [], "train_mse": [], "train_mae": [],
                "val_loss": [], "val_mse": [], "val_mae": [],
            }
        else:
            raise ValueError(f"Unsupported problem_type: {self.task.problem_type}")

        # Early stopping state
        self.bad_epochs = 0
        self.patience = int(args.get("patience", 20))
        self.min_delta = float(args.get("min_delta", 0.0))

    def _should_stop(self, current: float) -> Tuple[bool, bool]:
        """Check for improvement and whether early stopping condition is met.

        This method is the single source of truth for what constitutes an
        improvement. It updates the best metric and bad epoch counter.

        Returns
        -------
        is_better : bool
            True if the current metric is better than the best seen so far.
        should_stop : bool
            True if the patience has been exceeded.
        """
        is_better = False
        # Check for improvement, including min_delta
        if self.task.problem_type == "regression":
            if current < self.best_metric - self.min_delta:
                is_better = True
        else:  # classification
            if current > self.best_metric + self.min_delta:
                is_better = True

        if is_better:
            self.best_metric = current  # Update best metric here
            self.bad_epochs = 0
            return True, False  # Is better, should not stop

        # No improvement
        self.bad_epochs += 1
        should_stop = self.bad_epochs > self.patience
        return False, should_stop

    def save_history(self, save_dir: str) -> str:
        """Save training history to a JSON file.

        Parameters
        ----------
        save_dir : str
            Directory where the history file should be saved.

        Returns
        -------
        str
            Path to the saved history file.
        """
        history_path = os.path.join(save_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        return history_path

    @staticmethod
    def load_history(save_dir: str) -> Dict[str, list]:
        """Load training history from a JSON file.

        Parameters
        ----------
        save_dir : str
            Directory where the history file is saved.

        Returns
        -------
        dict
            Dictionary containing training history.
        """
        history_path = os.path.join(save_dir, "history.json")
        with open(history_path, "r") as f:
            return json.load(f)

    def fit(self) -> Tuple[float, str]:
        """Run the full training loop.

        Returns
        -------
        best_metric : float
            The best observed validation metric during training.
        best_path : str
            The path to the saved checkpoint corresponding to the best metric.
        """
        save_dir = self.args.get("save_dir", ".")
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, "best.pt")
        primed_scheduler = False

        # Pass task-specific parameters to the training loops
        loop_kwargs = {}
        if self.task.problem_type == "regression":
            flat_args: Dict[str, Any] = dict(self.args)
            flat_args.update(self.args.get("data_loader_kwargs", {}))
            loop_kwargs["pred_len"] = flat_args.get("pred_len")
            loop_kwargs["d_out"] = self.task.infer_num_classes(flat_args)


        for ep in range(self.args["epochs"] + 1):
            # Training
            tr = train_one_epoch(
                self.model,
                self.train_loader,
                self.opt,
                self.scaler,
                self.device,
                self.amp,
                self.criterion,
                self.ema,
                metrics_fn=self.metrics_fn,
                **loop_kwargs,
            )
            # Validation
            va = evaluate_one_epoch(
                self.model,
                self.val_loader,
                self.device,
                self.amp,
                self.criterion,
                self.ema,
                metrics_fn=self.metrics_fn,
                **loop_kwargs,
            )

            # Scheduler step after first optimisation step
            if tr.get("stepped", False):
                self.sch.step(); primed_scheduler = True
            elif not primed_scheduler:
                self.opt.step(); self.sch.step(); primed_scheduler = True

            # Track history
            self.history["train_loss"].append(tr.get("loss", 0.0))
            self.history["val_loss"].append(va.get("loss", 0.0))
            if self.task.problem_type == "regression":
                self.history["train_mse"].append(tr.get("mse", 0.0))
                self.history["train_mae"].append(tr.get("mae", 0.0))
                self.history["val_mse"].append(va.get("mse", 0.0))
                self.history["val_mae"].append(va.get("mae", 0.0))
            else:
                self.history["train_acc"].append(tr.get("acc", 0.0))
                self.history["train_f1_micro"].append(tr.get("f1_micro", 0.0))
                self.history["val_acc"].append(va.get("acc", 0.0))
                self.history["val_f1_micro"].append(va.get("f1_micro", 0.0))

            # Determine current metric for early stopping
            cur = va.get(self.early_key, None)
            metric_for_es = cur if cur is not None else (va["loss"] if self.task.problem_type == "regression" else -va["loss"])

            # Check for improvement and early stopping using the single source of truth
            is_better, should_stop = self._should_stop(metric_for_es)

            if is_better:
                # Filter out non-picklable objects from args (like functions)
                save_args = {k: v for k, v in self.args.items()
                            if not callable(v) and k not in {"block_cfg_ctor"}}
                torch.save({
                    "model": self.model.state_dict(),
                    "epoch": ep,
                    "val": va,
                    "args": save_args,
                    "history": self.history,
                }, best_path)
                print(f"💾 saved best model to {best_path}")
                print(f"✅ new best {self.early_key} {self.best_metric:.4f}")

            if should_stop:
                print(
                    f"⏹ Early stopping (patience={self.patience}, best={self.best_metric:.4f})."
                )
                break

            # Logging (replace prints with proper logging if needed)
            # Show a concise summary each epoch
            metric_name = self.early_key
            if self.task.problem_type == "regression":
                tr_metric = tr.get(metric_name, tr.get("mse", float("nan")))
                va_metric = va.get(metric_name, va.get("mse", float("nan")))
            else:
                tr_metric = tr.get(metric_name, tr.get("acc", tr.get("f1_micro", float("nan"))))
                va_metric = va.get(metric_name, va.get("acc", va.get("f1_micro", float("nan"))))
            print(
                f"Epoch {ep:03d}/{self.args['epochs']} | "
                f"train {tr['loss']:.4f}/{tr_metric:.4f} | "
                f"val {va['loss']:.4f}/{va_metric:.4f} | "
                f"t {tr['time_s']:.1f}s/{va['time_s']:.1f}s | "
                f"lr {tr.get('lr', self.opt.param_groups[0]['lr']):.2e}"
            )

        # Save final history to JSON file
        history_path = self.save_history(save_dir)
        print(f"📊 Training history saved to {history_path}")

        # Return best metric and checkpoint path
        return self.best_metric, best_path

    def compute_pos_weights(self) -> torch.Tensor:
        """
        Compute per-class positive weights for BCEWithLogitsLoss.

        For imbalanced datasets, pos_weight = N_neg / N_pos helps the model
        pay more attention to rare positive samples.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_classes,) with pos_weight for each class.
        """
        all_labels = []

        print("Computing pos_weight from training set...")
        for batch in self.train_loader:
            _, y = batch
            all_labels.append(y)

        y_train = torch.cat(all_labels, dim=0)
        n_samples = y_train.shape[0]

        pos = y_train.sum(dim=0)
        neg = (1 - y_train).sum(dim=0)

        pos_weight = neg / (pos + 1e-8)

        # Log class distribution
        print(f"  Total training samples: {n_samples}")
        print(f"  Positive samples per class: {pos.numpy()}")
        print(f"  Negative samples per class: {neg.numpy()}")
        prevalence = (pos / n_samples * 100).numpy()
        print(f"  Class prevalence: {prevalence}%")

        return pos_weight
