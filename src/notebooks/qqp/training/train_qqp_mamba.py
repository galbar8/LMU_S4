"""
QQP Mamba Training Script
Trains a Mamba model on the Quora Question Pairs dataset for duplicate detection.

BASIC USAGE:
    python train_qqp_mamba.py --device cuda --amp --evaluate

PROGRAMMATIC USAGE:
    from src.notebooks.qqp.training.train_qqp_mamba import main_fraction, main_multi_fraction

    # Single fraction
    main_fraction(fraction=0.1, epochs=10, device="cuda")

    # Multiple fractions
    results = main_multi_fraction([0.1, 0.25, 0.5], epochs=20, device="cuda")

KEY ARGUMENTS:
  --device {auto,cuda,mps,cpu}  Device (default: auto)
  --amp                         Enable mixed precision
  --batch N                     Batch size (default: 256)
  --epochs N                    Number of epochs (default: 100)
  --d_model N                   Model dimension (default: 128)
  --depth N                     Number of layers (default: 1)
  --max_len N                   Max sequence length per question (default: 64)
  --fraction F                  Data fraction (default: None, use all)
  --evaluate                    Evaluate on test set

Run with --help for all options.
"""

from __future__ import annotations
from typing import Dict, Any
import argparse
import torch
from pathlib import Path
import sys

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.notebooks.qqp.qqp_task import QQPTask
from src.train_utils.trainer import Trainer
from src.utils.block_factory import make_mamba_block_cfg_ctor
from src.utils.visualization import plot_classification_history as plot_history
from src.utils.checkpoint import load_trainer_from_checkpoint
from src.eval.eval_utils import evaluate_classification_model as evaluate_best_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Mamba model on QQP dataset')

    # Data parameters
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to data directory (default: auto-detect)')
    parser.add_argument('--batch', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers (default: 0)')

    # QQP-specific data parameters
    parser.add_argument('--max_len', type=int, default=64,
                        help='Max sequence length per question (default: 64)')
    parser.add_argument('--max_vocab', type=int, default=50000,
                        help='Maximum vocabulary size (default: 50000)')
    parser.add_argument('--min_freq', type=int, default=2,
                        help='Minimum word frequency (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs (default: 5)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')

    # Model parameters
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension (default: 128)')
    parser.add_argument('--depth', type=int, default=1,
                        help='Number of layers (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')

    # Mamba-specific parameters
    parser.add_argument('--d_state', type=int, default=16,
                        help='SSM state dimension (default: 16)')
    parser.add_argument('--expand_factor', type=int, default=2,
                        help='Mamba expansion factor (default: 2)')
    parser.add_argument('--d_conv', type=int, default=4,
                        help='Local convolution width (default: 4)')

    # Device and optimization
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision')

    # Checkpointing and evaluation
    parser.add_argument('--save_dir', type=str, default='./runs/qqp_mamba_task',
                        help='Checkpoint directory (default: ./runs/qqp_mamba_task)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate on test set after training')
    parser.add_argument('--plot', action='store_true',
                        help='Plot training history')

    # Data fraction
    parser.add_argument('--fraction', type=float, default=None,
                        help='Fraction of training data (e.g., 0.1 for 10%%)')

    return parser.parse_args()


def setup_device(device_str: str):
    """Setup device and determine if AMP should be enabled."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Auto-detected CUDA GPU: {torch.cuda.get_device_name(0)}")
            return device, True
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Auto-detected MPS (Apple Silicon)")
            return device, False
        else:
            device = torch.device('cpu')
            print("Auto-detected CPU")
            return device, False
    else:
        device = torch.device(device_str)
        print(f"Using {device_str.upper()}")
        return device, device_str == 'cuda'


def build_config(args_parsed):
    """Build configuration dictionary from parsed arguments."""
    # Setup data root
    if args_parsed.data_root is None:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent.parent
        data_root = str(project_root / "src" / "datasets" / "qqp" / "data")
    else:
        data_root = args_parsed.data_root

    # Setup device
    device, amp_enabled = setup_device(args_parsed.device)
    if args_parsed.amp:
        amp_enabled = True

    args: Dict[str, Any] = {
        "data_root": data_root,
        "batch": args_parsed.batch,
        "data_loader_kwargs": {
            "num_workers": args_parsed.num_workers,
            "pin_memory": device.type == "cuda",
            "persistent_workers": args_parsed.num_workers > 0,
            "seed": args_parsed.seed,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "lowercase": True,
            "max_len": args_parsed.max_len,
            "max_vocab": args_parsed.max_vocab,
            "min_freq": args_parsed.min_freq,
        },
        "epochs": args_parsed.epochs,
        "lr": args_parsed.lr,
        "wd": args_parsed.wd,
        "amp": amp_enabled,
        "save_dir": args_parsed.save_dir,
        "warmup_epochs": args_parsed.warmup_epochs,
        "patience": args_parsed.patience,
        "min_delta": 0.001,
        "early_key": "accuracy",
        "d_model": args_parsed.d_model,
        "depth": args_parsed.depth,
        "dropout": args_parsed.dropout,
        "mlp_ratio": 2.0,
        "droppath_final": 0.0,
        "layerscale_init": 0.0,
        "residual_gain": 1.0,
        "pool": "mean",
        "d_state": args_parsed.d_state,
        "expand_factor": args_parsed.expand_factor,
        "d_conv": args_parsed.d_conv,
        "use_external_mlp": False,
        "max_len": args_parsed.max_len,
        "device": device,
    }

    # Add fraction if specified
    if args_parsed.fraction is not None:
        args["fraction"] = args_parsed.fraction

    # Create block configuration
    args["block_cfg_ctor"] = make_mamba_block_cfg_ctor(
        d_state=args["d_state"],
        expand_factor=args["expand_factor"],
        d_conv=args["d_conv"],
        use_external_mlp=args["use_external_mlp"],
        dropout=args["dropout"],
        mlp_ratio=args["mlp_ratio"],
        droppath_final=args["droppath_final"],
        layerscale_init=args["layerscale_init"],
        residual_gain=args["residual_gain"],
        pool=args["pool"],
    )

    return args


def train(args: Dict[str, Any]):
    """Train the model."""
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}")
    print(f"Device: {args['device']}")
    print(f"Model: Mamba (d_model={args['d_model']}, depth={args['depth']})")
    print(f"Batch size: {args['batch']}")
    print(f"Epochs: {args['epochs']}")
    if 'fraction' in args:
        print(f"Data fraction: {args['fraction']*100:.1f}%")
    print(f"{'='*80}\n")

    # Create task and trainer
    task = QQPTask()
    trainer = Trainer(args=args, task=task)

    # Train
    best_metric, ckpt_path = trainer.fit()

    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Best accuracy: {best_metric:.4f}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*80}\n")

    return best_metric, ckpt_path


def evaluate(args: Dict[str, Any], best_model_path: str):
    """Evaluate the best model on test set."""
    print(f"\n{'='*80}")
    print("Evaluating on Test Set")
    print(f"{'='*80}\n")

    evaluate_best_model(
        args=args,
        task=QQPTask(),
        best_model_path=best_model_path,
        num_classes=2,
        use_test_set=True,
    )

    print(f"\n{'='*80}")
    print("Evaluation Complete")
    print(f"{'='*80}\n")


def main():
    """Main function."""
    args_parsed = parse_args()
    args = build_config(args_parsed)
    best_metric, ckpt_path = train(args)

    if args_parsed.plot:
        try:
            trainer = load_trainer_from_checkpoint(ckpt_path, args, QQPTask())
            plot_history(trainer.history, model_name="Mamba")
            print("Training history plotted successfully.")
        except Exception as e:
            print(f"Warning: Could not plot training history: {e}")

    if args_parsed.evaluate:
        evaluate(args, ckpt_path)


def main_fraction(fraction: float = 0.1, epochs: int = 100, device: str = "auto",
                  evaluate_model: bool = True, **kwargs):
    """Train with a fraction of the data for quick experiments."""
    device_obj, amp_enabled = setup_device(device)

    # Auto-detect data root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent.parent
    data_root = str(project_root / "src" / "datasets" / "qqp" / "data")

    args: Dict[str, Any] = {
        "data_root": data_root,
        "batch": kwargs.get("batch", 256),
        "data_loader_kwargs": {
            "num_workers": kwargs.get("num_workers", 0),
            "pin_memory": device_obj.type == "cuda",
            "persistent_workers": kwargs.get("num_workers", 0) > 0,
            "seed": kwargs.get("seed", 42),
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "lowercase": True,
            "max_len": kwargs.get("max_len", 64),
            "max_vocab": kwargs.get("max_vocab", 50000),
            "min_freq": kwargs.get("min_freq", 2),
        },
        "epochs": epochs,
        "lr": kwargs.get("lr", 1e-3),
        "wd": kwargs.get("wd", 1e-4),
        "amp": kwargs.get("amp", amp_enabled),
        "save_dir": kwargs.get("save_dir", f"./runs/qqp_mamba_task_frac_{int(fraction*100)}"),
        "warmup_epochs": kwargs.get("warmup_epochs", 5),
        "patience": kwargs.get("patience", 5),
        "min_delta": 0.001,
        "early_key": "accuracy",
        "d_model": kwargs.get("d_model", 128),
        "depth": kwargs.get("depth", 1),
        "dropout": kwargs.get("dropout", 0.1),
        "mlp_ratio": 2.0,
        "droppath_final": 0.0,
        "layerscale_init": 0.0,
        "residual_gain": 1.0,
        "pool": "mean",
        "d_state": kwargs.get("d_state", 16),
        "expand_factor": kwargs.get("expand_factor", 2),
        "d_conv": kwargs.get("d_conv", 4),
        "use_external_mlp": False,
        "max_len": kwargs.get("max_len", 64),
        "device": device_obj,
        "fraction": fraction,
    }

    # Create block configuration
    args["block_cfg_ctor"] = make_mamba_block_cfg_ctor(
        d_state=args["d_state"],
        expand_factor=args["expand_factor"],
        d_conv=args["d_conv"],
        use_external_mlp=args["use_external_mlp"],
        dropout=args["dropout"],
        mlp_ratio=args["mlp_ratio"],
        droppath_final=args["droppath_final"],
        layerscale_init=args["layerscale_init"],
        residual_gain=args["residual_gain"],
        pool=args["pool"],
    )

    best_metric, ckpt_path = train(args)

    if evaluate_model:
        evaluate(args, ckpt_path)

    return best_metric, ckpt_path


def main_multi_fraction(fractions: list[float] = None, epochs: int = 100, device: str = "auto",
                        evaluate_model: bool = True, **kwargs):
    """Run training over multiple fractions and collect results."""
    if fractions is None:
        fractions = [0.1, 0.25, 0.5]

    device_obj, amp_enabled = setup_device(device)

    # Auto-detect data root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent.parent
    data_root = str(project_root / "src" / "datasets" / "qqp" / "data")

    results = {}

    for frac in fractions:
        print(f"\nTraining with fraction: {frac}")

        args: Dict[str, Any] = {
            "data_root": data_root,
            "batch": kwargs.get("batch", 256),
            "data_loader_kwargs": {
                "num_workers": kwargs.get("num_workers", 0),
                "pin_memory": device_obj.type == "cuda",
                "persistent_workers": kwargs.get("num_workers", 0) > 0,
                "seed": kwargs.get("seed", 42),
                "val_ratio": 0.1,
                "test_ratio": 0.1,
                "lowercase": True,
                "max_len": kwargs.get("max_len", 64),
                "max_vocab": kwargs.get("max_vocab", 50000),
                "min_freq": kwargs.get("min_freq", 2),
            },
            "epochs": epochs,
            "lr": kwargs.get("lr", 1e-3),
            "wd": kwargs.get("wd", 1e-4),
            "amp": kwargs.get("amp", amp_enabled),
            "save_dir": f"./runs/qqp_mamba_task_frac_{int(frac*100)}",
            "warmup_epochs": kwargs.get("warmup_epochs", 5),
            "patience": kwargs.get("patience", 5),
            "min_delta": 0.001,
            "early_key": "accuracy",
            "d_model": kwargs.get("d_model", 128),
            "depth": kwargs.get("depth", 1),
            "dropout": kwargs.get("dropout", 0.1),
            "mlp_ratio": 2.0,
            "droppath_final": 0.0,
            "layerscale_init": 0.0,
            "residual_gain": 1.0,
            "pool": "mean",
            "d_state": kwargs.get("d_state", 16),
            "expand_factor": kwargs.get("expand_factor", 2),
            "d_conv": kwargs.get("d_conv", 4),
            "use_external_mlp": False,
            "max_len": kwargs.get("max_len", 64),
            "device": device_obj,
            "fraction": frac,
        }

        # Create block configuration
        args["block_cfg_ctor"] = make_mamba_block_cfg_ctor(
            d_state=args["d_state"],
            expand_factor=args["expand_factor"],
            d_conv=args["d_conv"],
            use_external_mlp=args["use_external_mlp"],
            dropout=args["dropout"],
            mlp_ratio=args["mlp_ratio"],
            droppath_final=args["droppath_final"],
            layerscale_init=args["layerscale_init"],
            residual_gain=args["residual_gain"],
            pool=args["pool"],
        )

        try:
            trainer = Trainer(args=args, task=QQPTask())
            best_metric, best_path = trainer.fit()

            print(f"\nTraining complete for fraction {frac}! Best validation accuracy: {best_metric:.4f}")
            print(f"Best model saved to: {best_path}")

            plot_history(trainer.history, model_name="Mamba")

            if evaluate_model:
                evaluate_best_model(
                    args=args,
                    task=QQPTask(),
                    best_model_path=best_path,
                    num_classes=2,
                    use_test_set=True,
                )

            results[frac] = {
                'best_metric': best_metric,
                'checkpoint_path': best_path,
                'status': 'success'
            }

        except Exception as e:
            print(f"\nTraining failed for fraction {frac}: {e}")
            results[frac] = {
                'best_metric': None,
                'checkpoint_path': None,
                'status': 'failed',
                'error': str(e)
            }

    return results


if __name__ == "__main__":
    main()
    main_multi_fraction()

