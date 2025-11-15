# src/utils/common.py
from typing import Dict

import torch, random, numpy as np
from contextlib import contextmanager

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def count_params(model, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable params. If False, count all params.

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def device_auto():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available(): # for Apple Silicon
        torch.set_float32_matmul_precision("high")
        return torch.device("mps")
    else:
        return torch.device("cpu")

@contextmanager
def amp_autocast(enabled: bool):
    if not enabled: yield
    else:
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
            yield


def print_model_details(model, trainer=None):
    """
    Print comprehensive model details including architecture, parameters, and configuration.

    Args:
        model: PyTorch model
        trainer: Optional trainer instance for training history
    """

    # Header
    print("\n" + "=" * 70)
    print(f"{'MODEL DETAILS':^70}")
    print("=" * 70)

    # 1. Model Type
    print(f"\n{'ARCHITECTURE':^70}")
    print("-" * 70)
    print(f"Model Type: {type(model).__name__}")

    core_type = "Unknown"
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        first_block = model.blocks[0]
        # Get the core layer from the block (e.g., S4Block, LMUBlock)
        if hasattr(first_block, 'core'):
            core_type = type(first_block.core).__name__
        else:
            core_type = type(first_block).__name__

    print(f"Core Layer Type: {core_type}")
    print(f"Number of Blocks: {len(model.blocks) if hasattr(model, 'blocks') else 'N/A'}")

    # 2. Parameter Statistics
    print(f"\n{'PARAMETERS':^70}")
    print("-" * 70)
    total_params = count_params(model, trainable_only=False)
    trainable = count_params(model, trainable_only=True)
    non_trainable = total_params - trainable

    print(f"Total parameters:        {total_params:>15,}")
    print(f"Trainable parameters:    {trainable:>15,}")
    print(f"Non-trainable parameters:{non_trainable:>15,}")
    print(f"Model size (MB):         {(total_params * 4) / (1024 ** 2):>15.2f}")

    # 3. Model Configuration (from model attributes)
    print(f"\n{'MODEL ATTRIBUTES':^70}")
    print("-" * 70)

    # Extract common attributes if they exist
    if hasattr(model, 'd_model'):
        print(f"d_model:         {model.d_model:>6}")
    if hasattr(model, 'depth') or hasattr(model, 'num_layers'):
        depth = getattr(model, 'depth', getattr(model, 'num_layers', 'N/A'))
        print(f"depth:           {depth:>6}")
    if hasattr(model, 'dropout'):
        dropout = model.dropout.p if hasattr(model.dropout, 'p') else model.dropout
        print(f"dropout:         {dropout:>6}")
    if hasattr(model, 'mlp_ratio'):
        print(f"mlp_ratio:       {model.mlp_ratio:>6}")
    if hasattr(model, 'num_classes'):
        print(f"num_classes:     {model.num_classes:>6}")

    # Core-specific parameters
    if hasattr(model, 'd_state'):
        print(f"\nS4-Specific Parameters:")
        print(f"  d_state:       {model.d_state:>6}")
        if hasattr(model, 'channels'):
            print(f"  channels:      {model.channels:>6}")
        if hasattr(model, 'bidirectional'):
            print(f"  bidirectional: {model.bidirectional:>6}")
        if hasattr(model, 'mode'):
            print(f"  mode:          {model.mode:>6}")
    elif hasattr(model, 'memory_size'):
        print(f"\nLMU-Specific Parameters:")
        print(f"  memory_size:   {model.memory_size:>6}")
        if hasattr(model, 'theta'):
            print(f"  theta:         {model.theta:>6}")

    # 4. Module Structure
    print(f"\n{'MODULE BREAKDOWN':^70}")
    print("-" * 70)
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"{name:<30} {type(module).__name__:<30} {module_params:>10,}")

    # 5. Training History (if trainer provided)
    if trainer and hasattr(trainer, 'history') and trainer.history:
        print(f"\n{'TRAINING RESULTS':^70}")
        print("-" * 70)

        best_val_acc = max(trainer.history['val_acc']) if trainer.history['val_acc'] else 0
        best_epoch = trainer.history['val_acc'].index(best_val_acc) if trainer.history['val_acc'] else 0
        final_train_acc = trainer.history['train_acc'][-1] if trainer.history['train_acc'] else 0
        final_val_acc = trainer.history['val_acc'][-1] if trainer.history['val_acc'] else 0
        total_epochs = len(trainer.history['val_acc']) if trainer.history['val_acc'] else 0

        print(f"Total epochs trained:  {total_epochs:>6}")
        print(f"Best val accuracy:     {best_val_acc:>6.4f} (epoch {best_epoch})")
        print(f"Final train accuracy:  {final_train_acc:>6.4f}")
        print(f"Final val accuracy:    {final_val_acc:>6.4f}")
        print(f"Overfitting gap:       {(final_train_acc - final_val_acc):>6.4f}")

    # 6. Layer-by-Layer Breakdown
    print(f"\n{'LAYER-BY-LAYER PARAMETERS':^70}")
    print("-" * 70)
    print(f"{'Layer Name':<45} {'Shape':<20} {'Parameters':>10}")
    print("-" * 70)

    for name, param in model.named_parameters():
        if param.requires_grad:
            shape_str = str(list(param.shape))
            print(f"{name[:44]:<45} {shape_str:<20} {param.numel():>10,}")

    print("=" * 70 + "\n")
