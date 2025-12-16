from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from src.utils.checkpoint import load_test_results

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_run_results(run_dir: str) -> Dict[str, Any]:
    """
    Load results from a run directory including test metrics.

    Args:
        run_dir: Path to the run directory

    Returns:
        Dictionary with results including history and test metrics
    """

    run_path = Path(run_dir)
    checkpoint_name = "best.pt"
    checkpoint_path = run_path / checkpoint_name

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return {}

    history = load_checkpoint_history(str(checkpoint_path))

    # Extract key metrics
    results = {
        'run_dir': str(run_dir),
        'history': history,
        'checkpoint_path': str(checkpoint_path),
    }

    # Extract the best metrics from history
    if 'val_loss' in history and history['val_loss']:
        results['best_val_loss'] = min(history['val_loss'])
        results['final_val_loss'] = history['val_loss'][-1]
        results['epochs_trained'] = len(history['val_loss'])

    if 'val_mae' in history and history['val_mae']:
        results['best_val_mae'] = min(history['val_mae'])
        results['final_val_mae'] = history['val_mae'][-1]

    if 'train_loss' in history and history['train_loss']:
        results['final_train_loss'] = history['train_loss'][-1]

    # Load test metrics from saved test_results.json
    saved_test_results = load_test_results(str(checkpoint_path))
    if saved_test_results:
        # For regression tasks (PPG, ETTS)
        results['test_mse'] = saved_test_results.get('mse')
        results['test_mae'] = saved_test_results.get('mae')
        results['test_rmse'] = saved_test_results.get('rmse')

        # For classification tasks (PS-MNIST, ESC-50, PTB-XL)
        results['test_acc'] = saved_test_results.get('test_accuracy')
        results['test_loss'] = saved_test_results.get('test_loss')
        results['per_class_accuracy'] = saved_test_results.get('per_class_accuracy')

        # For multi-label classification (PTB-XL)
        results['test_f1_micro'] = saved_test_results.get('f1_micro')
        results['threshold'] = saved_test_results.get('threshold')

        # Number of samples
        results['num_test_samples'] = saved_test_results.get('num_samples') or saved_test_results.get('num_test_samples')

    return results

def load_checkpoint_history(checkpoint_path: str) -> Dict[str, List[float]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        history = checkpoint.get('history', {})

        if not history:
            print(f"No history found in {checkpoint_path}")
            return {}

        return history
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return {}

def load_all_experiments(
    base_dir: str,
    models: List[str],
    fractions: List[float],
    task_name: str = "task",
    sub_task: str = None,
    print_logs: bool = True
) -> Dict[str, Dict[float, Dict[str, Any]]]:

    base_path = Path(base_dir)
    all_results = {}

    for model in models:
        all_results[model] = {}

        for frac in fractions:
            run_name = f"{task_name}_{model}_task"

            if sub_task:
                run_name = f"{run_name}_{sub_task}"

            if frac < 1.0:
                # Fractional dataset
                frac_pct = int(frac * 100)
                run_name = f"{run_name}_frac_{frac_pct}"

            run_dir = base_path / run_name

            if run_dir.exists():
                if print_logs:
                    print(f"Loading {model.upper()} @ {frac*100:.0f}%: {run_name}")
                results = load_run_results(str(run_dir))
                all_results[model][frac] = results
            else:
                print(f"Run not found: {run_name}")
                all_results[model][frac] = {}

    return all_results

def plot_learning_curves(
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
    metric: str = 'val_loss',
    title_suffix: str = '',
    ylabel: str = 'Validation Loss'
):
    """
    Plot learning curves for all model/fraction combinations.

    Args:
        all_results: Results from load_all_experiments()
        metric: Metric to plot from history
        title_suffix: Additional text for title
        ylabel: Y-axis label
    """
    models = list(all_results.keys())
    fractions = sorted(set(f for model_results in all_results.values()
                          for f in model_results.keys()))

    n_fracs = len(fractions)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    colors = {'s4': '#1f77b4', 'lmu': '#ff7f0e', 'default': '#2ca02c'}

    for idx, frac in enumerate(fractions):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break

        for model in models:
            if frac in all_results[model] and all_results[model][frac]:
                history = all_results[model][frac].get('history', {})
                if metric in history and history[metric]:
                    epochs = range(1, len(history[metric]) + 1)
                    color = colors.get(model.lower(), colors['default'])
                    ax.plot(epochs, history[metric],
                           label=model.upper(),
                           linewidth=2,
                           marker='o',
                           markersize=3,
                           color=color,
                           alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'Data Fraction: {frac*100:.0f}%', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_fracs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Learning Curves: {metric.replace("_", " ").title()}{title_suffix}',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def plot_metric_comparison_bar(
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
    metric_key: str = 'best_val_mae',
    title: str = 'Best Validation MAE by Model and Data Fraction',
    ylabel: str = 'MAE (bpm)',
):
    """
    Create bar chart comparing a metric across models and fractions.

    Args:
        all_results: Results from load_all_experiments()
        metric_key: Key for the metric to compare
        title: Plot title
        ylabel: Y-axis label
    """
    models = list(all_results.keys())
    fractions = sorted(set(f for model_results in all_results.values()
                          for f in model_results.keys()))

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(fractions))
    width = 0.35
    colors = {'s4': '#1f77b4', 'lmu': '#ff7f0e'}

    for i, model in enumerate(models):
        values = []
        for frac in fractions:
            if frac in all_results[model] and all_results[model][frac]:
                val = all_results[model][frac].get(metric_key, np.nan)
                values.append(val if val is not None else np.nan)
            else:
                values.append(np.nan)

        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, values, width,
                     label=model.upper(),
                     alpha=0.8,
                     color=colors.get(model.lower(), '#2ca02c'))

        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Data Fraction', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(f*100)}%' for f in fractions])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

def create_test_comparison_table(
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
    metrics: List[str] = ['test_mae', 'test_mse', 'test_rmse']
) -> pd.DataFrame:
    """
    Create a comparison table focused on test metrics.

    Args:
        all_results: Results from load_all_experiments()
        metrics: List of test metric keys to include

    Returns:
        pandas DataFrame with test metrics comparison
    """
    rows = []

    for model in sorted(all_results.keys()):
        for frac in sorted(all_results[model].keys()):
            results = all_results[model][frac]
            if not results:
                continue

            # Only include if test metrics exist
            if not any(results.get(metric) is not None for metric in metrics):
                continue

            row = {
                'model': model.upper(),
                'data_pct': int(frac * 100),
            }

            for metric in metrics:
                val = results.get(metric)
                row[metric] = val if val is not None else np.nan

            # Add number of test samples if available
            if 'num_test_samples' in results:
                row['num_samples'] = results['num_test_samples']

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_test_metric_comparison(
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
    metric_key: str = 'test_mae',
    title: str = 'Test MAE Comparison',
    ylabel: str = 'Test MAE (bpm)',
    lower_is_better: bool = True
):
    """
    Create bar chart comparing test metrics across models and fractions.

    Args:
        all_results: Results from load_all_experiments()
        metric_key: Key for the test metric to compare
        title: Plot title
        ylabel: Y-axis label
        lower_is_better: If True, lower values are better (for MAE, MSE)
    """
    models = list(all_results.keys())
    fractions = sorted(set(f for model_results in all_results.values()
                          for f in model_results.keys()))

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(fractions))
    width = 0.35
    colors = {'s4': '#1f77b4', 'lmu': '#ff7f0e'}

    for i, model in enumerate(models):
        values = []
        for frac in fractions:
            if frac in all_results[model] and all_results[model][frac]:
                val = all_results[model][frac].get(metric_key, np.nan)
                values.append(val if val is not None else np.nan)
            else:
                values.append(np.nan)

        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, values, width,
                     label=model.upper(),
                     alpha=0.8,
                     color=colors.get(model.lower(), '#2ca02c'))

        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Data Fraction', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(f*100)}%' for f in fractions])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def plot_test_data_efficiency(
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
    metric_key: str = 'test_mae',
    ylabel: str = 'Test MAE (bpm)'
):
    """
    Plot test metric vs data fraction to show data efficiency on test set.

    Args:
        all_results: Results from load_all_experiments()
        metric_key: Test metric to plot
        ylabel: Y-axis label
    """
    models = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'s4': '#1f77b4', 'lmu': '#ff7f0e'}
    markers = {'s4': 'o', 'lmu': 's'}

    for model in models:
        fractions = []
        values = []

        for frac in sorted(all_results[model].keys()):
            if all_results[model][frac]:
                val = all_results[model][frac].get(metric_key)
                if val is not None and not np.isnan(val):
                    fractions.append(frac * 100)  # Convert to percentage
                    values.append(val)

        if fractions:
            ax.plot(fractions, values,
                   label=model.upper(),
                   marker=markers.get(model.lower(), 'o'),
                   markersize=10,
                   linewidth=2.5,
                   color=colors.get(model.lower(), '#2ca02c'),
                   alpha=0.8)

    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Test Set Performance: Data Efficiency Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([10, 25, 50, 100])

    plt.tight_layout()
    plt.show()


def plot_data_efficiency(
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
    metric_key: str = 'best_val_mae',
    ylabel: str = 'Best Validation MAE (bpm)'
):
    """
    Plot metric vs data fraction to show data efficiency.

    Args:
        all_results: Results from load_all_experiments()
        metric_key: Metric to plot
        ylabel: Y-axis label
    """
    models = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'s4': '#1f77b4', 'lmu': '#ff7f0e'}
    markers = {'s4': 'o', 'lmu': 's'}

    for model in models:
        fractions = []
        values = []

        for frac in sorted(all_results[model].keys()):
            if all_results[model][frac]:
                val = all_results[model][frac].get(metric_key)
                if val is not None and not np.isnan(val):
                    fractions.append(frac * 100)  # Convert to percentage
                    values.append(val)

        if fractions:
            ax.plot(fractions, values,
                   label=model.upper(),
                   marker=markers.get(model.lower(), 'o'),
                   markersize=8,
                   linewidth=2.5,
                   color=colors.get(model.lower(), '#2ca02c'),
                   alpha=0.8)

    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Data Efficiency: Performance vs Training Data Size',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([10, 25, 50, 100])

    plt.tight_layout()
    plt.show()


def plot_single_metric_comparison(
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
    metric: str = 'val_loss',
    title: str = 'Metric Comparison',
    ylabel: str = 'Metric Value',
):
    """
    Plot learning curves comparing models for a single metric (no multiple fractions).
    Useful for tasks like PS-MNIST where we only have full dataset runs.

    Args:
        all_results: Results from load_all_experiments() or similar structure
        metric: Metric to plot from history (e.g., 'val_loss', 'val_acc', 'train_loss')
        title: Plot title
        ylabel: Y-axis label
    """
    models = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'s4': '#1f77b4', 'lmu': '#ff7f0e', 'default': '#2ca02c'}
    markers = {'s4': 'o', 'lmu': 's', 'default': '^'}

    for model in models:
        # Get the first available fraction (usually 1.0 for full dataset)
        fractions = sorted(all_results[model].keys())
        if not fractions:
            continue

        # Use the first available fraction
        frac = fractions[0]
        results = all_results[model][frac]

        if results:
            history = results.get('history', {})
            if metric in history and history[metric]:
                epochs = range(1, len(history[metric]) + 1)

                # Convert to percentage if it's accuracy
                values = history[metric]
                if metric == 'val_acc' or metric == 'train_acc':
                    values = [v * 100 for v in values]

                color = colors.get(model.lower(), colors['default'])
                marker = markers.get(model.lower(), markers['default'])

                ax.plot(epochs, values,
                       label=model.upper(),
                       linewidth=2.5,
                       marker=marker,
                       markersize=4,
                       markevery=max(1, len(epochs) // 10),
                       color=color,
                       alpha=0.8)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
