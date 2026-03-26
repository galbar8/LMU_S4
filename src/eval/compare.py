from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from src.utils.checkpoint import load_test_results, load_checkpoint_history

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

        results['accuracy'] = saved_test_results.get('accuracy')
        results['loss'] = saved_test_results.get('loss')
        results['per_class_accuracy'] = saved_test_results.get('per_class_accuracy')

        # For multi-label classification (PTB-XL)
        results['test_f1_micro'] = saved_test_results.get('f1_micro')
        results['threshold'] = saved_test_results.get('threshold')

        # For classification tasks
        results['f1_score'] = saved_test_results.get('f1_score')
        results['precision'] = saved_test_results.get('precision')
        results['recall'] = saved_test_results.get('recall')
        results['pr_auc'] = saved_test_results.get('pr_auc')
        results['roc_auc'] = saved_test_results.get('roc_auc')

        # Number of samples
        results['num_test_samples'] = saved_test_results.get('num_samples') or saved_test_results.get('num_test_samples')

    return results


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
                frac_pct = int(frac * 100)
                run_name = f"{run_name}_frac_{frac_pct}"
                run_dir = base_path / run_name
            else:
                run_name_equal_params = f"{run_name}_equal_params"
                run_dir_equal_params = base_path / run_name_equal_params

                if run_dir_equal_params.exists():
                    run_dir = run_dir_equal_params
                    run_name = run_name_equal_params
                else:
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
    Dynamically adjusts to handle any number of models.

    Args:
        all_results: Results from load_all_experiments()
        metric_key: Key for the metric to figs
        title: Plot title
        ylabel: Y-axis label
    """
    models = list(all_results.keys())
    fractions = sorted(set(f for model_results in all_results.values()
                          for f in model_results.keys()))

    n_models = len(models)

    # Dynamic figure sizing
    fig_width = max(12, len(fractions) * 2 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x = np.arange(len(fractions))

    # Dynamically calculate bar width
    total_width = 0.8
    width = total_width / n_models if n_models > 0 else 0.8

    # Dynamic color palette
    if n_models <= 3:
        color_map = {'s4': '#1f77b4', 'lmu': '#ff7f0e', 'mamba': '#2ca02c'}
        colors = [color_map.get(model.lower(), plt.cm.tab10(i)) for i, model in enumerate(models)]
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, model in enumerate(models):
        values = []
        for frac in fractions:
            if frac in all_results[model] and all_results[model][frac]:
                val = all_results[model][frac].get(metric_key, np.nan)
                values.append(val if val is not None else np.nan)
            else:
                values.append(np.nan)

        offset = width * (i - n_models/2 + 0.5)
        bars = ax.bar(x + offset, values, width,
                     label=model.upper(),
                     alpha=0.8,
                     color=colors[i])

        # Add value labels on bars with dynamic font size
        font_size = max(7, 9 - n_models // 2)
        for j, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=font_size)

    ax.set_xlabel('Data Fraction', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(f*100)}%' for f in fractions])

    # Adjust legend
    if n_models <= 4:
        ax.legend(fontsize=11, loc='best')
    else:
        ax.legend(fontsize=9, loc='best', ncol=min(3, (n_models + 2) // 3))

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
    save_path: str = None,
):
    """
    Create bar chart comparing test metrics across models and fractions.
    Dynamically adjusts to handle any number of models.

    Args:
        all_results: Results from load_all_experiments()
        metric_key: Key for the test metric to figs
        title: Plot title
        ylabel: Y-axis label
        save_path: Optional path to save the plot as PNG
    """
    models = list(all_results.keys())
    fractions = sorted(set(f for model_results in all_results.values()
                          for f in model_results.keys()))

    n_models = len(models)

    # Dynamic figure sizing based on number of fractions
    fig_width = max(12, len(fractions) * 2 + 4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x = np.arange(len(fractions))

    # Dynamically calculate bar width based on number of models
    # Ensure bars don't overlap and leave space between groups
    total_width = 0.8  # Total width for all bars in a group
    width = total_width / n_models if n_models > 0 else 0.8

    # Use a color palette that scales with number of models
    if n_models <= 3:
        # Use predefined colors for common cases
        color_map = {'s4': '#1f77b4', 'lmu': '#ff7f0e', 'mamba': '#2ca02c'}
        colors = [color_map.get(model.lower(), plt.cm.tab10(i)) for i, model in enumerate(models)]
    else:
        # Use colormap for many models
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, model in enumerate(models):
        values = []
        for frac in fractions:
            if frac in all_results[model] and all_results[model][frac]:
                val = all_results[model][frac].get(metric_key, np.nan)
                values.append(val if val is not None else np.nan)
            else:
                values.append(np.nan)

        # Center the bars around each x position
        offset = width * (i - n_models/2 + 0.5)
        bars = ax.bar(x + offset, values, width,
                     label=model.upper(),
                     alpha=0.8,
                     color=colors[i])

        # Add value labels on bars
        # Adjust font size based on number of models
        font_size = max(7, 9 - n_models // 2)
        for j, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=font_size, fontweight='bold')

    ax.set_xlabel('Data Fraction', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(f*100)}%' for f in fractions])

    # Adjust legend based on number of models
    if n_models <= 4:
        ax.legend(fontsize=11, loc='best')
    else:
        # Use smaller font and multiple columns for many models
        ax.legend(fontsize=9, loc='best', ncol=min(3, (n_models + 2) // 3))

    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_test_data_efficiency(
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
    metric_key: str = 'test_mae',
    ylabel: str = 'Test MAE (bpm)',
    save_path: str = None,
):
    """
    Plot test metric vs data fraction to show data efficiency on test set.
    Dynamically adjusts to handle any number of models.

    Args:
        all_results: Results from load_all_experiments()
        metric_key: Test metric to plot
        ylabel: Y-axis label
        save_path: Optional path to save the plot as PNG
    """
    models = list(all_results.keys())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extended color and marker palettes
    color_map = {'s4': '#1f77b4', 'lmu': '#ff7f0e', 'mamba': '#2ca02c'}
    marker_map = {'s4': 'o', 'lmu': 's', 'mamba': '^'}

    # Generate colors and markers for all models
    if n_models <= 3:
        colors = [color_map.get(m.lower(), plt.cm.tab10(i)) for i, m in enumerate(models)]
        markers = [marker_map.get(m.lower(), ['o', 's', '^', 'D', 'v', '*'][i % 6]) for i, m in enumerate(models)]
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))
        markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', '<', '>']
        markers = [markers[i % len(markers)] for i in range(n_models)]

    for i, model in enumerate(models):
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
                   marker=markers[i],
                   markersize=10,
                   linewidth=2.5,
                   color=colors[i],
                   alpha=0.8)

    ax.set_xlabel('Training Data (%)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Test Set Performance: Data Efficiency Comparison',
                 fontsize=14, fontweight='bold')

    # Adjust legend based on number of models
    if n_models <= 4:
        ax.legend(fontsize=11, loc='best')
    else:
        ax.legend(fontsize=9, loc='best', ncol=min(2, (n_models + 1) // 2))

    ax.grid(True, alpha=0.3)
    ax.set_xticks([10, 25, 50, 100])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

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


def plot_performance_heatmap(
    all_results: Dict[str, Dict[float, Dict[str, Any]]],
    metric_key: str = 'accuracy',
    title: str = 'Performance Heatmap',
    cmap: str = 'RdYlGn',
    annot_format: str = '.4f',
    vmin: float = None,
    vmax: float = None,
    save_path: str = None,
):
    """
    Create a heatmap showing metric values across models and data fractions.

    Args:
        all_results: Results from load_all_experiments()
        metric_key: Metric to visualize (e.g., 'test_f1_micro', 'test_mae', 'accuracy')
        title: Plot title
        cmap: Colormap name (default: 'RdYlGn' for green=good)
        annot_format: Format string for annotations (e.g., '.4f', '.2f', '.2%')
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        save_path: Optional path to save the plot as PNG
    """
    models = sorted(all_results.keys())
    fractions = sorted(set(f for model_results in all_results.values()
                          for f in model_results.keys()))

    # Create matrix for heatmap
    data = []
    for model in models:
        row = []
        for frac in fractions:
            if frac in all_results[model] and all_results[model][frac]:
                val = all_results[model][frac].get(metric_key, np.nan)
                row.append(val if val is not None else np.nan)
            else:
                row.append(np.nan)
        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(
        data,
        index=[m.upper() for m in models],
        columns=[f'{int(f*100)}%' for f in fractions]
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 4))

    # Reverse colormap if lower is better (e.g., for loss, mae, mse)
    reverse_metrics = ['loss', 'mae', 'mse', 'rmse', 'test_loss', 'test_mae', 'test_mse', 'test_rmse']
    if any(metric in metric_key.lower() for metric in reverse_metrics):
        cmap = cmap + '_r'

    sns.heatmap(
        df,
        annot=True,
        fmt=annot_format,
        cmap=cmap,
        cbar_kws={'label': metric_key.replace('_', ' ').title()},
        linewidths=0.5,
        linecolor='gray',
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    ax.set_xlabel('Data Fraction', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

