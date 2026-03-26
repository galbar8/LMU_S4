import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_forecasting_comparison(model, preds, targets, pred_len, dataset_name, fraction=1.0, num_samples=5, save_path=None):
    if hasattr(preds, 'cpu'):
        preds = preds.cpu().numpy()
    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()

    num_windows = preds.shape[0]

    # Randomly select samples to plot
    np.random.seed(42)
    sample_indices = np.random.choice(num_windows, min(num_samples, num_windows), replace=False)

    # Create figure
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for idx, sample_idx in enumerate(sample_indices):
        ax = axes[idx]

        # Extract predictions and targets for this sample
        pred_signal = preds[sample_idx, :, 0]  # First feature (target variable)
        target_signal = targets[sample_idx, :, 0]

        # Time steps
        time_steps = np.arange(pred_len)

        # Plot both signals
        ax.plot(time_steps, target_signal, 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.plot(time_steps, pred_signal, 'r--', linewidth=2, label='Prediction', alpha=0.8)

        # Calculate error metrics for this sample
        mae = np.mean(np.abs(pred_signal - target_signal))
        mse = np.mean((pred_signal - target_signal) ** 2)
        rmse = np.sqrt(mse)

        # Styling
        ax.set_xlabel('Time Steps', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.set_title(f'Sample {sample_idx} | MAE: {mae:.4f}, RMSE: {rmse:.4f}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Overall title
    frac_str = f"{int(fraction * 100)}%" if fraction < 1.0 else "100%"
    fig.suptitle(f'{dataset_name.upper()} - {model} Forecasting Results ({frac_str} Data)\n'
                 f'Predicted vs Ground Truth (Horizon: {pred_len} steps)',
                 fontsize=14, fontweight='bold', y=1.0)

    plt.tight_layout()

    if save_path is not None:
        frac_suffix = f"_frac{int(fraction * 100)}" if fraction < 1.0 else ""
        filename = f"forecasting_{dataset_name.lower()}{frac_suffix}_comparison.png"
        save_file = Path(save_path) / filename
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_file}")

    plt.show()
