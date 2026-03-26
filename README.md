# LMU_S4: Sequential Model Comparative Analysis

Comprehensive evaluation framework comparing **LMU**, **S4**, and **Mamba** sequence models across 12 diverse datasets with sample efficiency analysis (10%-100% training data fractions).

## Overview

This project implements and evaluates three state-of-the-art sequence modeling architectures:
- **LMU** (Legendre Memory Unit) - Global temporal accumulation
- **S4** (Structured State Space) - Structured long-range modeling
- **Mamba** - Selective attention for efficient long-range dependencies

**Key Result**: Model effectiveness is strongly shaped by dataset characteristics. S4 dominates vision/audio/NLP tasks, LMU excels at sequential pixels and ECG, MAMBA shows superior scaling on synthetic data.

## Datasets Covered

| Category | Datasets | Metrics |
|----------|----------|---------|
| **Vision** | CIFAR-10 (Sequential), SMNIST, PS-MNIST | Accuracy |
| **Audio** | ESC-50 | Accuracy |
| **Medical** | PTB-XL (ECG), PPG (Heart Rate) | F1-Micro, MAE |
| **NLP** | QQP (Question Pairs) | Accuracy |
| **Long-Range** | ListOps (Synthetic) | Accuracy |
| **Time Series** | ETTh1, ETTh2, ETTm1, ETTm2 (Energy) | MAE |

**Total**: 12 datasets × 3 models × 4 training fractions = 129+ experiments

## Quick Start

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB+ recommended)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd LMU_S4

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Train a single model on a dataset
python main.py --dataset cifar10 --model s4 --fraction 100

# Full evaluation pipeline
python main.py --all-experiments

# View results
python -m src.eval.report  # Generate comparison report
```

## Project Structure

```
LMU_S4/
├── src/
│   ├── datasets/              # Dataset implementations (12 datasets)
│   │   ├── CIFAR10/
│   │   ├── ESC50/
│   │   ├── ETTS/              # Time series (ETTh1, ETTh2, ETTm1, ETTm2)
│   │   ├── ListOps/
│   │   ├── PPG/
│   │   ├── PS-MNIST/
│   │   ├── PTB-XL/
│   │   ├── QQP/
│   │   └── SMNIST/
│   ├── models/                # Model architectures
│   │   ├── lmu.py
│   │   ├── s4/                # S4 implementation
│   │   │   ├── krylov.py
│   │   │   └── s4_block.py
│   │   └── v2/                # Unified model interface
│   │       ├── base.py
│   │       ├── classifier.py
│   │       ├── lmu_block.py
│   │       ├── mamba_block.py
│   │       └── s4_block.py
│   ├── eval/                  # Evaluation utilities
│   │   ├── metrics.py
│   │   ├── infer.py
│   │   ├── report.py
│   │   └── compare.py
│   ├── train_utils/           # Training infrastructure
│   │   ├── loops.py
│   │   └── trainer.py
│   ├── utils/                 # Common utilities
│   │   ├── checkpoint.py
│   │   ├── logging.py
│   │   ├── visualization.py
│   │   └── block_factory.py
│   └── types/
│       └── task_protocol.py
├── notebooks/                 # Jupyter experiments (by dataset)
│   ├── cifar10/
│   ├── esc50/
│   ├── etts/
│   ├── listops/
│   ├── ppg/
│   ├── psmnist/
│   ├── ptbxl/
│   ├── qqp/
│   ├── smnist/
│   └── research_outputs/
├── tests/                     # Unit tests
├── main.py                    # Entry point
├── pyproject.toml            # Project metadata
├── requirements.txt          # Dependencies
├── requirements-dev.txt      # Development dependencies
├── RESULTS.md                # Comprehensive evaluation results
└── README.md                 # This file
```

## Results & Analysis

### Main Results Document
See **[RESULTS.md](RESULTS.md)** for comprehensive cross-dataset analysis including:
- Complete performance tables (10%, 25%, 50%, 100% data fractions)
- **Δ (Delta) framework** for comparative model strength analysis
- Sample efficiency analysis (10%→100% improvement percentages)
- Task-dependent strengths by domain
- Model recommendations for different scenarios

### Key Findings

#### Model Dominance (100% Training Data)
- **S4**: 6 dataset wins (Vision, Audio, NLP, Regression)
  - CIFAR-10: 74.21% (best sequential vision)
  - ESC-50: 56.00% (best audio)
  - PPG: 4.814 MAE (best regression)
  
- **LMU**: 4 dataset wins (Sequential pixels, ECG)
  - SMNIST: 98.70% (sequential MNIST)
  - PTB-XL: 85.89% (ECG classification)
  
- **Mamba**: 1 dataset win (Synthetic long-range)
  - ListOps: 42.85% (structured dependencies)

#### Sample Efficiency Insights
- **Data-Efficient**: SMNIST LMU (+7.89% improvement 10%→100%)
- **Data-Hungry**: ESC-50 S4 (+285.6% improvement 10%→100%)
- **Unusual**: PPG S4 (-111.5% reversal: worst at 10%, best at 100%)

#### Surprising Findings
1. **PTB-XL LMU Dominance**: 13% absolute advantage (85.89% vs 72.83%)
2. **ETTS Non-Monotonic**: Some models perform better at 50% than 100%
3. **MAMBA Scaling**: Worst at 10% (17.8%), best at 100% (42.85%)
4. **S4 Audio**: Wins all ESC-50 fractions (no crossover behavior)

## Experimental Setup

### Model Configurations
- **d_model**: 128 (embedding dimension)
- **depth**: 4 (number of layers)
- **S4-specific**: d_state=32, channels vary by dataset
- **LMU-specific**: memory_size varies by task
- **Mamba-specific**: d_state/expand ratios optimized per dataset

### Training Parameters
- **Optimizer**: AdamW
- **Learning Rate**: Task-dependent (1e-3 to 1e-4)
- **Batch Size**: Dataset-dependent (16-256)
- **Epochs**: Until convergence (100-500)
- **Validation**: 20% hold-out, early stopping enabled

### Data Fractions
Models evaluated at 10%, 25%, 50%, and 100% training data to assess sample efficiency and scaling behavior.

## Evaluation Metrics

| Task Type | Primary Metric | Secondary |
|-----------|---|---|
| Classification | Accuracy | Loss, F1 (multi-label) |
| Regression | MAE | MSE, RMSE |
| Time Series | MAE | MSE, RMSE |

### Training a Custom Model
```bash
python main.py \
  --dataset cifar10 \
  --model s4 \
  --fraction 50 \
  --d-model 128 \
  --depth 4 \
  --epochs 200 \
  --save-dir ./checkpoints
```

### Evaluating Checkpoints
```bash
python -c "
from src.utils.checkpoint import load_trainer_from_checkpoint
from src.notebooks.cifar10.cifar10_task import CIFAR10Task

trainer = load_trainer_from_checkpoint(
    checkpoint_path='./checkpoints/best.pt',
    task=CIFAR10Task()
)
results = trainer.evaluate()
print(results)
"
```

### Notebooks
Training and evaluation notebooks in `src/notebooks/{dataset}/`:
- `{dataset}_{model}_task.py` - Configuration and setup
- `{dataset}_{model}_task.ipynb` - Interactive experiment (Jupyter)
- `runs/{experiment_name}/` - Results and checkpoints

## Related Work

### State-of-the-art Sequence Models
- **LMU**: [Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf)
- **S4**: [Paper](https://arxiv.org/pdf/2111.00396)
- **Mamba**: [Paper](https://arxiv.org/pdf/2312.00752)

### Datasets
- CIFAR-10: [Link](https://www.cs.toronto.edu/~kriz/cifar.html)
- ESC-50: [Link](https://github.com/karolpiczak/ESC-50)
- PTB-XL: [Link](https://physionet.org/content/ptb-xl/1.0.1/)
- Long Range Arena: [Link](https://arxiv.org/pdf/2011.04006)

- ...
---