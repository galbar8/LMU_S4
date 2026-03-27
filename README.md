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
- **S4**: 6 dataset wins
  - CIFAR-10: 0.7421 (74.21% accuracy)
  - ESC-50: 0.5600 (56.00% accuracy)
  - PPG: 4.814 MAE (heart rate bpm)
  - QQP: 0.8071 (80.71% accuracy)
  - ETTm1: 0.1656 MAE
  - ListOps: 0.4235 (42.35% accuracy, tied)
  
- **LMU**: 4 dataset wins
  - SMNIST: 0.9870 (98.70% accuracy)
  - PTB-XL: 0.8589 (85.89% F1-Micro)
  - PS-MNIST: 0.9624 (96.24% accuracy, small-data dominant)
  - ETTh2: 0.4060 MAE
  
- **Mamba**: 1 dataset win
  - ListOps: 0.4285 (42.85% accuracy)

#### Sample Efficiency Insights
- **Data-Efficient**: SMNIST LMU (0.9149→0.9870, +7.89%)
- **Data-Hungry**: ESC-50 S4 (0.1450→0.5600, +285.6%)
- **Unusual**: PPG S4 (10.149→4.814 MAE, severe reversal from worst to best)

#### Surprising Findings
1. **PTB-XL LMU Dominance**: LMU (0.8589, 85.89%) vs S4 (0.7283, 72.83%) - 13.1% absolute difference
2. **PPG S4 Paradox**: S4 worst at 10% (10.149 MAE) but best at 100% (4.814 MAE) - extreme reversal
3. **ETTS Non-Monotonic Scaling**: Some models perform better at 50% than at 100% data
4. **Mamba ListOps**: Improves from 17.8% (10%) to 42.85% (100%) - +140% improvement
5. **S4 Audio Consistency**: Wins all ESC-50 fractions (10%, 25%, 50%, 100%) with no crossover

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