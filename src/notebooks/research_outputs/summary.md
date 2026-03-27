# Comparative Analysis: LMU vs S4 - Summary Report

**Generated**: 2026-03-27 15:41:08

---

## Dataset Coverage

| Dataset | LMU | S4 | Total | Fractions |
|---------|-|-|-------|----------|----------|
| cifar10 | 4 | 4 | 8 | 10%, 25%, 50%, 100% |
| esc50 | 4 | 4 | 8 | 10%, 25%, 50%, 100% |
| etts | 16 | 16 | 32 | 10%, 25%, 50%, 100% |
| listops | 4 | 4 | 8 | 10%, 25%, 50%, 100% |
| ppg | 4 | 4 | 8 | 10%, 25%, 50%, 100% |
| psmnist | 4 | 4 | 8 | 10%, 25%, 50%, 100% |
| ptbxl | 4 | 4 | 8 | 10%, 25%, 50%, 100% |
| qqp | 4 | 4 | 8 | 10%, 25%, 50%, 100% |
| smnist | 4 | 4 | 8 | 10%, 25%, 50%, 100% |

**Total**: 96 experiments across 9 datasets

---

## Pairwise Win Rates

| Comparison | First Model Wins | Total | Win Rate |
|------------|------------------|-------|----------|
| LMU vs S4 | 23 | 48 | 47.9% |

## Best Model Frequency

| Model | Times Best | Total | Frequency |
|-------|------------|-------|----------|
| LMU | 23 | 48 | 47.9% |
| S4 | 25 | 48 | 52.1% |

---

## Top-5 LMU Wins over S4 (by absolute Δ)

| Rank | Dataset | Fraction | Δ | Value (Model1) | Value (Model2) |
|------|---------|----------|---|----------------|----------------|
| 58 | ppg | 25% | +5.2206 | 6.3911 | 11.6117 |
| 56 | ppg | 10% | +3.3962 | 6.7529 | 10.1492 |
| 60 | ppg | 50% | +0.4440 | 6.5306 | 6.9746 |
| 88 | smnist | 10% | +0.1966 | 0.9149 | 0.7183 |
| 64 | psmnist | 10% | +0.1743 | 0.9154 | 0.7411 |

---

## Performance by Task Type


### LMU vs S4

| Task Type | Mean Δ @ 10% | Mean Δ @ 100% | Change | Count |
|-----------|--------------|---------------|--------|-------|
| binary | +0.0180 | -0.0201 | -0.0381 | 4 |
| forecasting | -0.0023 | +0.0002 | +0.0025 | 16 |
| multiclass | +0.0610 | -0.0397 | -0.1007 | 20 |
| multilabel | -0.0193 | +0.0172 | +0.0366 | 4 |
| regression | +3.3962 | -1.9557 | -5.3519 | 4 |

---

## Overall Statistics

- **Total Experiments**: 96
- **Total Pairwise Comparisons**: 48
- **Datasets**: 9
- **Models**: lmu, s4

---

*Δ > 0 indicates first model performs better; Δ < 0 indicates second model performs better*
