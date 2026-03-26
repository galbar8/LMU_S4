# Comparative Analysis: LMU vs S4 vs MAMBA - Summary Report

**Generated**: 2026-02-10 23:20:09

---

## Dataset Coverage

| Dataset | LMU | S4 | MAMBA | Total | Fractions |
|---------|-|-|-|-------|----------|----------|
| cifar10 | 4 | 4 | 4 | 12 | 10%, 25%, 50%, 100% |
| listops | 4 | 4 | 4 | 12 | 10%, 25%, 50%, 100% |
| ppg | 4 | 4 | 4 | 12 | 10%, 25%, 50%, 100% |
| psmnist | 4 | 4 | 4 | 12 | 10%, 25%, 50%, 100% |
| qqp | 4 | 4 | 4 | 12 | 10%, 25%, 50%, 100% |

**Total**: 60 experiments across 5 datasets

---

## Pairwise Win Rates

| Comparison | First Model Wins | Total | Win Rate |
|------------|------------------|-------|----------|
| LMU vs MAMBA | 16 | 20 | 80.0% |
| LMU vs S4 | 9 | 20 | 45.0% |
| MAMBA vs S4 | 2 | 20 | 10.0% |

## Best Model Frequency

| Model | Times Best | Total | Frequency |
|-------|------------|-------|----------|
| LMU | 9 | 20 | 45.0% |
| S4 | 10 | 20 | 50.0% |
| MAMBA | 1 | 20 | 5.0% |

---

## Top-5 LMU Wins over MAMBA (by absolute Δ)

| Rank | Dataset | Fraction | Δ | Value (Model1) | Value (Model2) |
|------|---------|----------|---|----------------|----------------|
| 37 | ppg | 25% | +4.7013 | 6.3911 | 11.0925 |
| 41 | ppg | 50% | +4.4734 | 6.5306 | 11.0040 |
| 33 | ppg | 10% | +3.4864 | 6.7529 | 10.2394 |
| 45 | ppg | 100% | +2.2054 | 6.7700 | 8.9754 |
| 49 | psmnist | 10% | +0.6379 | 0.9154 | 0.2775 |

---

## Top-5 LMU Wins over S4 (by absolute Δ)

| Rank | Dataset | Fraction | Δ | Value (Model1) | Value (Model2) |
|------|---------|----------|---|----------------|----------------|
| 36 | ppg | 25% | +5.2206 | 6.3911 | 11.6117 |
| 32 | ppg | 10% | +3.3962 | 6.7529 | 10.1492 |
| 40 | ppg | 50% | +0.4440 | 6.5306 | 6.9746 |
| 48 | psmnist | 10% | +0.1743 | 0.9154 | 0.7411 |
| 52 | psmnist | 25% | +0.0656 | 0.9414 | 0.8758 |

---

## Top-5 MAMBA Wins over S4 (by absolute Δ)

| Rank | Dataset | Fraction | Δ | Value (Model1) | Value (Model2) |
|------|---------|----------|---|----------------|----------------|
| 38 | ppg | 25% | +0.5193 | 11.0925 | 11.6117 |
| 30 | listops | 100% | +0.0050 | 0.4285 | 0.4235 |

---

## Performance by Task Type


### LMU vs MAMBA

| Task Type | Mean Δ @ 10% | Mean Δ @ 100% | Change | Count |
|-----------|--------------|---------------|--------|-------|
| binary | +0.6019 | +0.0106 | -0.5913 | 4 |
| multiclass | +0.2841 | -0.0018 | -0.2859 | 12 |
| regression | +3.4864 | +2.2054 | -1.2811 | 4 |

### LMU vs S4

| Task Type | Mean Δ @ 10% | Mean Δ @ 100% | Change | Count |
|-----------|--------------|---------------|--------|-------|
| binary | +0.0180 | -0.0201 | -0.0381 | 4 |
| multiclass | +0.0436 | -0.0455 | -0.0891 | 12 |
| regression | +3.3962 | -1.9557 | -5.3519 | 4 |

### MAMBA vs S4

| Task Type | Mean Δ @ 10% | Mean Δ @ 100% | Change | Count |
|-----------|--------------|---------------|--------|-------|
| binary | -0.5839 | -0.0307 | +0.5532 | 4 |
| multiclass | -0.2405 | -0.0436 | +0.1968 | 12 |
| regression | -0.0902 | -4.1611 | -4.0708 | 4 |

---

## Overall Statistics

- **Total Experiments**: 60
- **Total Pairwise Comparisons**: 60
- **Datasets**: 5
- **Models**: lmu, s4, mamba

---

*Δ > 0 indicates first model performs better; Δ < 0 indicates second model performs better*
