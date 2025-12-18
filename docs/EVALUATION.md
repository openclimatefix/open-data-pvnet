# Evaluating PVNet

This guide provides comprehensive instructions for evaluating trained PVNet models using the `open-data-pvnet` evaluation pipeline.

## 1. Prerequisites

Ensure your environment meets the following requirements:

- **Python**: Version 3.10 or higher
- **Dependencies**: Installed via `pip install -e .` (includes matplotlib, seaborn)
- **Trained Checkpoint**: A Lightning `.ckpt` file from a completed training run
- **Data Configuration**: The same `data_configuration.yaml` used during training

## 2. Quick Start

```bash
python -m open_data_pvnet.evaluate_pipeline \
    --checkpoint path/to/epoch=0-step=100.ckpt \
    --data-config data_configuration.yaml \
    --output-dir ./eval_output \
    --test-start 2023-10-01 \
    --test-end 2023-12-31
```

## 3. Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint, -c` | Required | Path to Lightning checkpoint (.ckpt) |
| `--data-config, -d` | Required | Path to ocf-data-sampler config YAML |
| `--output-dir, -o` | `./eval_output` | Directory for results |
| `--test-start` | `2023-10-01` | Test period start (YYYY-MM-DD) |
| `--test-end` | `2023-12-31` | Test period end (YYYY-MM-DD) |
| `--batch-size, -b` | `32` | Batch size for evaluation |
| `--limit-batches` | None | Limit batches for quick testing |
| `--device` | `cpu` | Device: 'cpu' or 'cuda' |
| `--quantiles` | `0.02,0.1,0.25,0.5,0.75,0.9,0.98` | Quantile values |
| `--wandb/--no-wandb` | `--no-wandb` | Enable W&B logging |
| `--seed` | `42` | Random seed |

## 4. Output Files

After evaluation, results are saved to a timestamped subdirectory:

```
eval_output/
└── 20231218_143928/
    ├── metrics_summary.csv      # Overall metrics
    ├── horizon_metrics.csv      # Per-horizon breakdown
    ├── config_snapshot.yaml     # Reproducibility config
    └── plots/
        ├── mae_vs_horizon.png   # Error vs lead time
        ├── scatter.png          # Predicted vs actual
        ├── reliability_diagram.png
        └── coverage.png
```

## 5. Understanding Metrics

### 5.1 Point Forecast Metrics

These metrics use the **0.5 quantile (median)** as the point forecast.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | `mean(|y - ŷ|)` | Average error magnitude. Lower values indicate better performance. |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Penalizes large errors more heavily. RMSE ≥ MAE always. |

**Typical Performance Ranges:**

- MAE < 0.10: Excellent performance
- MAE 0.10-0.20: Good performance
- MAE 0.20-0.30: Fair performance
- MAE > 0.30: Requires improvement

### 5.2 Probabilistic Metrics

| Metric | Description | Desired Value |
|--------|-------------|---------------|
| **Pinball Loss** | Asymmetric quantile loss. Measures calibration per quantile. | Lower is better |
| **CRPS** | Continuous Ranked Probability Score. Overall distribution quality. | Lower is better; approximately equal to MAE for deterministic forecasts |
| **Coverage** | Fraction of actuals below predicted quantile. | Should equal quantile value (e.g., 0.90 for q0.90) |

### 5.3 Per-Horizon Metrics

The `horizon_metrics.csv` contains MAE and RMSE for each forecast step:

```csv
horizon_idx,mae,rmse
0,0.053,0.109
1,0.056,0.111
...
15,0.424,0.433
```

**Interpretation:**

- Error typically increases with horizon (forecast lead time)
- Sharp spikes may indicate data quality issues at specific hours
- Flat or decreasing error suggests potential model problems

## 6. Understanding Plots

### 6.1 MAE vs Forecast Horizon

**File:** `plots/mae_vs_horizon.png`

This plot shows how forecast error grows with lead time.

**Key Indicators:**

- **Gradual increase**: Normal skill degradation as forecast lead time increases
- **Sharp spikes**: Possible data quality issues at those specific hours
- **Decreasing trend**: Indicates potential model or data pipeline problems

### 6.2 Scatter Plot (Predicted vs Actual)

**File:** `plots/scatter.png`

Visualizes overall prediction quality through comparison of predicted and actual values.

**Key Indicators:**

- **Points clustered on diagonal**: Indicates good prediction accuracy
- **Systematic offset from diagonal**: Suggests presence of bias
- **Wide horizontal/vertical spread**: Indicates high variance in predictions

**Plot Elements:**

- Red dashed line: Perfect forecast (y = x)
- Green line: Actual trend with slope and intercept

### 6.3 Reliability Diagram

**File:** `plots/reliability_diagram.png`

Displays calibration quality of quantile forecasts.

**Key Indicators:**

- **Points on diagonal**: Well-calibrated forecasts
- **Points above diagonal**: Over-confident predictions (intervals too narrow)
- **Points below diagonal**: Under-confident predictions (intervals too wide)

**Colored Regions:**

- Red area: Under-confident zone
- Blue area: Over-confident zone

### 6.4 Coverage Bar Chart

**File:** `plots/coverage.png`

Compares expected versus observed coverage per quantile.

**Key Indicators:**

- Light bars: Expected coverage (should equal quantile value)
- Dark bars: Observed coverage (actual)
- Large gaps indicate miscalibration at that quantile

## 7. Example Output

```
============================================================
EVALUATION SUMMARY
============================================================
Samples evaluated: 96
MAE: 0.1569
RMSE: 0.2604
CRPS: 0.1568
Pinball (overall): 0.0784
============================================================
Results saved to: eval_output/20251218_143928
============================================================
```

## 8. Common Issues

### Shape Mismatch Error

**Symptom:** `RuntimeError: The size of tensor a (16) must match the size of tensor b (21)`

**Cause:** Model outputs fewer horizons than data provides.

**Solution:** This is now handled automatically by aligning to minimum horizon.

### Empty Test Set

**Symptom:** `RuntimeError: No batches were successfully processed`

**Cause:** Test period has no data in the Zarr file.

**Solution:** Verify that test dates overlap with available data in the dataset.

### CUDA Out of Memory

**Symptom:** `CUDA out of memory`

**Solution:** Use `--device cpu` or reduce `--batch-size`.

## 9. Reproducibility

The pipeline ensures reproducibility through:

1. **Fixed Seed**: Default `--seed 42` controls all randomness
2. **Config Snapshot**: `config_snapshot.yaml` saves all parameters
3. **Timestamped Outputs**: Each run receives a unique directory

To reproduce results:

```bash
# Use same parameters from config_snapshot.yaml
python -m open_data_pvnet.evaluate_pipeline \
    --checkpoint same_checkpoint.ckpt \
    --data-config same_config.yaml \
    --seed 42
```

## 10. Weights & Biases Integration

For offline W&B logging:

```bash
# Enable offline mode
$env:WANDB_MODE = "offline"

# Run with W&B
python -m open_data_pvnet.evaluate_pipeline \
    --checkpoint model.ckpt \
    --data-config config.yaml \
    --wandb

# Sync later when online
wandb sync ./wandb/offline-run-*
```