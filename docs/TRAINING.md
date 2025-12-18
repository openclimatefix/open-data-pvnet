# Training PVNet (Open-Data Pipeline)

This guide describes how to train the PVNet model using the open-data-pvnet training pipeline. It covers environment setup, configuration, execution, and common pitfalls encountered during local and Windows-based runs.

## 1. Prerequisites

Ensure the following requirements are met before running the training pipeline:

**Python:** Version 3.10+

**Dependencies:**
Install all required packages:

```bash
pip install -r requirements.txt
```

**Local PVNet Library:**
The pvnet repository must exist as a sibling directory to open-data-pvnet.

## 2. Expected Directory Structure

Your workspace must follow this layout for module imports to resolve correctly:

```
workspace/
├── open-data-pvnet/        # Training pipeline (this repository)
│   ├── src/
│   ├── configs/
│   └── ...
└── pvnet/                  # Core PVNet model library
```

If pvnet is missing or misplaced, you will encounter:

```
ModuleNotFoundError: No module named 'pvnet'
```

## 3. Configuration Overview

Training is driven by Hydra YAML configurations, split into:

- Pipeline / data configuration
- Model configuration

### 3.1 Data Configuration

The data configuration defines how GSP, NWP, and Satellite data are sampled.

**Default location:**

```
src/open_data_pvnet/configs/pipeline_config.yaml
```

**Key Sections:**

- **general:** Run metadata and experiment naming
- **input_data:**
  - generation (GSP data)
  - nwp (Numerical Weather Prediction)
  - satellite (Satellite imagery)

All paths must point to valid Zarr datasets and be compatible with ocf-data-sampler.

### 3.2 DataLoader Settings (Critical for Stability)

When using Windows, debugging locally, or running single-process training, the following must be enforced under `streamed_batches`:

```yaml
streamed_batches:
  num_workers: 0
  prefetch_factor: null
```

**Why this is required:**

In PyTorch, `prefetch_factor` is only valid when `num_workers > 0`. If `num_workers = 0` and `prefetch_factor` is set, PyTorch raises:

```
ValueError: prefetch_factor option could only be specified in multiprocessing
```

Explicitly setting `prefetch_factor: null` prevents this crash and ensures compatibility with single-process loading.

## 4. Model Configuration Fix (Important)

### include_time in multimodal.yaml

In pvnet's model configuration (typically `multimodal.yaml`), ensure:

```yaml
include_time: false
```

**Explanation:**

When `include_time: true`, the model expects pre-computed time features (e.g., sine/cosine encodings of timestamps) to be present in each batch. The current data pipeline does not generate these features. As a result, the model fails with a missing-key error during the forward pass.

Setting `include_time: false` keeps the model's expectations aligned with the actual batch structure, allowing training to proceed correctly.

## 5. Running the Training Pipeline

Use `train_pipeline.py` to launch training.

### Example Command (PowerShell)

```powershell
# Enable full Hydra error traces
$env:HYDRA_FULL_ERROR=1

# Absolute path to data configuration
$configPath = "C:\path\to\data_configuration.yaml"

# Run training
python src/open_data_pvnet/train_pipeline.py `
    --start-date "2023-01-01" `
    --end-date "2023-01-03" `
    --data-configuration $configPath `
    --val-split 0.2 `
    --test-split 0.2 `
    --num-workers 0 `
    --batch-size 4
```

### Arguments Explained

| Argument | Description |
|----------|-------------|
| `--start-date`, `--end-date` | Total data range (YYYY-MM-DD) |
| `--data-configuration` | Absolute path to sampler YAML |
| `--val-split` | Fraction of data used for validation |
| `--test-split` | Fraction of data used for testing |
| `--num-workers` | DataLoader subprocesses (0 recommended for Windows/debugging) |
| `--batch-size` | Samples per training batch |

## 6. Troubleshooting & Common Issues

### Invalid or Empty Date Splits

Very short date ranges (1–2 days) may not support validation/test splits. This can lead to warnings or empty datasets.

### Path Resolution Errors

Hydra requires absolute paths for external configuration files. Relative paths may fail silently or resolve incorrectly.

### pvnet Import Errors

Ensure the directory structure is correct and pvnet is a sibling folder, not installed separately or nested incorrectly.

### DataLoader Crashes

Verify:

```yaml
num_workers: 0
prefetch_factor: null
```

Any other combination with `num_workers = 0` will fail.

## Summary

- Use absolute paths for all configs
- Keep `num_workers = 0` with `prefetch_factor = null`
- Disable `include_time` unless time features are explicitly generated
- Ensure pvnet is correctly placed in the workspace

This configuration reflects the stable, correct setup for running PVNet training using the open-data pipeline.