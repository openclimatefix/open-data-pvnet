# India Solar Data Pipeline for PVNet

This contribution adds support for **India solar generation data** to the open-data-pvnet project.

## Data Source

**Mendeley Dataset**: [DOI 10.17632/y58jknpgs8.2](https://data.mendeley.com/datasets/y58jknpgs8/2)
- 29 monthly Excel files (Sep 2021 - Jun 2025)
- 5-minute resolution solar/wind generation data
- Covers all 5 Indian regional grids (NR, WR, SR, ER, NER)

## Files Added

### Configuration Files
| File | Description |
|------|-------------|
| `configs/india_pv_data_config.yaml` | India solar data settings |
| `configs/india_gfs_config.yaml` | GFS NWP config for India region |
| `configs/india_regions.csv` | 5 regional grid metadata |
| `configs/PVNet_configs/datamodule/configuration/india_configuration.yaml` | Complete PVNet config |

### Scripts
| File | Description |
|------|-------------|
| `scripts/download_mendeley_india.py` | Dataset download instructions |
| `scripts/process_india_data.py` | Excel → Zarr conversion |
| `scripts/test_india_pipeline.py` | Pipeline validation tests |
| `scripts/train_india_baseline.py` | Solar-only baseline model |

## Data Processing Results

| Metric | Value |
|--------|-------|
| **Rows** | 5,184 hourly |
| **Date Range** | Jan 1, 2024 → Jun 30, 2025 |
| **Mean Solar** | 15,899 MW |
| **Max Solar** | 64,701 MW |

## Baseline Model Results

A simple temporal model (hour, month, lag features) achieves:
- **RMSE**: 8,270 MW
- **MAE/Mean**: ~52%

## Known Limitations

1. **2021-2023 data**: Uses SCADA codes as column headers - requires manual mapping
2. **NWP coverage**: OCF's GFS S3 data only covers UK region. India NWP needs NOAA GFS processing.

## Next Steps

1. Process NOAA GFS for India (68-98°E, 6-38°N)
2. Add 2021-2023 data with SCADA code mapping
3. Integrate with full PVNet model architecture

## Related Issue

Closes #121 (India contribution)

---

*Contribution by Siddhant Jain ([@Raakshass](https://github.com/Raakshass)) for GSoC 2026*
