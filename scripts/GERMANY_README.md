# Germany Solar Forecasting Pipeline

A complete pipeline for training solar forecasting models for Germany using GFS weather data and SMARD PV generation data.

## Quick Start

### 1. Download Data

```bash
# Download PV generation data
python src/open_data_pvnet/scripts/downloading_pv_germany.py

# Download GFS weather data (recent data)
python src/open_data_pvnet/scripts/download_gfs_germany_fast.py --year 2026 --months 2 --max-days 1

# For historical data, install herbie-data first
pip install herbie-data
python src/open_data_pvnet/scripts/download_gfs_germany_fast.py --year 2024 --months 1 --max-days 1 --source herbie
```

### 2. Process Data

```bash
python src/open_data_pvnet/scripts/germany_pipeline.py process \
  --pv-zarr ./data/germany/generation/germany_pv_2021.zarr \
  --gfs-zarr ./data/germany/gfs/zarr/germany_gfs_2021_01.zarr \
  --output-dir ./data/germany/processed
```

### 3. Run Tests

```bash
python src/open_data_pvnet/scripts/germany_pipeline.py test \
  --pv-zarr ./data/germany/generation/germany_pv_2021.zarr \
  --gfs-zarr ./data/germany/gfs/zarr/germany_gfs_2021_01.zarr
```

### 4. Train Model

```bash
python src/open_data_pvnet/scripts/train_germany_baseline.py \
  --epochs 10 \
  --output-dir ./models/germany
```

## Pipeline Commands

### Inspect Zarr Files

```bash
python src/open_data_pvnet/scripts/germany_pipeline.py inspect \
  --zarr ./data/germany/gfs/zarr/germany_gfs_2021_01.zarr
```

### Process Data

Validates, aligns, and calculates normalization constants for PV and GFS data.

```bash
python src/open_data_pvnet/scripts/germany_pipeline.py process \
  --pv-zarr <path> \
  --gfs-zarr <path> \
  --output-dir <output_directory>
```

**Outputs:**
- `normalization_constants.yaml` - Normalization statistics
- `processing_report.txt` - Data processing summary

### Run Tests

Validates data loading, temporal alignment, and data quality.

```bash
python src/open_data_pvnet/scripts/germany_pipeline.py test \
  --pv-zarr <path> \
  --gfs-zarr <path> \
  --output-dir <output_directory>
```

**Outputs:**
- `test_report.txt` - Test results

## Data Formats

### PV Data
- **Dimensions:** (datetime_gmt, gsp_id)
- **Variables:** generation_mw, capacity_mwp, installedcapacity_mwp
- **Coordinates:** datetime_gmt, gsp_id

### GFS Data
- **Dimensions:** (init_time_utc, step, latitude, longitude)
- **Variables:** 14 weather channels (dlwrf, dswrf, hcc, mcc, lcc, prate, r, t, tcc, u10, u100, v10, v100, vis)
- **Resolution:** 0.25° (~25km)

## Configuration

- `src/open_data_pvnet/configs/germany_gfs_config.yaml` - GFS download settings
- `src/open_data_pvnet/configs/germany_pv_data_config.yaml` - PV data settings
- `src/open_data_pvnet/configs/germany_regions.csv` - Regional boundaries
- `src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/germany_configuration.yaml` - Model config

## Scripts

| Script | Purpose |
|--------|---------|
| `downloading_pv_germany.py` | Download PV generation data from SMARD API |
| `download_gfs_germany_fast.py` | Download GFS weather data with subregion filtering |
| `germany_pipeline.py` | Main pipeline orchestrator (inspect, process, test) |
| `germany_utils.py` | Shared utility functions |
| `train_germany_baseline.py` | Train baseline forecasting model |

## Troubleshooting

### GFS Download Issues
- **403 Forbidden:** Historical data (>10 days old) requires S3 access. Use recent dates or install `herbie-data`.
- **Connection timeouts:** Check internet connection and NOAA server availability.

### Data Processing Issues
- Ensure Zarr files exist at specified paths
- Check temporal overlap in processing report
- Verify configuration paths are correct

### Test Failures
- Review test report at `./data/germany/tests/test_report.txt`
- Check data loading and temporal alignment
- Verify data quality and completeness

## Requirements

- Python 3.9+
- xarray, zarr, requests, pyyaml, tqdm, cfgrib, torch, pandas, numpy

## Data Sources

- **PV Generation:** [SMARD API](https://www.smard.de/) (Bundesnetzagentur)
- **Weather Data:** [GFS](https://www.ncei.noaa.gov/products/weather-global-forecast-system) (NOAA)
- **Region:** Germany (47-55°N, 5-15°E)
