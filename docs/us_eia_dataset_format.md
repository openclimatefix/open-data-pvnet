# US EIA Dataset Technical Documentation

## Overview

We collect hourly solar generation data for the United States from the **US Energy Information Administration (EIA) Open Data API**. This dataset serves as the primary ground truth for training solar forecasting models for the US region.

Key characteristics:
- **Source**: [EIA Hourly Electricity Grid Monitor](https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48)
- **Granularity**: Hourly resolution
- **Coverage**: Major US Balancing Authorities (ISOs/RTOs)
- **License**: Public Domain (US Government Data)

---

## Data Formats

### 1. Raw Data (Intermediate)

The data collected by `collect_eia_data.py` is stored in Zarr format with the following structure:

- **Dimensions**: `(timestamp, ba_code)`
- **Variables**:
  - `generation_mw`: Electricity generation in Megawatts (MW)
  - `ba_name`: Full name of the Balancing Authority
  - `latitude`: Approximate centroid latitude
  - `longitude`: Approximate centroid longitude
  - `value-units`: Unit string (e.g., "megawatthours")

### 2. Processed Data (Ready for Training)

The raw data is preprocessed by `preprocess_eia_for_sampler.py` to match the format required by `ocf-data-sampler`. This format aligns with the UK GSP dataset structure.

- **Dimensions**: `(ba_id, datetime_gmt)`
- **Chunking**: `{"ba_id": 1, "datetime_gmt": 1000}`
- **Variables**:

| Variable | Type | Description |
|----------|------|-------------|
| `generation_mw` | `float32` | Solar generation in MW |
| `capacity_mwp` | `float32` | Estimated installed capacity in MWp |

- **Coordinates**:

| Coordinate | Type | Description |
|------------|------|-------------|
| `ba_id` | `int64` | Numeric ID mapped to each BA code |
| `datetime_gmt` | `datetime64[ns]` | Timestamp in UTC |
| `ba_code` | `string` | ISO/RTO code (e.g., "CISO") |
| `ba_name` | `string` | Full name of the BA |
| `latitude` | `float32` | Centroid latitude |
| `longitude` | `float32` | Centroid longitude |

---

## Metadata & Mapping

A metadata CSV file (`us_ba_metadata.csv`) is generated alongside the processed data. It maps numeric `ba_id`s to their corresponding codes and locations.

| ba_id | ba_code | ba_name | latitude | longitude |
|-------|---------|---------|----------|-----------|
| 0 | CISO | California ISO | 37.0 | -120.0 |
| 1 | ERCO | Electric Reliability Council of Texas | 31.0 | -99.0 |
| ... | ... | ... | ... | ... |

---

## Capacity Estimation

Unlike UK PVLive, the EIA dataset does not provide historical installed capacity. We estimate capacity using a heuristic based on maximum historical generation:

```python
capacity = max(generation_mw) * 1.15
min_capacity = 100.0 MW
```

- **Method**: `estimate` (Default)
- **Safety Factor**: 1.15 (Assumes max generation is ~85% of theoretical capacity due to efficiencies/weather)
- **Minimum**: 100 MW floor to prevent zeros for missing data intervals

---

## Data Quality & Validation

- **Missing Data**: Intervals with missing data are typically represented as NaNs. The `ocf-data-sampler` handles this by finding valid contiguous time periods.
- **Timezone**: All timestamps are converted to **UTC**.
- **Negative Generation**: Clipped to 0.

## Usage

### Loading Data with Xarray

```python
import xarray as xr
import s3fs

# Local Load
ds = xr.open_zarr("src/open_data_pvnet/data/target_eia_data_processed.zarr", consolidated=True)

# S3 Load (Public)
s3 = s3fs.S3FileSystem(anon=True)
ds = xr.open_zarr(s3.get_mapper("s3://ocf-open-data-pvnet/data/us/eia/latest/target_eia_data_processed.zarr"), consolidated=True)
```
