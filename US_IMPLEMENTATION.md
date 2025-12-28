# US Generalisation Implementation for PVNet

This document outlines the changes and approaches used to extend PVNet to support the United States geography.

## Overview
The goal was to enable training, validation, and inference for U.S. regions using GFS weather data and EIA solar generation targets.

## Data Ingestion: EIA Solar Generation
We implemented a pipeline to ingest historical U.S. solar generation time series from the EIA Open Data API.

### Components
- **`src/open_data_pvnet/scripts/fetch_eia_data.py`**: A dedicated `EIAData` class handles interactions with the EIA API (`https://api.eia.gov/v2`).
    - Fetches "hourly" electricity generation data.
    - Filters for fuel type `SUN` (Solar).
    - Supports filtering by Balancing Authority (BA) codes.
    - Handles pagination (5000 records per page) and request timeouts.
- **`src/open_data_pvnet/scripts/collect_eia_data.py`**: A CLI script to execute the data collection.
    - **Default BAs**: Top ISOs/RTOs including CAISO (`CISO`), ERCOT (`ERCO`), PJM (`PJM`), MISO (`MISO`), NYISO (`NYIS`), ISO-NE (`ISNE`), and SPP (`SWPP`).
    - **Geographic Alignment**: Maps BAs to approximate latitude/longitude centroids to align with GFS data.
    - **Output**: Saves the processed data (timestamp, ba_code, generation_mw, lat/lon) to a Zarr dataset (or NetCDF).

## Weather Data: GFS Integration
We extended the GFS processing pipeline to support a US-specific configuration alongside the global one.

### Components
- **`src/open_data_pvnet/nwp/gfs.py`**: Updated to handle region-specific processing.
    - Added `process_gfs_data(..., region="us")` which loads the US configuration.
    - Automates downloading GRIB2 files from NOAA's S3 bucket (`noaa-gfs-bdp-pds`).
    - Converts GRIB2 files to Zarr format using `cfgrib` and `xarray`.
- **`src/open_data_pvnet/configs/gfs_us_data_config.yaml`**: New configuration file for US GFS data.
    - **Resolution**: 3 hours (180 minutes).
    - **Channels**: Selected relevant channels for solar forecasting:
        - `dlwrf`, `dswrf` (Model-calculated radiation)
        - `tcc`, `hcc`, `mcc`, `lcc` (Cloud cover)
        - `t` (Temperature)
        - `vis` (Visibility)
        - `prate` (Precipitation)
        - `u10`, `v10`, `u100`, `v100` (Wind components)

## Geographic Units
The primary geographic unit for US implementation is the **Balancing Authority (BA)**.
- **Granularity**: Aggregated solar generation at the BA level.
- **Alignment**: Each BA is assigned a centroid (lat/lon) to spatially align with the gridded GFS weather data.

## Usage
To collect US data:
```bash
python src/open_data_pvnet/scripts/collect_eia_data.py --start 2022-01-01 --end 2022-01-31 --output src/open_data_pvnet/data/us_solar.zarr
```

To process US GFS data (programmable usage via `nwp.gfs`):
```python
from open_data_pvnet.nwp.gfs import process_gfs_data
process_gfs_data(2023, 1, 1, region="us")
```
