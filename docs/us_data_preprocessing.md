# US Data Preprocessing for ocf-data-sampler

This document describes how to preprocess EIA solar generation data for use with ocf-data-sampler and PVNet training.

## Overview

The EIA data collected by `collect_eia_data.py` needs to be preprocessed to match the format expected by ocf-data-sampler, which follows the UK GSP data structure.

## Data Format Requirements

### Input Format (from `collect_eia_data.py`)
- **Dimensions**: `(timestamp, ba_code)`
- **Variables**: `generation_mw`, `ba_name`, `latitude`, `longitude`, `value-units`
- **Index**: MultiIndex on `(timestamp, ba_code)`

### Output Format (for ocf-data-sampler)
- **Dimensions**: `(ba_id, datetime_gmt)` where `ba_id` is int64
- **Variables**: `generation_mw`, `capacity_mwp`
- **Coordinates**: `ba_code`, `ba_name`, `latitude`, `longitude` (optional)
- **Chunking**: `{"ba_id": 1, "datetime_gmt": 1000}`
- **Format**: Zarr with consolidated metadata

## Preprocessing Steps

The preprocessing script (`preprocess_eia_for_sampler.py`) performs the following transformations:

1. **Rename timestamp to datetime_gmt**
   - Converts to UTC timezone
   - Removes timezone info (matches UK format)

2. **Map BA codes to numeric IDs**
   - Creates numeric `ba_id` (0, 1, 2, ...) for each unique `ba_code`
   - Generates metadata CSV with mapping

3. **Add capacity data**
   - Estimates `capacity_mwp` from maximum historical generation
   - Applies safety factor (1.15x) to account for capacity > max generation
   - Ensures capacity >= generation

4. **Restructure dataset**
   - Sets index to `(ba_id, datetime_gmt)`
   - Applies proper chunking for efficient access
   - Saves with consolidated metadata

## Usage

### Step 1: Collect Raw EIA Data

```bash
python src/open_data_pvnet/scripts/collect_eia_data.py \
    --start 2020-01-01 \
    --end 2023-12-31 \
    --output src/open_data_pvnet/data/target_eia_data.zarr
```

### Step 2: Preprocess for ocf-data-sampler

```bash
python src/open_data_pvnet/scripts/preprocess_eia_for_sampler.py \
    --input src/open_data_pvnet/data/target_eia_data.zarr \
    --output src/open_data_pvnet/data/target_eia_data_processed.zarr \
    --metadata-output src/open_data_pvnet/data/us_ba_metadata.csv
```

### Step 3: Verify Compatibility

```bash
python src/open_data_pvnet/scripts/test_eia_sampler_compatibility.py \
    --data-path src/open_data_pvnet/data/target_eia_data_processed.zarr \
    --config-path src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/us_configuration.yaml
```

## Capacity Data Options

The preprocessing script supports three methods for obtaining capacity data:

### 1. Estimate from Generation (Default)
```bash
--capacity-method estimate
```
Estimates capacity as `max(generation) * 1.15`. This is a simple heuristic that works reasonably well for initial testing.

### 2. Load from File
```bash
--capacity-method file --capacity-file path/to/capacity.csv
```
Loads capacity data from a CSV file with columns `ba_code` and `capacity_mwp`.

### 3. Static Value (Not Recommended)
```bash
--capacity-method static
```
Uses a static capacity value for all BAs. Only for testing.

## Output Files

### Processed Zarr Dataset
- **Location**: `src/open_data_pvnet/data/target_eia_data_processed.zarr`
- **Format**: Zarr v3 with consolidated metadata
- **Structure**: Matches UK GSP format for ocf-data-sampler compatibility

### BA Metadata CSV
- **Location**: `src/open_data_pvnet/data/us_ba_metadata.csv`
- **Columns**: `ba_id`, `ba_code`, `ba_name`, `latitude`, `longitude`
- **Purpose**: Mapping between numeric IDs and BA codes, plus spatial coordinates

## Configuration

Update the US configuration file to point to the processed data:

```yaml
# src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/us_configuration.yaml
input_data:
  gsp:
    zarr_path: "src/open_data_pvnet/data/target_eia_data_processed.zarr"
    time_resolution_minutes: 60  # EIA data is hourly
    # ... other settings
```

## Troubleshooting

### Issue: "ba_code not found in dataset"
**Solution**: Ensure the input file was created by `collect_eia_data.py` and has the correct structure.

### Issue: "Missing capacity data"
**Solution**: The script will estimate capacity automatically. If you have external capacity data, use `--capacity-method file`.

### Issue: "ocf-data-sampler compatibility test fails"
**Solution**: 
1. Check that dimensions are `(ba_id, datetime_gmt)`
2. Verify `datetime_gmt` is datetime64 without timezone
3. Ensure `capacity_mwp` variable exists
4. Check that Zarr has consolidated metadata

### Issue: "Generation exceeds capacity"
**Solution**: The script automatically ensures `capacity_mwp >= generation_mw * 1.01`. If this fails, check for data quality issues.

## Next Steps

After preprocessing:
1. Verify data with `test_eia_sampler_compatibility.py`
2. Update configuration files
3. Test data loading with ocf-data-sampler
4. Proceed with PVNet training setup

## References

- UK GSP Data Format: `src/open_data_pvnet/scripts/generate_combined_gsp.py`
- ocf-data-sampler Documentation: https://github.com/openclimatefix/ocf-data-sampler
- EIA Data Collection: `src/open_data_pvnet/scripts/collect_eia_data.py`



## S3 Data Storage & Retrieval

We support storing processed US EIA data in S3 for easier access across environments.

### 1. Uploading to S3

You can upload processed data directly to S3 using the `collect_and_preprocess_eia.py` script. This requires AWS credentials with write access to the target bucket.

```bash
python src/open_data_pvnet/scripts/collect_and_preprocess_eia.py \
    --start 2023-01-01 --end 2023-01-07 \
    --upload-to-s3 \
    --s3-bucket ocf-open-data-pvnet \
    --s3-version v1 \
    --public  # make objects public-read
```

Or uplaod existing data using the utility script:

```bash
python src/open_data_pvnet/scripts/upload_eia_to_s3.py \
    --input src/open_data_pvnet/data/target_eia_data_processed.zarr \
    --version v1 \
    --public
```

### 2. Accessing from S3

Update your `us_configuration.yaml` to point to the S3 path:

```yaml
input_data:
  gsp:
    zarr_path: "s3://ocf-open-data-pvnet/data/us/eia/v1/target_eia_data_processed.zarr"
    public: True
```

Or access directly in Python:

```python
import s3fs
import xarray as xr

s3 = s3fs.S3FileSystem(anon=True)
ds = xr.open_zarr(s3.get_mapper("s3://ocf-open-data-pvnet/data/us/eia/v1/target_eia_data_processed.zarr"), consolidated=True)
```
