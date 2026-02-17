# General Steps on How to Train a Model for a New Country

This guide provides step-by-step instructions for training a solar forecasting model for a new country using the Open Data PVNet framework. The process involves collecting data, configuring the system, and training the model.

## Prerequisites

Before starting, ensure you have:

- **Project Setup**: The Open Data PVNet project installed and configured (see [Getting Started Guide](getting_started.md))
- **Python Environment**: Python 3.9, 3.10, or 3.11 with all dependencies installed
- **Data Access**: Access to or ability to download:
  - Solar PV generation data for your target country
  - GFS (Global Forecast System) weather data
- **Storage**: Sufficient disk space for data storage (Zarr files can be large)
- **Optional**: AWS CLI configured (for S3 access), Hugging Face account (for model sharing)

## Choosing a Country

When selecting a country for training a new model, consider:

### Recommended Starting Countries

The following countries are good starting points due to data availability and solar capacity:

- **United Kingdom** - Already implemented (reference implementation)
- **United States** - Large solar capacity, good data availability
- **Netherlands** - High solar penetration, good data infrastructure
- **Belgium** - Strong renewable energy data systems
- **Germany** - One of the world's largest solar markets
- **France** - Significant solar capacity, good data access

### Finding Countries with Large Solar Installations

To identify other suitable countries:

1. **International Energy Agency (IEA)**: Check IEA's solar capacity statistics
2. **IRENA (International Renewable Energy Agency)**: Provides global renewable energy data
3. **National Grid Operators**: Many countries publish solar generation data
4. **Research Institutions**: Universities often maintain solar generation datasets

**Key Factors to Consider:**
- **Solar Capacity**: Countries with significant installed PV capacity (>1 GW)
- **Data Availability**: Public APIs or data portals for generation data
- **Data Quality**: Reliable, consistent, and regularly updated data
- **Geographic Coverage**: National or regional-level data (not just individual sites)

## Overview

Training a model for a new country requires:
1. **Generation Data**: Historical solar PV generation data for the country
2. **NWP Data**: Numerical Weather Prediction (weather forecast) data
3. **Configuration**: Setting up country-specific configuration files
4. **Training**: Running the model training pipeline
5. **Model Weights**: Saving and sharing the trained model

---

## Step 1: Get Generation Data

Generation data represents the actual solar PV power output for your target country. This serves as the "ground truth" for training the model.

### Options for Generation Data

#### Option A: Using Existing APIs (Recommended)

1. **PVlive API** (UK-specific, but similar APIs may exist for other countries)
   ```python
   from pvlive_api import PVLive
   
   pvl = PVLive()
   # Fetch generation data
   data = pvl.get_latest()
   ```

2. **Country-Specific APIs**
   - **Europe (Netherlands, Belgium, Germany, France)**: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) - Provides generation data for European countries
   - **United States**: 
     - EIA (Energy Information Administration) - National level data
     - Regional ISOs: CAISO (California), ERCOT (Texas), PJM (Eastern US)
   - **United Kingdom**: PVlive API (already implemented in this project)
   - Check national grid operators and government energy data portals

3. **Manual Data Collection**
   - Download from national energy/grid operator websites
   - Use data from research institutions or universities
   - **Format**: Must be converted to Zarr format

### Generation Data Schema

The data **must** be saved in Zarr format with the following schema:

- **Dimensions**: `(time_utc, location_id)`
- **Data Variables**:
  - `generation_mw`: (float32) Generation in MW
  - `capacity_mwp`: (float32) Capacity in MW peak
- **Coordinates**:
  - `time_utc`: (datetime64[ns]) Time in UTC
  - `location_id`: (int) Unique identifier for each location
  - `longitude`: (float) Longitude of the location
  - `latitude`: (float) Latitude of the location

> [!TIP]
> If `capacity_mwp` is not available, you can approximate it by taking the maximum generation value observed for that location over the time period.

For reference on how generation data is loaded, see [ocf-data-sampler/load/generation.py](https://github.com/openclimatefix/ocf-data-sampler/blob/main/ocf_data_sampler/load/generation.py).

### Storage Location

Store your generation data in a location accessible to the training pipeline:
- **Local**: `./data/{country}/generation/{year}.zarr` (Note: Do not commit this in the repository)
- **S3**: `s3://ocf-open-data-pvnet/data/{country}/generation/{year}.zarr` (Contact @peterdudfield to upload data here after model verification)
- **Hugging Face**: Upload to a Hugging Face dataset

---

## Step 2: Get NWP Data (Start with GFS)

Numerical Weather Prediction (NWP) data provides weather forecasts that the model uses to predict solar generation. For new countries, **GFS (Global Forecast System)** is recommended as it provides global coverage.

### Why GFS?

- **Global Coverage**: Available for any country worldwide
- **Public Access**: Free and openly available
- **Good Resolution**: 0.25° (~25km) resolution
- **Multiple Variables**: Includes all necessary weather parameters

### Required GFS Variables

The model needs these weather variables (channels):

- `dswrf` - Downward shortwave radiation flux (solar radiation)
- `dlwrf` - Downward longwave radiation flux
- `tcc` - Total cloud cover
- `hcc` - High cloud cover
- `mcc` - Medium cloud cover
- `lcc` - Low cloud cover
- `t` - 2-metre temperature
- `r` - Relative humidity
- `u10`, `v10` - 10-metre wind components
- `u100`, `v100` - 100-metre wind components
- `prate` - Precipitation rate
- `vis` - Visibility

### Processing GFS Data for Your Country

1. **Crop to Country Region**: Extract only the geographic area covering your country
2. **Convert to Zarr**: Save in Zarr format for efficient access
3. **Time Alignment**: Ensure timestamps align with generation data

Example:

```python
import xarray as xr
import s3fs

# Load GFS data from S3 (no need to download all of it)
gfs_ds = xr.open_zarr('s3://ocf-open-data-pvnet/data/gfs_global.zarr') # Update with actual S3 path if different

# Define bounding box for your country (example: Germany)
lat_min, lat_max = 47.0, 55.0
lon_min, lon_max = 5.0, 15.0

# Crop to country region
country_ds = gfs_ds.sel(
    latitude=slice(lat_max, lat_min),
    longitude=slice(lon_min, lon_max)
)

# Save country-specific GFS data
country_ds.to_zarr('gfs_germany.zarr', mode='w')
```

### Storage Location

Store processed GFS data:
- **Local**: `./data/{country}/gfs/{year}.zarr`
- **S3**: `s3://ocf-open-data-pvnet/data/{country}/gfs/{year}.zarr`

---

## Step 3: Make Configs for That Country

Create configuration files that tell the system where to find your data and how to process it.

### 3.1 Create Data Configuration File

Create a new configuration file: `src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/{country}_configuration.yaml`

Example for Germany (`germany_configuration.yaml`):

Please refer to the [example_configuration.yaml](https://github.com/openclimatefix/open-data-pvnet/blob/main/src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/example_configuration.yaml) for the most up-to-date structure and zarr path examples.

You can copy this file and adapt it for your country.

### 3.2 Calculate Normalization Constants

You need to calculate mean and std for each GFS channel from your data:

```python
import xarray as xr
import numpy as np

# Load your GFS data
gfs_ds = xr.open_zarr('gfs_{country}.zarr')

normalization = {}
# List all GFS channels you're using
channels = ['dlwrf', 'dswrf', 'hcc', 'mcc', 'lcc', 'prate', 'r', 't', 'tcc', 
            'u10', 'u100', 'v10', 'v100', 'vis']

for channel in channels:
    if channel in gfs_ds:
        data = gfs_ds[channel].values
        normalization[channel] = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data))
        }
    else:
        print(f"Warning: Channel {channel} not found in dataset")

print(normalization)
# Copy these values into your config file
```

### 3.3 Update Training Configuration

Edit `src/open_data_pvnet/configs/PVNet_configs/datamodule/streamed_batches.yaml`:

```yaml
_target_: pvnet.data.DataModule
# Update this path to your country configuration
configuration: "/full/path/to/open-data-pvnet/src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/{country}_configuration.yaml"

num_workers: 8
prefetch_factor: 2
batch_size: 8

# Training period (adjust to your data availability)
train_period:
  - "2023-01-01"
  - "2023-06-30"
  
# Validation period
val_period:
  - "2023-07-01"
  - "2023-12-31"
```

### 3.4 Update Main Config

Edit `src/open_data_pvnet/configs/PVNet_configs/config.yaml`:

```yaml
# ... existing config ...

# Update renewable type if needed (currently "pv_uk")
renewable: "pv_uk"  # Keep as is for now, or create new type

# ... rest of config ...
```

---

## Step 4: Run and Train Model

### 4.1 Prepare Training Samples (Optional but Recommended)

For faster training, pre-generate samples:

```bash
# Update streamed_batches.yaml first (see Step 3.3)
python src/open_data_pvnet/scripts/save_samples.py
```

This creates training samples in `GFS_samples/` directory.

### 4.2 Configure Weights & Biases (Optional)

If using W&B for experiment tracking:

1. Create account at https://wandb.ai/
2. Edit `src/open_data_pvnet/configs/PVNet_configs/logger/wandb.yaml`:
   ```yaml
   project: "{country}_solar_forecast"
   save_dir: "{country}_runs"
   ```

### 4.3 Run Training

#### Option A: Using Pre-generated Samples (Faster)

1. Update `config.yaml`:
   ```yaml
   defaults:
     - datamodule: premade_batches.yaml
   ```

2. Update `premade_batches.yaml`:
   ```yaml
   configuration: "/full/path/to/your/{country}_configuration.yaml"
   sample_dir: "GFS_samples"  # Directory containing train/ and val/ subdirectories with batches
   ```

3. Run training:
   ```bash
   python run.py
   ```

#### Option B: Streaming Data (Slower but No Pre-processing)

1. Update `config.yaml`:
   ```yaml
   defaults:
     - datamodule: streamed_batches.yaml
   ```

2. Run training:
   ```bash
   python run.py
   ```

### 4.4 Monitor Training

- **Console Output**: Watch for loss values and metrics
- **Weights & Biases**: If configured, view at https://wandb.ai/
- **Logs**: Check `outputs/` directory for training logs

### 4.5 Training Tips

- **Start Small**: Use `num_train_samples: 5` and `num_val_samples: 5` for testing
- **Debug Mode**: Use `python run.py debug=true` for quick testing
- **GPU**: Training is faster on GPU (if available)
- **Data Quality**: Ensure data alignment and no missing values

---

## Step 5: Share / Save Model Weights

### 5.1 Locate Trained Model

After training, model checkpoints are saved in:
- `outputs/{timestamp}/checkpoints/` (Hydra default)
- Or location specified in your config

### 5.2 Save Model Weights

```python
import torch

# Load the best checkpoint
checkpoint = torch.load('outputs/.../checkpoints/best.ckpt')

# Extract model weights
model_state = checkpoint['state_dict']

# Save only weights
torch.save(model_state, '{country}_model_weights.pt')
```

### 5.3 Share Model Weights

#### Option A: Hugging Face Hub (Recommended)

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj='{country}_model_weights.pt',
    path_in_repo='models/{country}_weights.pt',
    repo_id='openclimatefix/{country}-solar-forecast',
    repo_type='model',
    token=os.getenv('HUGGINGFACE_TOKEN')
)
```

#### Option B: GitHub Releases

1. Create a new release on GitHub
2. Upload model weights as release assets
3. Document model version and training details

#### Option C: S3 Bucket

```bash
aws s3 cp {country}_model_weights.pt \
  s3://ocf-open-data-pvnet/models/{country}/weights.pt
```

### 5.4 Document Model

Create a README for your model:

```markdown
# {Country} Solar Forecasting Model

## Model Details
- **Country**: {Country Name}
- **Training Period**: {Start Date} to {End Date}
- **Data Sources**: GFS NWP, {Generation Data Source}
- **Performance**: MAE: {value}, RMSE: {value}

## Usage
[Include instructions on how to use the model]
```

---

## Troubleshooting

### Common Issues

1. **Data Not Found**
   - Check file paths in configuration
   - Verify S3 bucket permissions
   - Ensure data is in Zarr format

2. **Dimension Mismatches**
   - Verify `image_size_pixels_height/width` match your data
   - Check time alignment between generation and NWP data

3. **Out of Memory**
   - Reduce `batch_size` in config
   - Use smaller `num_workers`
   - Process data in smaller chunks

4. **Poor Model Performance**
   - Check data quality (missing values, outliers)
   - Verify normalization constants
   - Ensure sufficient training data (1+ years recommended)

---

## Next Steps

After training your first model:

1. **Evaluate Performance**: Compare predictions with actual generation
2. **Iterate**: Try different hyperparameters, data periods, or features
3. **Document**: Share your findings and model with the community
4. **Contribute**: Submit your model to the Open Climate Fix repository

---

## Resources

- [Getting Started Guide](getting_started.md)
- [Working with NWP Data](working_with_nwp.md)
- [GFS Data Understanding](notebooks/understanding_gfs_data.ipynb)
- [PVNet Documentation](https://github.com/openclimatefix/pvnet)

---



---

**Need Help?** Open an issue on [GitHub](https://github.com/openclimatefix/open-data-pvnet/issues) or check the [Getting Started Guide](getting_started.md).

