# Working with Numerical Weather Prediction (NWP) Data

This guide provides detailed information about working with Numerical Weather Prediction (NWP) data in the context of solar forecasting, serving as a starting point for specific implementations found in the project's codebase and documentation.

## Table of Contents
1. [Introduction](#introduction)
2. [Common NWP Data Sources](#common-nwp-data-sources)
3. [Data Formats and Structure](#data-formats-and-structure)
4. [Open NWP Data Sources](#open-nwp-data-sources)
5. [Key Variables for Solar Forecasting](#key-variables-for-solar-forecasting)
6. [Working with NWP Data in Python](#working-with-nwp-data-in-python)
7. [Best Practices](#best-practices)
8. [Common Challenges](#common-challenges)
9. [Additional Resources](#additional-resources)
10. [Configuration Files](#configuration-files)

## Introduction

**Numerical Weather Prediction (NWP)** data uses mathematical models of the atmosphere and oceans to forecast weather. It predicts various atmospheric conditions such as temperature, pressure, wind speed, humidity, precipitation type and amount, cloud cover, and sometimes even surface conditions and air quality—all of which are crucial for solar forecasting.

## Common NWP Data Sources

### Global Models
- **ECMWF IFS**
  - High-resolution global forecasts
  - Requires license/subscription
  - Available through Copernicus Climate Data Store

- **GFS (Global Forecast System)**
  - Free, global coverage
  - Lower resolution than ECMWF
  - Updated every 6 hours

- **ERA5**
  - ECMWF's reanalysis dataset
  - Historical weather data from 1940 onwards
  - Excellent for training models

### Regional Models
- **UK Met Office UKV**
  - High-resolution UK coverage
  - Specifically tuned for UK weather patterns

- **DWD ICON**
  - German Weather Service model
  - High resolution over Europe

## Data Formats and Structure

### Common File Formats
- **GRIB2**: Standard format for weather data
  ```python
  import xarray as xr
  import cfgrib

  # Reading GRIB2 files
  ds = xr.open_dataset('forecast.grib', engine='cfgrib')
  ```

- **NetCDF (.nc)**: Common for research and archived data
  ```python
  # Reading NetCDF files
  ds = xr.open_dataset('forecast.nc')
  ```

- **Zarr**: Cloud-optimized format
  ```python
  # Reading Zarr files
  ds = xr.open_zarr('s3://bucket/forecast.zarr')
  ```

### Data Structure
NWP data typically includes:
- Spatial dimensions (latitude, longitude)
- Vertical levels (pressure or height)
- Time dimension
- Multiple variables

## Open NWP Data Sources
[Go to Datasets](datasets.md)


### Essential Variables
1. **Cloud Cover**
   - Total cloud cover
   - Cloud cover by layer
   - Cloud type

2. **Radiation Components**
   - Global Horizontal Irradiance (GHI)
   - Direct Normal Irradiance (DNI)
   - Diffuse Horizontal Irradiance (DHI)

3. **Atmospheric Conditions**
   - Temperature
   - Humidity
   - Aerosol optical depth
   - Pressure

### Example Variable Access
```python
import xarray as xr

# Load dataset
ds = xr.open_dataset('nwp_forecast.nc')

# Access specific variables
cloud_cover = ds['total_cloud_cover']
temperature = ds['temperature']
ghi = ds['surface_solar_radiation_downwards']
```

## Working with NWP Data in Python

### Essential Libraries
```python
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy  # for geographic plotting
```

### Common Operations

#### Selecting a Location
```python
def get_location_data(ds, lat, lon):
    """Extract time series for a specific location."""
    return ds.sel(latitude=lat, longitude=lon, method='nearest')
```

#### Time Series Extraction
```python
def extract_forecast_timeline(ds, variable, lat, lon):
    """Extract forecast timeline for a specific variable and location."""
    location_data = get_location_data(ds, lat, lon)
    return location_data[variable].to_pandas()
```

#### Spatial Subsetting
```python
def subset_region(ds, lat_range, lon_range):
    """Subset data for a specific geographic region."""
    return ds.sel(
        latitude=slice(lat_range[0], lat_range[1]),
        longitude=slice(lon_range[0], lon_range[1])
    )
```

## Best Practices

1. **Data Loading**
   - Use dask for large datasets
   - Load only required variables
   - Subset data spatially when possible

2. **Memory Management**
   - Close datasets when done
   - Use chunks appropriately
   - Clean up temporary files

3. **Preprocessing**
   - Check for missing values
   - Validate data ranges
   - Align timestamps to your needs

## Common Challenges

1. **Missing Data**
   ```python
   def handle_missing_data(ds, variable):
       """Handle missing values in NWP data."""
       # Check for missing values
       missing = ds[variable].isnull()

       # Basic interpolation for missing values
       if missing.any():
           return ds[variable].interpolate_na(dim='time')
       return ds[variable]
   ```

2. **Time Zone Handling**
   ```python
   def standardize_timezone(ds):
       """Convert timestamps to UTC if needed."""
       if ds.time.dtype != 'datetime64[ns]':
           ds['time'] = pd.to_datetime(ds.time)
       return ds
   ```

3. **Coordinate Systems**
   ```python
   def ensure_standard_coords(ds):
       """Ensure coordinates are in standard format."""
       # Standardize longitude to -180 to 180
       if (ds.longitude > 180).any():
           ds['longitude'] = xr.where(
               ds.longitude > 180,
               ds.longitude - 360,
               ds.longitude
           )
       return ds
   ```

## Additional Resources

- [ECMWF Documentation](https://www.ecmwf.int/en/forecasts/documentation-and-support)
- [GFS Documentation](https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php)
- [xarray Documentation](https://xarray.pydata.org/en/stable/)
- [Pangeo - Big Data Geoscience](https://pangeo.io/)
- [NetCDF Documentation](https://www.unidata.ucar.edu/software/netcdf/docs/)
- [NetCDF User Guide](https://docs.unidata.ucar.edu/netcdf-c/current/guide.html)
- [AWS CLI Documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html)
- [AWS CLI S3 Commands](https://docs.aws.amazon.com/cli/latest/reference/s3/)

---

## Configuration Files
The `configs/` directory contains YAML configuration files for various NWP data sources. These files define the input variables, output paths, and processing parameters.

- `met_office_data_config.yaml`: Configuration for Met Office NWP data.
- `gfs_data_config.yaml`: Configuration for GFS NWP data (to be implemented).

#### Modifying Configurations
To customize the processing pipeline:
1. Navigate to the `configs/` directory.
2. Edit the YAML files using any text editor.
3. Ensure paths and parameters match your local or cloud setup.
