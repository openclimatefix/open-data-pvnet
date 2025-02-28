# Met Office UK Deterministic (UKV) Dataset

## Overview
The Met Office UK Deterministic (UKV) dataset provides high-resolution weather forecasts for the UK region. This document details the dataset structure, variables, and their relevance to solar forecasting.

## Dataset Structure
The dataset uses a Lambert Azimuthal Equal Area projection centered on the UK with:
- Height: 970 pixels
- Width: 1042 pixels
- Grid Resolution: 2km
- Temporal Resolution: 60 minutes
- Forecast Range: 54 hours

## Variables

### Cloud Coverage Variables
These variables are critical for predicting solar irradiance attenuation:

- **high_type_cloud_area_fraction**
  - Description: Fraction of high-altitude clouds
  - Units: 1 (fraction)
  - Impact: High clouds typically have less impact on solar radiation than lower clouds
  - Typical Range: 0-1

- **medium_type_cloud_area_fraction**
  - Description: Fraction of medium-altitude clouds
  - Units: 1 (fraction)
  - Impact: Moderate impact on solar radiation
  - Typical Range: 0-1

- **low_type_cloud_area_fraction**
  - Description: Fraction of low-altitude clouds
  - Units: 1 (fraction)
  - Impact: Most significant impact on solar radiation
  - Typical Range: 0-1

- **cloud_area_fraction**
  - Description: Total cloud coverage
  - Units: 1 (fraction)
  - Impact: Overall indicator of solar radiation reduction
  - Typical Range: 0-1

### Radiation Flux Variables
Direct measurements of solar radiation components:

- **surface_downwelling_shortwave_flux_in_air**
  - Description: Total downward solar radiation at surface
  - Units: W m⁻²
  - Impact: Primary predictor for solar PV generation
  - Typical Range: 0-1000+ W/m²
  - Notes: Includes both direct and diffuse radiation

- **surface_downwelling_longwave_flux_in_air**
  - Description: Thermal radiation from atmosphere
  - Units: W m⁻²
  - Impact: Affects panel temperature and efficiency
  - Typical Range: 200-500 W/m²

- **surface_downwelling_ultraviolet_flux_in_air**
  - Description: UV component of solar radiation
  - Units: W m⁻²
  - Impact: Can affect panel degradation and specific PV technologies
  - Typical Range: 0-100 W/m²

### Meteorological Variables
Environmental conditions affecting solar panel efficiency:

- **air_temperature**
  - Description: Air temperature at 2m height
  - Units: K (Kelvin)
  - Impact: Panel efficiency decreases with temperature
  - Typical Range: 250-320K
  - Note: Convert to Celsius by subtracting 273.15

- **wind_speed**
  - Description: Wind speed at surface level
  - Units: m s⁻¹
  - Impact: Affects panel cooling and efficiency
  - Typical Range: 0-30 m/s

- **wind_from_direction**
  - Description: Wind direction at surface level
  - Units: degrees
  - Impact: Can influence panel temperature and local weather patterns
  - Range: 0-360°
  - Note: 0° is North, 90° is East

- **lwe_thickness_of_surface_snow_amount**
  - Description: Snow depth in water equivalent
  - Units: m
  - Impact: Affects ground albedo and potential panel coverage
  - Typical Range: 0-1m

### Coordinate System
The dataset uses Lambert Azimuthal Equal Area projection:

- **projection_x_coordinate**
  - Description: X-axis grid coordinates
  - Units: m
  - Range: Covers UK extent

- **projection_y_coordinate**
  - Description: Y-axis grid coordinates
  - Units: m
  - Range: Covers UK extent

### Time Variables
Temporal information for forecasts:

- **forecast_period**
  - Description: Time offset from reference
  - Type: timedelta64[ns]

- **forecast_reference_time**
  - Description: Start time of forecast
  - Type: datetime64[ns]

- **time**
  - Description: Valid time for forecast step
  - Type: datetime64[ns]

## Usage in Solar Forecasting

### Primary Predictors
1. **surface_downwelling_shortwave_flux_in_air**: Direct indicator of solar energy availability
2. **cloud_area_fraction** variables: Key for radiation attenuation
3. **air_temperature**: Critical for panel efficiency calculations

### Secondary Factors
1. **wind_speed**: Panel cooling effects
2. **snow_amount**: Ground reflectance and coverage
3. **UV flux**: Specific panel technology considerations

## Data Quality Considerations
- Least significant digit information provided for each variable
- Grid mapping information available in lambert_azimuthal_equal_area variable
- All variables follow CF-1.7 conventions

## Data Availability and Format
This dataset is hosted on Hugging Face at [openclimatefix/met-office-uk-deterministic-solar](https://huggingface.co/datasets/openclimatefix/met-office-uk-deterministic-solar).

### File Format
- Files are stored in `.zarr.zip` format
- Each file represents a specific timestamp (e.g., `2023-01-16-00.zarr.zip`)
- Zarr format is optimized for:
  - Cloud storage access
  - Parallel I/O operations
  - Efficient chunked access to large arrays
  - Integration with data science tools (xarray, pandas, dask)

## License
British Crown copyright 2022-2024, the Met Office, licensed under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).

## Citation
Met Office UK Deterministic (UKV)2km on a 2-year rolling archive accessed from [AWS Registry](https://registry.opendata.aws/met-office-uk-deterministic).
