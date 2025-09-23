"""Constants for NWP data processing."""

import xarray as xr
import numpy as np

# GFS channels as defined in gfs_data_config.yaml
GFS_CHANNELS = [
    "dlwrf",  # Downward Long-Wave Radiation Flux
    "dswrf",  # Downward Short-Wave Radiation Flux
    "hcc",  # High Cloud Cover
    "lcc",  # Low Cloud Cover
    "mcc",  # Medium Cloud Cover
    "prate",  # Precipitation Rate
    "r",  # Relative Humidity
    "t",  # Temperature
    "tcc",  # Total Cloud Cover
    "u10",  # U-Component of Wind at 10m
    "u100",  # U-Component of Wind at 100m
    "v10",  # V-Component of Wind at 10m
    "v100",  # V-Component of Wind at 100m
    "vis",  # Visibility
]

# Define normalization constants
NWP_MEANS = {"gfs": xr.DataArray(np.zeros(len(GFS_CHANNELS)), coords={"channel": GFS_CHANNELS})}

NWP_STDS = {"gfs": xr.DataArray(np.ones(len(GFS_CHANNELS)), coords={"channel": GFS_CHANNELS})}
