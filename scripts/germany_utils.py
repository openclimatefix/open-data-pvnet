"""Utility functions for Germany solar forecasting pipeline."""

import logging
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr
import yaml

logger = logging.getLogger(__name__)


def get_time_coord(ds: xr.Dataset, is_pv: bool = False) -> str:
    """Get the time coordinate name from dataset."""
    if is_pv:
        return 'datetime_gmt' if 'datetime_gmt' in ds.dims else 'time'
    return 'init_time_utc' if 'init_time_utc' in ds.dims else 'time'


def get_gen_var(ds: xr.Dataset) -> str:
    """Get the generation variable name from PV dataset."""
    return 'generation_mw' if 'generation_mw' in ds else 'generation'


def validate_pv_data(ds: xr.Dataset) -> bool:
    """Validate PV generation data."""
    try:
        gen_var = get_gen_var(ds)
        gen = ds[gen_var].values
        if np.any(gen < 0):
            logger.error("Found negative generation values")
            return False
        logger.info("✓ PV data validation passed")
        return True
    except Exception as e:
        logger.error(f"✗ PV validation failed: {e}")
        return False


def validate_gfs_data(ds: xr.Dataset) -> bool:
    """Validate GFS weather data."""
    try:
        for var in ds.data_vars:
            data = ds[var].values
            nan_pct = (np.isnan(data).sum() / data.size) * 100
            if nan_pct > 50:
                logger.warning(f"{var}: {nan_pct:.1f}% NaN values")
        logger.info("✓ GFS data validation passed")
        return True
    except Exception as e:
        logger.error(f"✗ GFS validation failed: {e}")
        return False


def calculate_normalization(ds: xr.Dataset) -> Dict[str, Dict[str, float]]:
    """Calculate mean and std for all variables."""
    constants = {}
    for var in ds.data_vars:
        data = ds[var].values
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            constants[var] = {
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data))
            }
    return constants


def check_temporal_alignment(pv_ds: xr.Dataset, gfs_ds: xr.Dataset) -> Dict:
    """Check temporal overlap between PV and GFS data."""
    pv_time_coord = get_time_coord(pv_ds, is_pv=True)
    gfs_time_coord = get_time_coord(gfs_ds, is_pv=False)
    
    pv_times = pd.to_datetime(pv_ds[pv_time_coord].values)
    gfs_times = pd.to_datetime(gfs_ds[gfs_time_coord].values)
    
    pv_start, pv_end = pv_times.min(), pv_times.max()
    gfs_start, gfs_end = gfs_times.min(), gfs_times.max()
    
    overlap_start = max(pv_start, gfs_start)
    overlap_end = min(pv_end, gfs_end)
    overlap_days = (overlap_end - overlap_start).days if overlap_start < overlap_end else 0
    
    return {
        'pv_start': str(pv_start),
        'pv_end': str(pv_end),
        'gfs_start': str(gfs_start),
        'gfs_end': str(gfs_end),
        'overlap_start': str(overlap_start),
        'overlap_end': str(overlap_end),
        'overlap_days': overlap_days
    }


def save_normalization(constants: Dict, output_path: str):
    """Save normalization constants to YAML."""
    with open(output_path, 'w') as f:
        yaml.dump(constants, f, default_flow_style=False)
    logger.info(f"Saved normalization constants to {output_path}")


def print_zarr_summary(zarr_path: str):
    """Print a human-readable summary of a Zarr file."""
    try:
        ds = xr.open_zarr(zarr_path)
        print(f"\n{'='*60}")
        print(f"Zarr File: {zarr_path}")
        print(f"{'='*60}")
        print(ds)
        print(f"\nVariables: {list(ds.data_vars)}")
    except Exception as e:
        logger.error(f"Error: {e}")
