"""Module to process DWD (Deutscher Wetterdienst) ICON weather model data."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from cfgrib import open_dataset
import requests
import tempfile
import yaml

from open_data_pvnet.nwp.base import NWPDataSource
from open_data_pvnet.utils import get_zenith

logger = logging.getLogger(__name__)


class DWDDataSource(NWPDataSource):
    """Data source for DWD (Deutscher Wetterdienst) ICON weather model data."""

    def __init__(
        self,
        base_dir: str,
        config_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize DWD data source.

        Args:
            base_dir: Base directory for data storage
            config_path: Path to the configuration file
            **kwargs: Additional keyword arguments
        """
        super().__init__(base_dir, config_path, **kwargs)
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../..",
                "configs",
                "dwd_data_config.yaml",
            )
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Setup directories
        self.data_dir = os.path.join(base_dir, "dwd")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Set up parameters
        self.base_url = self.config["data_source"]["base_url"]
        self.variables = self.config["variables"]
        self.model_name = self.config["data_source"]["model_name"]
        self.domain = self.config["data_source"].get("domain", "europe")

    def download_data(
        self, 
        target_date: datetime,
        forecast_horizon: int,
    ) -> List[str]:
        """
        Download DWD ICON model data.

        Args:
            target_date: Date for which to download data
            forecast_horizon: Forecast horizon in hours

        Returns:
            List of paths to downloaded files
        """
        # Format the date for URL construction
        date_str = target_date.strftime("%Y%m%d")
        run_hour = target_date.strftime("%H")
        
        # Determine which forecast steps to download based on horizon
        forecast_steps = list(range(1, forecast_horizon + 1))
        
        downloaded_files = []
        
        for variable in self.variables:
            var_name = variable["name"]
            grib_shortname = variable["grib_shortName"]
            
            for step in forecast_steps:
                # Construct URL for the specific variable and forecast step
                url = f"{self.base_url}/{self.model_name}/{self.domain}/{run_hour}/grib/{var_name}/{grib_shortname}_{date_str}{run_hour}_{step:03d}.grib2"
                
                # Setup local file path
                local_path = os.path.join(
                    self.data_dir, 
                    f"{var_name}_{date_str}_{run_hour}_{step:03d}.grib2"
                )
                
                # Download the file
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    downloaded_files.append(local_path)
                    logger.info(f"Downloaded {url} to {local_path}")
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to download {url}: {e}")
        
        return downloaded_files

    def preprocess_data(
        self, 
        files: List[str],
        target_coords: Optional[pd.DataFrame] = None,
    ) -> xr.Dataset:
        """
        Preprocess DWD data files.

        Args:
            files: List of file paths to process
            target_coords: Optional DataFrame containing target coordinates

        Returns:
            Preprocessed xarray Dataset
        """
        # Process each file and combine them
        datasets = []
        
        for file_path in files:
            try:
                # Extract metadata from filename
                file_name = os.path.basename(file_path)
                var_name = file_name.split("_")[0]
                
                # Open and process the GRIB file
                ds = open_dataset(file_path)
                
                # Find the corresponding variable config
                var_config = next((v for v in self.variables if v["name"] == var_name), None)
                if var_config is None:
                    logger.warning(f"No configuration found for variable {var_name}, skipping")
                    continue
                
                # Apply conversion factor if specified
                if "conversion_factor" in var_config and var_config["conversion_factor"] != 1.0:
                    for data_var in ds.data_vars:
                        ds[data_var] = ds[data_var] * var_config["conversion_factor"]
                
                # Rename variables according to configuration
                if "output_name" in var_config:
                    # Rename the main data variable to the output name
                    for data_var in ds.data_vars:
                        ds = ds.rename({data_var: var_config["output_name"]})
                
                datasets.append(ds)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Merge all datasets
        if not datasets:
            raise ValueError("No valid datasets found to process")
        
        merged_ds = xr.merge(datasets)
        
        # If target coordinates are provided, subset the data
        if target_coords is not None:
            # Extract lat/lon points
            lats = target_coords["latitude"].values
            lons = target_coords["longitude"].values
            
            # Use xarray's sel method with method="nearest" to find closest grid points
            subset_ds = merged_ds.sel(
                latitude=xr.DataArray(lats, dims="location"),
                longitude=xr.DataArray(lons, dims="location"),
                method="nearest"
            )
            
            return subset_ds
        
        return merged_ds

    def load_forecast(
        self,
        target_date: datetime,
        forecast_horizon: int,
        target_coords: Optional[pd.DataFrame] = None,
    ) -> xr.Dataset:
        """
        Main method to load DWD forecast data.

        Args:
            target_date: Date for which to load forecast
            forecast_horizon: Forecast horizon in hours
            target_coords: Optional DataFrame containing target coordinates

        Returns:
            xarray Dataset containing forecast data
        """
        # Check if data exists already
        cache_file = os.path.join(
            self.data_dir,
            f"dwd_forecast_{target_date.strftime('%Y%m%d_%H')}_{forecast_horizon}.nc"
        )
        
        if os.path.exists(cache_file):
            logger.info(f"Loading cached forecast from {cache_file}")
            return xr.open_dataset(cache_file)
        
        # Download the data
        downloaded_files = self.download_data(target_date, forecast_horizon)
        
        if not downloaded_files:
            raise ValueError(f"No data files could be downloaded for {target_date}")
        
        # Preprocess the data
        ds = self.preprocess_data(downloaded_files, target_coords)
        
        # Add solar zenith angle if required variables are present
        if "longitude" in ds.coords and "latitude" in ds.coords and "time" in ds.coords:
            logger.info("Adding solar zenith angle to dataset")
            zenith = get_zenith(ds)
            ds["zenith"] = zenith
        
        # Save processed data to cache
        ds.to_netcdf(cache_file)
        logger.info(f"Saved processed forecast to {cache_file}")
        
        return ds
        
    def load_nwp_data(
        self,
        start_dt: datetime,
        end_dt: datetime,
        target_coords: pd.DataFrame,
        **kwargs,
    ) -> xr.Dataset:
        """
        Load NWP data for a time range.

        Args:
            start_dt: Start date
            end_dt: End date
            target_coords: DataFrame with target coordinates
            **kwargs: Additional keyword arguments

        Returns:
            xarray Dataset containing NWP data
        """
        # Calculate the forecast horizon
        forecast_horizon = int((end_dt - start_dt).total_seconds() / 3600)
        
        # Load the forecast
        forecast_ds = self.load_forecast(
            target_date=start_dt,
            forecast_horizon=forecast_horizon,
            target_coords=target_coords,
        )
        
        return forecast_ds


def get_data_source(base_dir, config_path=None, **kwargs):
    """
    Get DWD data source instance.

    Args:
        base_dir: Base directory for data storage
        config_path: Path to configuration file
        **kwargs: Additional keyword arguments

    Returns:
        DWDDataSource instance
    """
    return DWDDataSource(base_dir, config_path, **kwargs)