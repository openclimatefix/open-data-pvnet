"""
# How to run this script independently:
1. Ensure `ocf-data-sampler` is installed and properly configured.
2. Set the appropriate dataset path and config file.
3. Uncomment the main block below to run as a standalone script.
"""

# Standard library imports
from typing import Union, Optional, Dict, Any, List
import logging

# Third-party imports
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
import fsspec
import numpy as np
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

class GFSConfig(BaseModel):
    channels: List[str]
    time_resolution_minutes: int
    interval_start_minutes: int
    interval_end_minutes: int
    provider: str
    zarr_path: str
    image_size_pixels_height: int
    image_size_pixels_width: int
    max_staleness_minutes: Optional[int] = None  # Changed from max_staleness_hours
    dropout_timedeltas_minutes: Optional[List[int]] = None
    accum_channels: Optional[List[str]] = []

class NWPConfig(BaseModel):
    gfs: GFSConfig

class InputDataConfig(BaseModel):
    nwp: NWPConfig

class GeneralConfig(BaseModel):
    name: str
    description: str

class Configuration(BaseModel):
    general: GeneralConfig
    input_data: InputDataConfig

# Define default values for NWP statistics
DEFAULT_GFS_CHANNELS = [
    "dlwrf", "dswrf", "hcc", "lcc", "mcc", "prate", 
    "r", "t", "tcc", "u10", "u100", "v10", "v100", "vis"
]

try:
    from ocf_data_sampler.config import load_yaml_configuration
    from ocf_data_sampler.torch_datasets.utils.valid_time_periods import find_valid_time_periods
    from ocf_data_sampler.constants import NWP_MEANS, NWP_STDS
except ImportError:
    logging.warning("Could not import ocf_data_sampler modules. Using default implementations.")
    
    def load_yaml_configuration(config_path):
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def find_valid_time_periods(dataset_dict, config):
        # Simple implementation that returns all times
        times = dataset_dict['nwp']['gfs'].init_time_utc.values
        return pd.DataFrame({'t0': times})
    
    # Create default normalization values
    NWP_MEANS = {
        "gfs": xr.DataArray(
            np.zeros(len(DEFAULT_GFS_CHANNELS)),
            coords={'channel': DEFAULT_GFS_CHANNELS},
            dims=['channel']
        )
    }
    
    NWP_STDS = {
        "gfs": xr.DataArray(
            np.ones(len(DEFAULT_GFS_CHANNELS)),
            coords={'channel': DEFAULT_GFS_CHANNELS},
            dims=['channel']
        )
    }


# Configure logging
logging.basicConfig(level=logging.WARNING)

# Ensure xarray retains attributes during operations
xr.set_options(keep_attrs=True)


def open_gfs(path: str) -> Union[xr.Dataset, xr.DataArray]:
    """
    Open a GFS dataset from a Zarr store.
    
    Args:
        path: Path to the Zarr store
        
    Returns:
        xr.Dataset or xr.DataArray: The loaded GFS dataset
    """
    try:
        # Open the dataset
        ds = xr.open_zarr(path)
        
        # Basic validation
        required_dims = {"init_time_utc", "latitude", "longitude", "channel"}
        if not all(dim in ds.dims for dim in required_dims):
            raise ValueError(f"Dataset missing required dimensions: {required_dims}")
            
        # Ensure longitude is in [-180, 180] range
        if (ds.longitude > 180).any():
            ds['longitude'] = ((ds['longitude'] + 180) % 360) - 180
            ds = ds.sortby('longitude')
            
        return ds
        
    except Exception as e:
        raise IOError(f"Failed to open GFS dataset at {path}: {str(e)}")


def handle_nan_values(dataset: xr.Dataset, method: str = "fill", fill_value: float = 0.0) -> xr.Dataset:
    """Handle NaN values in the dataset.
    
    Args:
        dataset: Input xarray Dataset
        method: Method to handle NaNs ('fill' or 'drop')
        fill_value: Value to use for filling NaNs when method='fill'
    
    Returns:
        Processed dataset with NaNs handled
    """
    if method == "fill":
        return dataset.fillna(fill_value)
    elif method == "drop":
        # Drop time steps that contain any NaN values
        return dataset.dropna(dim='init_time_utc', how='any')
    else:
        raise ValueError(f"Unknown method: {method}")


class GFSDataSampler(Dataset):
    """
    A PyTorch Dataset for sampling and normalizing GFS data.
    """

    def __init__(
        self,
        dataset: xr.DataArray,
        config_filename: str,
        start_time: str = None,
        end_time: str = None,
    ):
        """
        Initialize the GFSDataSampler.

        Args:
            dataset (xr.DataArray): The dataset to sample from.
            config_filename (str): Path to the configuration file.
            start_time (str, optional): Start time for filtering data.
            end_time (str, optional): End time for filtering data.
        """
        logging.info("Initializing GFSDataSampler...")
        
        # Validate input dataset
        if not isinstance(dataset, (xr.DataArray, xr.Dataset)):
            raise TypeError("Dataset must be an xarray DataArray or Dataset")
        
        # Validate required dimensions
        required_dims = {'channel', 'init_time_utc', 'step', 'latitude', 'longitude'}
        missing_dims = required_dims - set(dataset.dims)
        if missing_dims:
            raise ValueError(f"Dataset missing required dimensions: {missing_dims}")
        
        self.dataset = dataset
        self.config = load_yaml_configuration(config_filename)
        
        # Validate config
        if not hasattr(self.config.input_data.nwp.gfs, 'channels'):
            raise ValueError("Config missing required field: input_data.nwp.gfs.channels")
        
        self.valid_t0_times = find_valid_time_periods({"nwp": {"gfs": self.dataset}}, self.config)
        logging.debug(f"Valid initialization times:\n{self.valid_t0_times}")

        if "start_dt" in self.valid_t0_times.columns:
            self.valid_t0_times = self.valid_t0_times.rename(columns={"start_dt": "t0"})

        # Filter by time range if provided
        if start_time:
            start_ts = pd.Timestamp(start_time)
            self.valid_t0_times = self.valid_t0_times[
                self.valid_t0_times["t0"] >= start_ts
            ]
        if end_time:
            end_ts = pd.Timestamp(end_time)
            self.valid_t0_times = self.valid_t0_times[
                self.valid_t0_times["t0"] <= end_ts
            ]

        if len(self.valid_t0_times) == 0:
            raise ValueError("No valid time periods found in dataset")

        logging.debug(f"Filtered valid_t0_times:\n{self.valid_t0_times}")

    def __len__(self):
        return len(self.valid_t0_times)

    def __getitem__(self, idx):
        t0 = self.valid_t0_times.iloc[idx]["t0"]
        return self._get_sample(t0)

    def _get_sample(self, t0: pd.Timestamp) -> xr.Dataset:
        """
        Retrieve a sample for a specific initialization time.

        Args:
            t0 (pd.Timestamp): The initialization time.

        Returns:
            xr.Dataset: The sampled data.
        """
        logging.info(f"Generating sample for t0={t0}...")
        interval_start = pd.Timedelta(minutes=self.config.input_data.nwp.gfs.interval_start_minutes)
        interval_end = pd.Timedelta(minutes=self.config.input_data.nwp.gfs.interval_end_minutes)
        time_resolution = pd.Timedelta(
            minutes=self.config.input_data.nwp.gfs.time_resolution_minutes
        )

        start_dt = t0 + interval_start
        end_dt = t0 + interval_end
        target_times = pd.date_range(start=start_dt, end=end_dt, freq=time_resolution)
        logging.debug(f"Target times: {target_times}")

        valid_steps = [np.timedelta64((time - t0).value, "ns") for time in target_times]
        available_steps = self.dataset.step.values
        valid_steps = [step for step in valid_steps if step in available_steps]

        if not valid_steps:
            raise ValueError(f"No valid steps found for t0={t0}")

        sliced_data = self.dataset.sel(init_time_utc=t0, step=valid_steps)
        return self._normalize_sample(sliced_data)

    def _normalize_sample(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Normalize the dataset using precomputed means and standard deviations.

        Args:
            dataset (xr.Dataset): The dataset to normalize.

        Returns:
            xr.Dataset: The normalized dataset.
        """
        logging.info("Starting normalization...")
        provider = self.config.input_data.nwp.gfs.provider
        dataset_channels = dataset.channel.values
        mean_channels = NWP_MEANS[provider].channel.values
        std_channels = NWP_STDS[provider].channel.values

        valid_channels = set(dataset_channels) & set(mean_channels) & set(std_channels)
        missing_in_dataset = set(mean_channels) - set(dataset_channels)
        missing_in_means = set(dataset_channels) - set(mean_channels)

        if missing_in_dataset:
            logging.warning(f"Channels missing in dataset: {missing_in_dataset}")
        if missing_in_means:
            logging.warning(f"Channels missing in normalization stats: {missing_in_means}")

        valid_channels = list(valid_channels)
        dataset = dataset.sel(channel=valid_channels)
        means = NWP_MEANS[provider].sel(channel=valid_channels)
        stds = NWP_STDS[provider].sel(channel=valid_channels)

        logging.debug(f"Selected Channels: {valid_channels}")
        logging.debug(f"Mean Values: {means.values}")
        logging.debug(f"Std Values: {stds.values}")

        try:
            normalized_dataset = (dataset - means) / stds
            logging.info("Normalization completed.")
            return normalized_dataset
        except Exception as e:
            logging.error(f"Error during normalization: {e}")
            raise e


# # Uncomment the block below to test
# if __name__ == "__main__":
#     dataset_path = "s3://ocf-open-data-pvnet/data/gfs.zarr"
#     config_path = "src/open_data_pvnet/configs/gfs_data_config.yaml"
#     dataset = open_gfs(dataset_path)
#     dataset = handle_nan_values(dataset, method="fill", fill_value=0.0)
#     sampler = GFSDataSampler(dataset, config_filename=config_path, start_time="2023-01-01T00:00:00", end_time="2023-01-30T00:00:00")
#     sample = sampler[0]
#     print(sample)
