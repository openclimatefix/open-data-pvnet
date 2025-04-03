"""
# How to run this script independently:
1. Ensure `ocf-data-sampler` is installed and properly configured.
2. Set the appropriate dataset path and config file.
3. Uncomment the main block below to run as a standalone script.
"""

import logging
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.torch_datasets.utils.valid_time_periods import find_valid_time_periods
from ocf_data_sampler.constants import NWP_MEANS, NWP_STDS
import fsspec
import numpy as np


# Configure logging
logging.basicConfig(level=logging.WARNING)

# Ensure xarray retains attributes during operations
xr.set_options(keep_attrs=True)


def open_gfs(dataset_path: str) -> xr.DataArray:
    """
    Opens the GFS dataset stored in Zarr format and prepares it for processing.

    Args:
        dataset_path (str): Path to the GFS dataset.

    Returns:
        xr.DataArray: The processed GFS data.
    """
    logging.info("Opening GFS dataset synchronously...")
    store = fsspec.get_mapper(dataset_path, anon=True)
    gfs_dataset: xr.Dataset = xr.open_dataset(
        store, engine="zarr", consolidated=True, chunks="auto"
    )
    gfs_data: xr.DataArray = gfs_dataset.to_array(dim="channel")

    if "init_time" in gfs_data.dims:
        logging.debug("Renaming 'init_time' to 'init_time_utc'...")
        gfs_data = gfs_data.rename({"init_time": "init_time_utc"})

    required_dims = ["init_time_utc", "step", "channel", "latitude", "longitude"]
    gfs_data = gfs_data.transpose(*required_dims)

    logging.debug(f"GFS dataset dimensions: {gfs_data.dims}")
    return gfs_data


def handle_nan_values(
    dataset: xr.DataArray, method: str = "fill", fill_value: float = 0.0
) -> xr.DataArray:
    """
    Handle NaN values in the dataset.

    Args:
        dataset (xr.DataArray): The dataset to process.
        method (str): The method for handling NaNs ("fill" or "drop").
        fill_value (float): Value to replace NaNs if method is "fill".

    Returns:
        xr.DataArray: The processed dataset.
    """
    if method == "fill":
        logging.info(f"Filling NaN values with {fill_value}.")
        return dataset.fillna(fill_value)
    elif method == "drop":
        logging.info("Dropping NaN values.")
        return dataset.dropna(dim="latitude", how="all").dropna(dim="longitude", how="all")
    else:
        raise ValueError("Invalid method for handling NaNs. Use 'fill' or 'drop'.")


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
        self.dataset = dataset
        self.config = load_yaml_configuration(config_filename)
        self.valid_t0_times = find_valid_time_periods({"nwp": {"gfs": self.dataset}}, self.config)
        logging.debug(f"Valid initialization times:\n{self.valid_t0_times}")

        if "start_dt" in self.valid_t0_times.columns:
            self.valid_t0_times = self.valid_t0_times.rename(columns={"start_dt": "t0"})

        if start_time:
            self.valid_t0_times = self.valid_t0_times[
                self.valid_t0_times["t0"] >= pd.Timestamp(start_time)
            ]
        if end_time:
            self.valid_t0_times = self.valid_t0_times[
                self.valid_t0_times["t0"] <= pd.Timestamp(end_time)
            ]

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
