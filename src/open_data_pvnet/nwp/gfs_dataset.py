"""
GFS Data Sampler

This script is designed to load, process, and sample Global Forecast System (GFS) data 
stored in Zarr format. It provides functionalities for handling NaN values, retrieving 
valid forecast initialization times, and normalizing the dataset for machine learning tasks.

The script is structured as follows:
1. **open_gfs**: Loads the dataset from an S3 or local Zarr file.
2. **handle_nan_values**: Handles missing data in the dataset.
3. **GFSDataSampler** (PyTorch Dataset): Samples and normalizes data for training.

To test the functionality, run this script directly. The `main` block loads the dataset 
and prints basic statistics for debugging.

"""

import logging
import pandas as pd
import xarray as xr
import numpy as np
import fsspec
from torch.utils.data import Dataset
from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.torch_datasets.utils.valid_time_periods import find_valid_time_periods
from ocf_data_sampler.constants import NWP_MEANS, NWP_STDS

# Configure logging format for readability
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure xarray retains attributes during operations
xr.set_options(keep_attrs=True)


def open_gfs(dataset_path: str) -> xr.DataArray:
    """
    Open the GFS dataset from a Zarr file stored remotely or locally.

    Args:
        dataset_path (str): Path to the GFS dataset (S3 or local).

    Returns:
        xr.DataArray: The processed GFS data array with required dimensions.
    """
    logging.info(f"Opening GFS dataset from {dataset_path}...")
    store = fsspec.get_mapper(dataset_path, anon=True)
    gfs_dataset: xr.Dataset = xr.open_dataset(
        store, engine="zarr", consolidated=True, chunks="auto"
    )

    # Convert dataset to DataArray for easier handling
    gfs_data: xr.DataArray = gfs_dataset.to_array(dim="channel")

    # Rename "init_time" to "init_time_utc" if necessary
    if "init_time" in gfs_data.dims:
        logging.debug("Renaming 'init_time' to 'init_time_utc'...")
        gfs_data = gfs_data.rename({"init_time": "init_time_utc"})

    # Ensure correct dimension order
    required_dims = ["init_time_utc", "step", "channel", "latitude", "longitude"]
    gfs_data = gfs_data.transpose(*required_dims)

    logging.info("GFS dataset loaded successfully.")
    return gfs_data


def handle_nan_values(
    dataset: xr.DataArray, method: str = "fill", fill_value: float = 0.0
) -> xr.DataArray:
    """
    Handle NaN values in the dataset.

    Args:
        dataset (xr.DataArray): The dataset to process.
        method (str): Method for handling NaNs ("fill" or "drop").
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

    Attributes:
        dataset (xr.DataArray): GFS dataset containing weather variables.
        valid_t0_times (pd.DataFrame): Dataframe of valid initialization times for sampling.
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
            config_filename (str): Path to the YAML configuration file.
            start_time (str, optional): Start time for filtering data.
            end_time (str, optional): End time for filtering data.
        """
        logging.info("Initializing GFSDataSampler...")
        self.dataset = dataset
        self.config = load_yaml_configuration(config_filename)

        # Retrieve valid initialization times
        self.valid_t0_times = find_valid_time_periods({"nwp": {"gfs": self.dataset}}, self.config)
        logging.info(f"Raw valid_t0_times:\n{self.valid_t0_times}")

        # Ensure multiple valid timestamps exist
        if len(self.valid_t0_times) <= 1:
            logging.warning("Only one valid t0 timestamp found. Consider adjusting max_staleness.")

        # Rename "start_dt" to "t0" for clarity
        if "start_dt" in self.valid_t0_times.columns:
            self.valid_t0_times = self.valid_t0_times.rename(columns={"start_dt": "t0"})

        # Apply time range filtering if specified
        if start_time:
            self.valid_t0_times = self.valid_t0_times[
                self.valid_t0_times["t0"] >= pd.Timestamp(start_time)
            ]
        if end_time:
            self.valid_t0_times = self.valid_t0_times[
                self.valid_t0_times["t0"] <= pd.Timestamp(end_time)
            ]

        logging.info(
            f"Total valid initialization times after filtering: {len(self.valid_t0_times)}"
        )

    def __len__(self):
        return len(self.valid_t0_times)

    def __getitem__(self, idx):
        """
        Fetch a sample based on the index.
        """
        t0 = self.valid_t0_times.iloc[idx]["t0"]
        logging.info(f"Fetching sample for t0={t0}.")
        return self._get_sample(t0)

    def _get_sample(self, t0: pd.Timestamp) -> xr.Dataset:
        """
        Retrieve a sample for a specific initialization time.

        Args:
            t0 (pd.Timestamp): The initialization time.

        Returns:
            xr.Dataset: The sampled and normalized data.
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

        logging.info(f"Expected target times: {target_times}")

        sliced_data = self.dataset.sel(
            init_time_utc=t0, step=[np.timedelta64((t - t0).value, "ns") for t in target_times]
        )
        return self._normalize_sample(sliced_data)

    def _normalize_sample(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Normalize the dataset using mean and standard deviation values.

        Args:
            dataset (xr.Dataset): The dataset to normalize.

        Returns:
            xr.Dataset: The normalized dataset.
        """
        logging.info("Normalizing dataset...")
        return (dataset - NWP_MEANS["gfs"]) / NWP_STDS["gfs"]


# if __name__ == "__main__":
#     """
#     Main block for testing the GFS data sampling process.
#     This section ensures the dataset loads correctly, handles NaNs, and samples data.

#     Steps:
#     1. Load the dataset from an S3 location.
#     2. Handle NaN values (filling with zero).
#     3. Initialize the GFSDataSampler.
#     4. Print dataset statistics for validation.
#     """
#     dataset_path = "s3://ocf-open-data-pvnet/data/gfs.zarr"
#     config_path = "src/open_data_pvnet/configs/gfs_data_config.yaml"

#     # Load dataset
#     dataset = open_gfs(dataset_path)

#     # Handle NaN values
#     dataset = handle_nan_values(dataset, method="fill", fill_value=0.0)

#     # Initialize sampler
#     sampler = GFSDataSampler(dataset, config_filename=config_path)

#     # Print statistics
#     logging.info(f"Total samples available: {len(sampler)}")
#     logging.info(f"First 5 samples: {[sampler[i] for i in range(min(5, len(sampler)))]}")
