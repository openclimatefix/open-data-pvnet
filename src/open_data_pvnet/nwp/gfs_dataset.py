import logging
import xarray as xr
import fsspec
import pandas as pd
from torch.utils.data import Dataset
from ocf_data_sampler.config import load_yaml_configuration
from ocf_data_sampler.select import fill_time_periods, slice_datasets_by_time
from ocf_data_sampler.torch_datasets.process_and_combine import process_and_combine_datasets
from ocf_data_sampler.torch_datasets.valid_time_periods import find_valid_time_periods

xr.set_options(keep_attrs=True)

def find_valid_t0_times(datasets_dict: dict, config):
    """
    Find the t0 times where all of the requested input data is available.

    Args:
        datasets_dict (dict): Dictionary containing datasets.
        config: Configuration object.
    """
    valid_time_periods = find_valid_time_periods(datasets_dict, config)
    valid_t0_times = fill_time_periods(
        valid_time_periods,
        freq=pd.Timedelta(minutes=config.input_data.nwp.gfs.time_resolution_minutes),
    )
    return valid_t0_times


class GFSDataSampler(Dataset):
    def __init__(
        self,
        dataset_path: str,
        config_filename: str,
        start_time: str = None,
        end_time: str = None,
    ):
        """
        Dataset to create samples from GFS data in Zarr format.

        Args:
            dataset_path (str): Path to the GFS dataset (Zarr format, S3 compatible).
            config_filename (str): Path to the YAML configuration file.
            start_time (str): Start time for filtering data (optional).
            end_time (str): End time for filtering data (optional).
        """
        # Access the dataset anonymously using fsspec
        store = fsspec.get_mapper(dataset_path, anon=True)
        self.dataset = xr.open_dataset(store, engine="zarr", consolidated=True)

        # Log dataset variables
        logging.info(f"Loaded dataset with variables: {list(self.dataset.data_vars.keys())}")

        # Load the YAML configuration file
        self.config = load_yaml_configuration(config_filename)

        # Dynamically include all variables if not specified in the config
        if not self.config.input_data.nwp.gfs.channels:
            self.config.input_data.nwp.gfs.channels = list(self.dataset.data_vars.keys())

        # Convert datasets to a dictionary compatible with ocf-data-sampler
        datasets_dict = {"nwp": {"gfs": self.dataset}}

        # Get valid t0 times
        self.valid_t0_times = find_valid_t0_times(datasets_dict, self.config)

        # Filter t0 times to the specified range
        if start_time is not None:
            self.valid_t0_times = self.valid_t0_times[self.valid_t0_times >= pd.Timestamp(start_time)]
        if end_time is not None:
            self.valid_t0_times = self.valid_t0_times[self.valid_t0_times <= pd.Timestamp(end_time)]

        logging.info(f"Number of valid t0 times: {len(self.valid_t0_times)}")

    def __len__(self):
        return len(self.valid_t0_times)

    def _get_sample(self, t0: pd.Timestamp):
        """
        Generate a sample for a specific t0 time.
        Args:
            t0 (pd.Timestamp): Initialization time (forecast reference time).
        """
        # Dictionary of datasets
        datasets_dict = {"nwp": {"gfs": self.dataset}}

        # Select only the variables (channels) specified in the configuration
        if self.config.input_data.nwp.gfs.channels:
            selected_variables = self.config.input_data.nwp.gfs.channels
            datasets_dict["nwp"]["gfs"] = self.dataset[selected_variables]

        # Compute the range of steps for slicing
        interval_start = pd.Timedelta(minutes=self.config.input_data.nwp.gfs.interval_start_minutes)
        interval_end = pd.Timedelta(minutes=self.config.input_data.nwp.gfs.interval_end_minutes)

        # Ensure `t0` and `init_time_utc` are compatible
        init_time_utc = t0.floor('H')  # Adjust `t0` to match the dataset's time resolution
        start_step = int((t0 + interval_start - init_time_utc).total_seconds() // 3600)
        end_step = int((t0 + interval_end - init_time_utc).total_seconds() // 3600)

        # Debugging: Log the calculated steps and `init_time_utc`
        logging.info(f"t0: {t0}, init_time_utc: {init_time_utc}, start_step: {start_step}, end_step: {end_step}")

        # Slice the dataset using init_time_utc and step
        sliced_dataset = datasets_dict["nwp"]["gfs"].sel(
            init_time_utc=init_time_utc,
            step=slice(start_step, end_step)
        )

        # Process and combine the dataset
        sample_dict = {"nwp": {"gfs": sliced_dataset}}
        sample = process_and_combine_datasets(sample_dict, self.config, t0)
        return sample



    def __getitem__(self, idx):
        t0 = self.valid_t0_times[idx]
        return self._get_sample(t0)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Specify the S3 dataset path and configuration file
    dataset_path = "s3://ocf-open-data-pvnet/data/gfs.zarr"
    config_filename = "src/open_data_pvnet/configs/gfs_data_config.yaml"

    # Create the sampler
    sampler = GFSDataSampler(
        dataset_path=dataset_path,
        config_filename=config_filename,
        start_time=None,
        end_time=None,
    )

    # Print the length of the dataset
    print(f"Number of valid t0 times: {len(sampler)}")

    # Retrieve and print a sample
    sample = sampler[0]
    print(sample)