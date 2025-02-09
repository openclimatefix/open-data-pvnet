# batch_samples.py
import logging
import sys
import math
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Import functions and classes from their canonical modules
from src.open_data_pvnet.nwp.gfs_dataset import open_gfs, handle_nan_values, GFSDataSampler
from src.open_data_pvnet.utils.batch_utils import process_and_save_batches

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


# Define an identity collate function at the module level.
# This function simply returns the batch as is.
def identity_collate(batch):
    return batch


def run_batch_samples(
    dataset_path: str,
    config_path: str,
    start_time: str,
    end_time: str,
    output_directory: str,
    dataloader_kwargs: dict = None,
    num_batches: int = None,
):
    """
    Runs the batch sampling process: loads the dataset, wraps it in a DataLoader,
    and processes/saves batches.
    """
    # Set the multiprocessing start method if not already set.
    if mp.get_start_method(allow_none=True) != "forkserver":
        try:
            mp.set_start_method("forkserver", force=True)
        except RuntimeError:
            logger.warning("Multiprocessing start method already set. Proceeding.")
    mp.set_sharing_strategy("file_system")

    # Set default DataLoader kwargs if not provided.
    if dataloader_kwargs is None:
        dataloader_kwargs = {
            "batch_size": 4,  # Example batch size; adjust as needed.
            "shuffle": True,
            "num_workers": 2,
            "prefetch_factor": 2,
            "pin_memory": False,
            # Use the module-level identity_collate function.
            "collate_fn": identity_collate,
        }

    # Load and preprocess the raw dataset.
    logger.info("Loading GFS dataset from %s", dataset_path)
    gfs_data = open_gfs(dataset_path)
    gfs_data = handle_nan_values(gfs_data, method="fill", fill_value=0.0)

    # Initialize the custom dataset sampler.
    logger.info("Initializing GFSDataSampler with config: %s", config_path)
    gfs_sampler = GFSDataSampler(
        gfs_data, config_filename=config_path, start_time=start_time, end_time=end_time
    )

    # Enhanced logging: log number of samples and expected number of batches.
    num_samples = len(gfs_sampler)
    batch_size = dataloader_kwargs.get("batch_size", 1)
    expected_batches = math.ceil(num_samples / batch_size)
    logger.info("Total samples in dataset: %d", num_samples)
    logger.info("Expected number of batches (with batch_size=%d): %d", batch_size, expected_batches)

    # Create a DataLoader to batch the dataset.
    logger.info("Creating DataLoader with parameters: %s", dataloader_kwargs)
    data_loader = DataLoader(gfs_sampler, **dataloader_kwargs)

    # Optional: verify by iterating over one batch.
    logger.info("Verifying DataLoader by iterating over the first batch...")
    for batch in data_loader:
        logger.info("Received a batch of samples (verification). Batch type: %s", type(batch))
        break

    # Process and save batches.
    logger.info("Processing and saving batches to directory: %s", output_directory)
    process_and_save_batches(data_loader, output_directory, num_batches=num_batches)
    logger.info("All batches processed and saved.")


# The __main__ block allows the script to be run standalone.
# However, the main functionality is encapsulated in run_batch_samples(),
# so it can also be imported and called from other parts of the repo.
if __name__ == "__main__":
    # Example configuration values.
    dataset_path = "s3://ocf-open-data-pvnet/data/gfs.zarr"
    config_path = "src/open_data_pvnet/configs/gfs_data_config.yaml"
    start_time = "2023-01-01T00:00:00"
    end_time = "2023-02-28T00:00:00"
    output_directory = "./saved_batches"

    run_batch_samples(
        dataset_path, config_path, start_time, end_time, output_directory, num_batches=10
    )
