# src/open_data_pvnet/utils/batch_utils.py
import os
import torch
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BatchSaveFunc:
    """
    A helper class to save batches to disk.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(self, batch, batch_num: int):
        # Define the filename (using .pt extension for PyTorch files)
        filename = os.path.join(self.output_dir, f"batch_{batch_num:08d}.pt")
        logger.info(f"Saving batch {batch_num} to {filename}")
        torch.save(batch, filename)


def process_and_save_batches(data_loader, output_dir, num_batches=None):
    """
    Iterate over the DataLoader, process each batch, and save it to disk.
    """
    save_func = BatchSaveFunc(output_dir)
    pbar = tqdm(total=num_batches or len(data_loader), desc="Saving batches")

    for batch_num, batch in enumerate(data_loader):
        save_func(batch, batch_num)
        pbar.update(1)
        if num_batches and (batch_num + 1) >= num_batches:
            break
    pbar.close()
    logger.info("Batch saving complete.")
