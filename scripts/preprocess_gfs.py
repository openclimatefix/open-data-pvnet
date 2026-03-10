import xarray as xr
import logging
from open_data_pvnet.nwp.gfs_dataset import preprocess_gfs_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_gfs_year(year: int, input_path: str, output_path: str) -> None:
    """
    Process GFS data for a specific year.

    Args:
        year (int): Year to process
        input_path (str): Input file pattern
        output_path (str): Output path for processed data
    """
    logger.info(f"Processing GFS data for year {year}")

    # Load data
    gfs = xr.open_mfdataset(f"{input_path}/{year}*.zarr.zip", engine="zarr")

    # Apply preprocessing
    gfs = preprocess_gfs_data(gfs)

    # Save processed data
    logger.info(f"Saving processed data to {output_path}")
    gfs.to_zarr(f"{output_path}/gfs_uk_{year}.zarr", mode="w")
    logger.info(f"Completed processing for {year}")


if __name__ == "__main__":
    for year in [2023, 2024]:
        process_gfs_year(year=year, input_path="/mnt/storage_b/nwp/gfs/global", output_path="uk")
