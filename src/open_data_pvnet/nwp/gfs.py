import xarray as xr
import logging
from pathlib import Path
from open_data_pvnet.utils.data_uploader import upload_to_huggingface

logger = logging.getLogger(__name__)

def process_gfs_data(
    year: int,
    month: int = None,
    day: int = None,
    skip_upload: bool = False,
    overwrite: bool = False,
    archive_type: str = "zarr.zip",
) -> None:
    """
    Process GFS data for a given time period and optionally upload to HuggingFace.

    Args:
        year (int): Year to process
        month (int, optional): Month to process. If None, processes entire year
        day (int, optional): Day to process. If None, processes entire month/year
        skip_upload (bool): If True, skips uploading to HuggingFace
        overwrite (bool): If True, overwrites existing files
        archive_type (str): Type of archive to create ("zarr.zip" or "tar")
    """
    try:
        # Load GFS data
        gfs = xr.open_mfdataset(
            f"/mnt/storage_b/nwp/gfs/global/{year}*.zarr.zip",
            engine="zarr"
        )
        logger.info(f"Loaded GFS data for {year}")

        # Fix longitude range and select UK region
        gfs['longitude'] = ((gfs['longitude'] + 360) % 360)
        gfs = gfs.sel(
            latitude=slice(65, 45),
            longitude=slice(350, 362)  # UK region in [0, 360) range
        )

        # Stack variables into channel dimension
        gfs = gfs.to_array(dim="channel")

        # Optimize chunking
        chunk_sizes = {
            'init_time_utc': 1,
            'step': 4,
            'channel': -1,  # Keep all channels together
            'latitude': 1,
            'longitude': 1
        }
        gfs = gfs.chunk(chunk_sizes)

        # Save locally
        output_dir = Path(f"data/gfs/uk/gfs_uk_{year}.zarr")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        gfs.to_zarr(output_dir, mode='w')
        logger.info(f"Saved processed data to {output_dir}")

        # Upload to HuggingFace if requested
        if not skip_upload:
            upload_to_huggingface(
                config_path=Path("config.yaml"),
                folder_name=str(year),
                year=year,
                month=month if month else 1,
                day=day if day else 1,
                overwrite=overwrite,
                archive_type=archive_type
            )
            logger.info("Upload to HuggingFace completed")

    except Exception as e:
        logger.error(f"Error processing GFS data: {e}")
        raise

def verify_gfs_data(zarr_path: str) -> bool:
    """
    Verify that the processed GFS data meets the expected format.

    Args:
        zarr_path (str): Path to the zarr file to verify

    Returns:
        bool: True if verification passes
    """
    try:
        ds = xr.open_zarr(zarr_path)
        
        # Verify dimensions
        required_dims = {"init_time_utc", "latitude", "longitude", "channel"}
        if not all(dim in ds.dims for dim in required_dims):
            raise ValueError(f"Dataset missing required dimensions: {required_dims}")
        
        # Verify longitude range
        assert 0 <= ds.longitude.min() < ds.longitude.max() <= 360, \
            "Longitude range must be [0, 360)"
        
        return True
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False