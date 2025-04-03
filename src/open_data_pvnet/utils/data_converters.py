import logging
from pathlib import Path

import xarray as xr
import zarr

logger = logging.getLogger(__name__)


def convert_nc_to_zarr(input_dir: Path, output_dir: Path, overwrite: bool = False):
    """
    Convert all .nc files in the input directory to Zarr format.

    Args:
        input_dir (Path): Directory containing .nc files.
        output_dir (Path): Directory to save the converted .zarr files.
        overwrite (bool): Whether to overwrite existing Zarr files. Default is False.

    Returns:
        tuple: (int, float) - Number of files converted and total size in MB.

    Raises:
        FileNotFoundError: If input_dir does not exist.
    """
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        total_files = 0
        total_size_bytes = 0
        compression_settings = {"compressor": zarr.Blosc(cname="zstd", clevel=5, shuffle=2)}

        nc_files = list(input_dir.glob("*.nc"))
        if not nc_files:
            logger.warning(f"No .nc files found in directory: {input_dir}")
            return total_files, total_size_bytes

        for nc_file in nc_files:
            logger.info(f"Processing file: {nc_file}")
            try:
                zarr_file = output_dir / f"{nc_file.stem}.zarr"

                if zarr_file.exists() and not overwrite:
                    logger.info(f"Skipping {nc_file}, Zarr file already exists: {zarr_file}")
                    continue

                # Open and process the NetCDF file
                with xr.open_dataset(nc_file) as ds:
                    try:
                        ds.to_zarr(
                            zarr_file,
                            mode="w",
                            encoding={var: compression_settings for var in ds.data_vars},
                        )
                    except ValueError as e:
                        logger.warning(f"Skipping compression for {nc_file}: {e}")
                        ds.to_zarr(zarr_file, mode="w")

                logger.info(f"Converted {nc_file} to {zarr_file}")

                total_files += 1
                total_size_bytes += sum(f.stat().st_size for f in zarr_file.rglob("*"))

            except Exception as e:
                logger.error(f"Error converting {nc_file} to Zarr: {e}")
                continue

        logger.info(
            f"Completed conversion: {total_files} files, {total_size_bytes / (1024 ** 2):.2f} MB"
        )
        return total_files, total_size_bytes / (1024**2)

    except Exception as e:
        logger.error(f"Error in conversion process: {e}")
        raise
