import logging
from pathlib import Path
import shutil

from open_data_pvnet.utils.env_loader import PROJECT_BASE
from open_data_pvnet.utils.config_loader import load_config
from open_data_pvnet.utils.data_converters import convert_nc_to_zarr
from open_data_pvnet.utils.data_uploader import upload_to_huggingface

import boto3

logger = logging.getLogger(__name__)

# Define the configuration paths for UK and Global
CONFIG_PATHS = {
    "uk": PROJECT_BASE / "src/open_data_pvnet/configs/met_office_uk_data_config.yaml",
    "global": PROJECT_BASE / "src/open_data_pvnet/configs/met_office_global_data_config.yaml",
}


def generate_prefix(region: str, year: int, month: int, day: int, hour: int) -> str:
    """
    Generate the S3 prefix for the given region, date, and hour.

    Args:
        region (str): Either 'uk' or 'global'.
        year (int): Year (e.g., 2022).
        month (int): Month (e.g., 12).
        day (int): Day (e.g., 1).
        hour (int): Hour in 24-hour format (e.g., 0 for 00:00Z).

    Returns:
        str: The S3 prefix for the specified date and hour.
    """
    folder = "uk-deterministic-2km" if region == "uk" else "global-deterministic-10km"
    return f"{folder}/{year:04d}{month:02d}{day:02d}T{hour:02d}00Z/"


def fetch_met_office_data(year: int, month: int, day: int, hour: int, region: str):
    """
    Fetch Met Office NWP data for the specified year, month, day, hour, and region.

    Args:
        year (int): Year to fetch data for.
        month (int): Month to fetch data for.
        day (int): Day to fetch data for.
        hour (int): Hour of data in 24-hour format.
        region (str): Region to fetch data for ("uk" or "global").
    """
    if region not in CONFIG_PATHS:
        raise ValueError(f"Invalid region '{region}'. Must be 'uk' or 'global'.")

    config_path = CONFIG_PATHS[region]
    config = load_config(config_path)

    s3_bucket = config["input_data"]["nwp"]["met_office"]["s3_bucket"]
    nwp_channels = set(config["input_data"]["nwp"]["met_office"]["nwp_channels"])
    nwp_accum_channels = set(config["input_data"]["nwp"]["met_office"]["nwp_accum_channels"])
    required_files = nwp_channels | nwp_accum_channels

    raw_dir = (
        Path(PROJECT_BASE)
        / config["input_data"]["nwp"]["met_office"]["local_output_dir"]
        / "raw"
        / f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fetching Met Office data to {raw_dir} from S3 bucket {s3_bucket}.")

    s3 = boto3.client("s3")
    prefix = generate_prefix(region, year, month, day, hour)
    total_files = 0

    try:
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
        if "Contents" not in response:
            logger.warning(f"No files found in S3 bucket {s3_bucket} with prefix {prefix}.")
            return 0

        for obj in response["Contents"]:
            s3_key = obj["Key"]
            file_name = Path(s3_key).name
            variable_name = file_name.split("-")[-1].replace(".nc", "")

            if variable_name not in required_files:
                continue

            local_file_path = raw_dir / file_name
            logger.info(f"Downloading {s3_key} to {local_file_path}.")
            s3.download_file(s3_bucket, s3_key, str(local_file_path))
            total_files += 1

    except Exception as e:
        logger.error(f"Error fetching Met Office data: {e}")
        raise

    return total_files


def process_met_office_data(
    year: int,
    month: int,
    day: int,
    hour: int,
    region: str,
    overwrite: bool = False,
    archive_type: str = "zarr.zip",
):
    """
    Fetch, convert, and upload Met Office data.

    Args:
        year (int): Year of data.
        month (int): Month of data.
        day (int): Day of data.
        hour (int): Hour of data.
        region (str): Region ('uk' or 'global').
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        archive_type (str): Type of archive to create ("zarr.zip" or "tar").
    """
    config_path = CONFIG_PATHS[region]
    config = load_config(config_path)
    local_output_dir = config["input_data"]["nwp"]["met_office"]["local_output_dir"]

    raw_dir = (
        Path(PROJECT_BASE) / local_output_dir / "raw" / f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    )
    zarr_dir = (
        Path(PROJECT_BASE) / local_output_dir / "zarr" / f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    )

    # Step 1: Fetch data
    if not raw_dir.exists():
        total_files = fetch_met_office_data(year, month, day, hour, region)
        if total_files == 0:
            logger.warning("No files downloaded. Exiting process.")
            return

    # Step 2: Convert to Zarr
    if not zarr_dir.exists():
        converted_files, _ = convert_nc_to_zarr(raw_dir, zarr_dir, overwrite=overwrite)
        if converted_files == 0:
            logger.warning("No files converted to Zarr. Exiting process.")
            return

    # Step 3: Upload Zarr directory
    try:
        upload_to_huggingface(config_path, zarr_dir.name, year, month, day, overwrite, archive_type)
        logger.info("Upload to Hugging Face completed.")
        shutil.rmtree(raw_dir)
        shutil.rmtree(zarr_dir)
        logger.info("Temporary directories cleaned up.")
    except Exception as e:
        logger.error(f"Error during upload: {e}")
