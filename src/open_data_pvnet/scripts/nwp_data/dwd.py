import logging
from pathlib import Path
import shutil
import requests
import xarray as xr
import bz2
import os
from urllib.parse import urljoin
from html.parser import HTMLParser

from open_data_pvnet.utils.env_loader import PROJECT_BASE
from open_data_pvnet.utils.config_loader import load_config
from open_data_pvnet.utils.data_uploader import upload_to_huggingface

logger = logging.getLogger(__name__)

CONFIG_PATH = PROJECT_BASE / "src/open_data_pvnet/configs/dwd_data_config.yaml"

class DWDHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr in attrs:
                if attr[0] == 'href':
                    self.links.append(attr[1])

    def error(self, message):
        pass  # Override to prevent errors from being raised

def generate_variable_url(variable: str, year: int, month: int, day: int, hour: int) -> str:
    """
    Generate the URL for a specific variable and forecast time.

    Args:
        variable (str): Variable name (e.g., 't_2m', 'clct')
        year (int): Year (e.g., 2023)
        month (int): Month (e.g., 1)
        day (int): Day (e.g., 1)
        hour (int): Hour in 24-hour format (e.g., 0 for 00:00Z)

    Returns:
        str: The URL for the specified variable and time
    """
    base_url = "https://opendata.dwd.de/weather/nwp/icon-eu/grib"
    timestamp = f"{year:04d}{month:02d}{day:02d}{hour:02d}"
    return f"{base_url}/{hour:02d}/{variable.lower()}/icon-eu_europe_regular-lat-lon_single-level_{timestamp}_*"

def decompress_bz2(input_path: Path, output_path: Path):
    """
    Decompress a bz2 file.

    Args:
        input_path (Path): Path to the bz2 file
        output_path (Path): Path to save the decompressed file
    """
    with bz2.open(input_path, 'rb') as source, open(output_path, 'wb') as dest:
        dest.write(source.read())

def fetch_dwd_data(year: int, month: int, day: int, hour: int):
    """
    Fetch DWD ICON-EU NWP data for the specified year, month, day, and hour.

    Args:
        year (int): Year to fetch data for
        month (int): Month to fetch data for
        day (int): Day to fetch data for
        hour (int): Hour of data in 24-hour format
    """
    config = load_config(CONFIG_PATH)
    nwp_channels = set(config["input_data"]["nwp"]["dwd"]["nwp_channels"])
    nwp_accum_channels = set(config["input_data"]["nwp"]["dwd"]["nwp_accum_channels"])
    required_files = nwp_channels | nwp_accum_channels

    raw_dir = (
        Path(PROJECT_BASE)
        / config["input_data"]["nwp"]["dwd"]["local_output_dir"]
        / "raw"
        / f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0

    try:
        # Fetch each required variable
        for variable in required_files:
            # Convert variable name to lowercase for URL
            var_lower = variable.lower()
            base_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{hour:02d}/{var_lower}/"
            logger.info(f"Checking DWD data for variable {variable} at {base_url}")

            # First, check if the directory exists
            response = requests.head(base_url)
            if response.status_code == 404:
                logger.warning(f"Directory not found for variable {variable} at {base_url}")
                continue

            # Get the list of available files
            response = requests.get(base_url)
            response.raise_for_status()

            # Parse the HTML to find the correct file
            parser = DWDHTMLParser()
            parser.feed(response.text)
            logger.debug(f"HTML content: {response.text}")
            links = parser.links
            logger.debug(f"Extracted links: {links}")
            
            timestamp = f"{year:04d}{month:02d}{day:02d}{hour:02d}"
            target_prefix = f"icon-eu_europe_regular-lat-lon_single-level_{timestamp}"
            
            found_file = False
            for href in links:
                if not href or not href.startswith(target_prefix):
                    continue

                file_url = urljoin(base_url, href)
                compressed_file = raw_dir / f"{variable}_{href}"
                decompressed_file = raw_dir / f"{variable}_{href.replace('.bz2', '')}"

                logger.info(f"Downloading {file_url} to {compressed_file}")
                file_response = requests.get(file_url, stream=True)
                file_response.raise_for_status()

                with open(compressed_file, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logger.info(f"Decompressing {compressed_file} to {decompressed_file}")
                decompress_bz2(compressed_file, decompressed_file)
                os.remove(compressed_file)  # Remove compressed file after decompression
                total_files += 1
                found_file = True
                break  # Found the file we want, no need to check other links

            if not found_file:
                logger.warning(f"No matching file found for variable {variable} at {base_url}")

    except Exception as e:
        logger.error(f"Error fetching DWD data: {e}")
        raise

    return total_files

def process_dwd_data(
    year: int,
    month: int,
    day: int,
    hour: int,
    overwrite: bool = False,
    archive_type: str = "zarr.zip",
    skip_upload: bool = True,  # Skip upload by default until HF token is set
):
    """
    Fetch, convert, and upload DWD ICON-EU data.

    Args:
        year (int): Year of data
        month (int): Month of data
        day (int): Day of data
        hour (int): Hour of data
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        archive_type (str): Type of archive to create ("zarr.zip" or "tar")
        skip_upload (bool): Whether to skip uploading to Hugging Face. Defaults to True.
    """
    config = load_config(CONFIG_PATH)
    local_output_dir = config["input_data"]["nwp"]["dwd"]["local_output_dir"]

    raw_dir = (
        Path(PROJECT_BASE) / local_output_dir / "raw" / f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    )
    zarr_dir = (
        Path(PROJECT_BASE) / local_output_dir / "zarr" / f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    )

    # Step 1: Fetch data
    if not raw_dir.exists() or overwrite:
        total_files = fetch_dwd_data(year, month, day, hour)
        if total_files == 0:
            logger.warning("No files downloaded. Exiting process.")
            return

    # Step 2: Convert GRIB2 files to Zarr
    if not zarr_dir.exists() or overwrite:
        zarr_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all GRIB2 files and combine them
        datasets = []
        for grib_file in raw_dir.glob("*.grib2"):
            try:
                ds = xr.open_dataset(grib_file, engine='cfgrib')
                variable_name = grib_file.stem.split("_")[0]  # Get variable name from our filename
                # Rename the main variable to match the filename
                main_var = list(ds.data_vars)[0]
                ds = ds.rename({main_var: variable_name})
                datasets.append(ds)
            except Exception as e:
                logger.error(f"Error reading {grib_file}: {e}")
                continue

        if not datasets:
            logger.warning("No valid GRIB2 files found. Exiting process.")
            return

        # Merge all datasets
        combined_ds = xr.merge(datasets)
        
        # Save to zarr format
        logger.info(f"Saving combined dataset to {zarr_dir}")
        combined_ds.to_zarr(zarr_dir, mode='w')

    # Step 3: Upload Zarr directory (optional)
    if not skip_upload:
        try:
            upload_to_huggingface(CONFIG_PATH, zarr_dir.name, year, month, day, overwrite, archive_type)
            logger.info("Upload to Hugging Face completed.")
            shutil.rmtree(raw_dir)
            shutil.rmtree(zarr_dir)
            logger.info("Temporary directories cleaned up.")
        except Exception as e:
            logger.error(f"Error during upload: {e}")
            raise
    else:
        logger.info("Skipping upload to Hugging Face (skip_upload=True)")
        logger.info(f"Data is available in {zarr_dir}")
