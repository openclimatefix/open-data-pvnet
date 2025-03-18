import logging
from pathlib import Path
import shutil
import requests
import xarray as xr
import bz2
import os
from urllib.parse import urljoin
from html.parser import HTMLParser
from typing import List, Dict

from open_data_pvnet.utils.env_loader import PROJECT_BASE
from open_data_pvnet.utils.config_loader import load_config

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

def process_dwd_data(file_path: str, config: dict) -> xr.Dataset:
    """Process a single DWD GRIB2 file.

    Args:
        file_path: Path to the GRIB2 file
        config: Configuration dictionary

    Returns:
        xr.Dataset: Processed dataset
    """
    # Define variable name mapping from DWD GRIB to our format
    var_name_mapping = {
        'T_2M': 't_2m',
        't2m': 't_2m',
        'T_G': 't_g',
        'ASWDIFD_S': 'aswdifd_s',
        'ASWDIR_S': 'aswdir_s',
        'RELHUM_2M': 'relhum_2m',
        'r2': 'relhum_2m',
        'TD_2M': 'td_2m',
        'd2m': 'td_2m',
        'U_10M': 'u_10m',
        'u10': 'u_10m',
        'V_10M': 'v_10m',
        'v10': 'v_10m',
        'VMAX_10M': 'vmax_10m',
        'fg10': 'vmax_10m',
        'PMSL': 'pmsl',
        'prmsl': 'pmsl',
        'CLCT': 'clct',
        'CLCL': 'clcl',
        'CLCM': 'clcm',
        'CLCH': 'clch',
        'WW': 'ww',
        'Z0': 'z0',
        'fsr': 'z0',
        'H_SNOW': 'h_snow',
        'sde': 'h_snow',
        'W_SNOW': 'w_snow',
        'sd': 'w_snow',
        'TOT_PREC': 'tot_prec',
        'tp': 'tot_prec',
        'RUNOFF_G': 'runoff_g',
        'RUNOFF_S': 'runoff_s',
        'CAPE_CON': 'cape_con',
        'ALB_RAD': 'alb_rad',
        'al': 'alb_rad'
    }

    try:
        ds = xr.open_dataset(file_path, engine='cfgrib', backend_kwargs={'indexpath': ''})
        
        # Drop unnecessary coordinates
        coords_to_drop = [
            'isobaricLayer', 
            'heightAboveGround', 
            'level', 
            'step', 
            'surface', 
            'valid_time',
            'depthBelowLandLayer'  # Add this to the list of coordinates to drop
        ]
        for coord in coords_to_drop:
            if coord in ds.coords:
                ds = ds.drop(coord)

        # Get the main variable name (should be only one)
        var_name = list(ds.data_vars.keys())[0]
        
        # Map variable name if it exists in mapping
        if var_name in var_name_mapping:
            ds = ds.rename({var_name: var_name_mapping[var_name]})
            var_name = var_name_mapping[var_name]
        elif var_name.upper() in var_name_mapping:
            ds = ds.rename({var_name: var_name_mapping[var_name.upper()]})
            var_name = var_name_mapping[var_name.upper()]
        else:
            logging.warning(f"Variable {var_name} not found in mapping, using as is")

        # Get grid bounds from config
        lat_bounds = config['grid']['latitude_bounds']
        lon_bounds = config['grid']['longitude_bounds']
        expected_height = config['grid']['height']
        expected_width = config['grid']['width']

        # Handle longitude wrapping (convert 0-360 to -180-180 if needed)
        if ds.longitude.max() > 180:
            ds.coords['longitude'] = (ds.coords['longitude'] + 180) % 360 - 180
            ds = ds.sortby('longitude')

        # Crop to grid bounds
        ds = ds.sel(
            latitude=slice(lat_bounds[1], lat_bounds[0]),  # Note: Reversed for descending order
            longitude=slice(lon_bounds[0], lon_bounds[1])
        )

        # Verify dimensions
        if ds.sizes['latitude'] != expected_height or ds.sizes['longitude'] != expected_width:
            raise ValueError(f"Dataset dimensions {ds.sizes} do not match expected dimensions (latitude: {expected_height}, longitude: {expected_width})")

        return ds

    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        raise

def fetch_and_process_dwd_data(
    year: int,
    month: int,
    day: int,
    hour: int,
    config: dict,
    overwrite: bool = False
) -> str:
    """Fetch and process DWD data.

    Args:
        year: Year
        month: Month
        day: Day
        hour: Hour
        config: Configuration dictionary
        overwrite: Whether to overwrite existing data

    Returns:
        str: Path to processed data
    """
    # Setup paths
    date_str = f"{year:04d}-{month:02d}-{day:02d}-{hour:02d}"
    raw_dir = os.path.join("tmp", "dwd", "raw", date_str)
    zarr_dir = os.path.join("tmp", "dwd", "zarr", date_str)

    # Clean up if overwrite is True
    if overwrite:
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)
        if os.path.exists(zarr_dir):
            shutil.rmtree(zarr_dir)

    # Create directories
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(zarr_dir, exist_ok=True)

    # Get required variables from config
    required_vars = set(config['variables'])

    # Download and process data
    processed_datasets = []
    processed_vars = set()
    
    for var in required_vars:
        try:
            # Download data
            file_path = download_dwd_data(var, year, month, day, hour, raw_dir)
            if file_path:
                # Process data
                ds = process_dwd_data(file_path, config)
                processed_datasets.append(ds)
                processed_vars.add(var)
                logging.info(f"Processed {var} from {file_path}")
        except Exception as e:
            logging.error(f"Error processing variable {var}: {str(e)}")

    if not processed_datasets:
        raise ValueError("No datasets were processed successfully")

    # Check for missing variables
    missing_vars = required_vars - processed_vars
    if missing_vars:
        logging.warning(f"Missing required variables: {missing_vars}")

    # Combine datasets with compat='override' to handle conflicting coordinates
    combined_ds = xr.merge(processed_datasets, compat='override')

    # Save to zarr
    logging.info(f"Saving combined dataset to {zarr_dir}")
    combined_ds.to_zarr(zarr_dir, mode='w')

    logging.info(f"Data is available in {zarr_dir}")
    return zarr_dir

def download_dwd_data(variable: str, year: int, month: int, day: int, hour: int, raw_dir: str) -> str:
    """Download DWD data for a specific variable.

    Args:
        variable: Variable name
        year: Year
        month: Month
        day: Day
        hour: Hour
        raw_dir: Directory to save raw data

    Returns:
        str: Path to downloaded file, or None if download failed
    """
    os.makedirs(raw_dir, exist_ok=True)
    
    # Convert variable name to lowercase for URL
    var_lower = variable.lower()
    base_url = f"https://opendata.dwd.de/weather/nwp/icon-eu/grib/{hour:02d}/{var_lower}/"
    logging.info(f"Checking DWD data for variable {variable} at {base_url}")

    # First, check if the directory exists
    response = requests.head(base_url)
    if response.status_code == 404:
        logging.warning(f"Directory not found for variable {variable} at {base_url}")
        return None

    # Get the list of available files
    response = requests.get(base_url)
    response.raise_for_status()

    # Parse the HTML to find the correct file
    parser = DWDHTMLParser()
    parser.feed(response.text)
    links = parser.links
    
    timestamp = f"{year:04d}{month:02d}{day:02d}{hour:02d}"
    target_prefix = f"icon-eu_europe_regular-lat-lon_single-level_{timestamp}"
    
    for href in links:
        if not href or not href.startswith(target_prefix):
            continue

        file_url = urljoin(base_url, href)
        compressed_file = os.path.join(raw_dir, f"{var_lower}_icon-eu_europe_regular-lat-lon_single-level_{timestamp}_000_{var_lower.upper()}.grib2.bz2")
        decompressed_file = compressed_file.replace('.bz2', '')

        logging.info(f"Downloading {file_url} to {compressed_file}")
        file_response = requests.get(file_url, stream=True)
        file_response.raise_for_status()

        with open(compressed_file, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"Decompressing {compressed_file} to {decompressed_file}")
        decompress_bz2(Path(compressed_file), Path(decompressed_file))
        os.remove(compressed_file)  # Remove compressed file after decompression
        return decompressed_file

    logging.warning(f"No matching file found for variable {variable} at {base_url}")
    return None
