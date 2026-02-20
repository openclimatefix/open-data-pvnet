"""
France PVNet Data Download Script

This script downloads and process mainland France solar generation data from RTE's éCO2mix platform for PVNet training.

Data source:
RTE éCO2mix Dataset: https://www.rte-france.com/en/data-publications/eco2mix/download-indicators
    - Half Hourly data for the 12 administrative regions of France, from Jan 2020 to Dec 2023 (definitive data)
    - Consolidated data for Jan to Dec 2024 (in-progress data)
    - Capacity (TCH) data available from Jan 2020

Usage:
    python get_generation_csv.py --start_yr 2020 --end_yr 2023 --consolidate_yr 2024
    # where users need to determine the consolidate year for assignment of a year in the file name
    # based on the latest available data on RTE. This way filenames will be consistent with the year of data they contain.
"""

import requests
import pandas as pd
import os
import sys
from time import sleep
import zipfile
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Get paths relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))  # .../scripts/fra/
workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))  # 4 levels up
metadata_file_dir = os.path.join(workspace_root, "src", "open_data_pvnet", "configs")
output_dir = os.path.join(workspace_root, "tmp")

# Load admin regions from CSV
# try catching metadata file not found error and log it
try:
    metadata_df = pd.read_csv(os.path.join(metadata_file_dir, "admin_region_lat_lon.csv"))
    admin_region_list = metadata_df["region"].tolist()
except FileNotFoundError:
    logger.error(f"Metadata file not found: {os.path.join(metadata_file_dir, 'admin_region_lat_lon.csv')}")
    sys.exit(1)


def get_region_generation_csv(region, year, consolidated=False) -> None:
    """Download and extract the annual generation CSV for a given region and year.

    Args:
        region (str): The name of the region.
        year (int): The year for which to download the data.
        consolidated (bool): If True, download consolidated (in-progress) data,
                           otherwise download definitive data.
    """
    if consolidated:
        url = f"https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_{region}_En-cours-Consolide.zip"
        data_type = "Consolidated"
    else:
        url = f"https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_{region}_Annuel-Definitif_{year}.zip"
        data_type = "Definitive"

    # Download the ZIP file
    response = requests.get(url)

    # Check if request was successful
    if response.status_code != 200:
        logger.error(
            f"Failed to download {region} {year} ({data_type}): HTTP {response.status_code}"
        )
        return

    # Save to temporary ZIP file
    temp_zip = f"temp_{region}_{year}.zip"
    with open(temp_zip, "wb") as f:
        f.write(response.content)

    # Try to extract the ZIP file
    try:
        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            # Get list of files in the zip
            file_list = zip_ref.namelist()
            # Find the XLS file (assuming there's one XLS file in the zip)
            xls_file = [f for f in file_list if f.endswith(".xls") or f.endswith(".xlsx")][0]
            # Extract just that file
            zip_ref.extract(xls_file)
    except zipfile.BadZipFile:
        logger.warning(f"Skipping {region} {year} ({data_type}): Not a valid ZIP file")
        os.remove(temp_zip)
        return
    except IndexError:
        logger.warning(f"Skipping {region} {year} ({data_type}): No XLS file found in ZIP")
        os.remove(temp_zip)
        return

    # The .xls file is actually tab-separated text, not Excel format
    # Read as CSV with tab delimiter and proper encoding for French characters
    df = pd.read_csv(xls_file, sep="\t", encoding="latin-1", low_memory=False)

    # Save as CSV in downloads subdirectory
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"eCO2mix_RTE_{region}_Annuel_{year}.csv")
    df.to_csv(csv_filename, index=False)

    # Clean up temporary files
    os.remove(temp_zip)
    os.remove(xls_file)

    logger.info(f"Saved {csv_filename}")
    sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download France RTE éCO2mix generation data for specified years"
    )
    parser.add_argument(
        "--start_yr", type=int, help="Start year for definitive data download (default: 2019)"
    )
    parser.add_argument(
        "--end_yr", type=int, help="End year for definitive data download (default: 2023)"
    )
    parser.add_argument(
        "--consolidate_yr",
        type=int,
        help="Year for consolidated (in-progress) data download (default: 2024)",
    )

    args = parser.parse_args()

    year_list = [year for year in range(args.start_yr, args.end_yr + 1)]

    # Run for consolidated data
    for region in admin_region_list:
        get_region_generation_csv(region, args.consolidate_yr, consolidated=True)

    # Run for all regions and definitive years
    for region in admin_region_list:
        for year in year_list:
            get_region_generation_csv(region, year)
