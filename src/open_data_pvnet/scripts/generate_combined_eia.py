"""
Generate Combined EIA Data Script

This script fetches EIA data for all US Balancing Authorities (BAs) and combines them into a single Zarr dataset,
matching the format required by ocf-data-sampler (equivalent to UK's generate_combined_gsp.py).

Usage:
    python src/open_data_pvnet/scripts/generate_combined_eia.py --start-year 2020 --end-year 2024 --output-folder data

Requirements:
    - EIA_API_KEY environment variable
    - pandas
    - xarray
    - zarr
    - typer

The script will:
1. Fetch data for all default BAs from EIA API (or specified BAs)
2. Preprocess data to match ocf-data-sampler format (ba_id, datetime_gmt)
3. Add capacity estimates
4. Convert to xarray Dataset and save as Zarr format
5. Output file: combined_eia_{start_date}_{end_date}.zarr

Note: This script combines collection and preprocessing into a single step, matching the UK pattern.
"""

import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
from typing import Optional, List
import pytz
import os
import typer
import logging
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from open_data_pvnet.scripts.fetch_eia_data import EIAData
from open_data_pvnet.scripts.preprocess_eia_for_sampler import (
    create_ba_mapping,
    estimate_capacity_from_generation,
)
from open_data_pvnet.utils.env_loader import load_environment_variables

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Major US ISOs/RTOs
DEFAULT_BAS = [
    'CISO',  # CAISO
    'ERCO',  # ERCOT
    'PJM',   # PJM
    'MISO',  # MISO
    'NYIS',  # NYISO
    'ISNE',  # ISO-NE
    'SWPP',  # SPP
]

# BA Centroids (Approximate)
BA_CENTROIDS = {
    'CISO': {'latitude': 37.0, 'longitude': -120.0},
    'ERCO': {'latitude': 31.0, 'longitude': -99.0},
    'PJM': {'latitude': 40.0, 'longitude': -77.0},
    'MISO': {'latitude': 40.0, 'longitude': -90.0},
    'NYIS': {'latitude': 43.0, 'longitude': -75.0},
    'ISNE': {'latitude': 44.0, 'longitude': -71.0},
    'SWPP': {'latitude': 38.0, 'longitude': -98.0},
}


def main(
    start_year: int = typer.Option(2020, help="Start year for data collection"),
    end_year: int = typer.Option(2025, help="End year for data collection"),
    output_folder: str = typer.Option("data", help="Output folder for the zarr dataset"),
    bas: Optional[List[str]] = typer.Option(None, help="List of BA codes (default: all major ISOs)"),
    capacity_method: str = typer.Option("estimate", help="Method for capacity data (estimate/file/static)"),
):
    """
    Generate combined EIA data for all BAs and save as a zarr dataset.
    
    This matches the UK generate_combined_gsp.py pattern but for US EIA data.
    """
    try:
        load_environment_variables()
    except Exception as e:
        logger.warning(f"Could not load environment variables: {e}")

    range_start = datetime(start_year, 1, 1, tzinfo=pytz.UTC)
    range_end = datetime(end_year, 1, 1, tzinfo=pytz.UTC)

    # Use default BAs if not specified
    if bas is None:
        bas = DEFAULT_BAS

    eia = EIAData()
    if not eia.api_key:
        logger.error("EIA_API_KEY not set. Exiting.")
        raise typer.Exit(code=1)

    logger.info(f"Fetching EIA data from {range_start.date()} to {range_end.date()} for BAs: {bas}")

    # Fetch data for all BAs
    df = eia.get_hourly_solar_data(
        start_date=range_start.strftime("%Y-%m-%d"),
        end_date=range_end.strftime("%Y-%m-%d"),
        ba_codes=bas
    )

    if df.empty:
        logger.error("No data retrieved for any BAs - terminating")
        raise typer.Exit(code=1)

    logger.info(f"Fetched {len(df)} rows for {df['ba_code'].nunique()} BAs")

    # Add coordinates
    df["latitude"] = df["ba_code"].map(lambda x: BA_CENTROIDS.get(x, {}).get('latitude', np.nan))
    df["longitude"] = df["ba_code"].map(lambda x: BA_CENTROIDS.get(x, {}).get('longitude', np.nan))

    # Rename timestamp to datetime_gmt and ensure proper format
    if "timestamp" in df.columns:
        df["datetime_gmt"] = pd.to_datetime(df["timestamp"], utc=True)
        df["datetime_gmt"] = df["datetime_gmt"].dt.tz_convert(None)
        df = df.drop(columns=["timestamp"])
    elif "datetime_gmt" not in df.columns:
        logger.error("No timestamp or datetime_gmt column found")
        raise typer.Exit(code=1)

    # Ensure generation_mw is numeric
    if "generation_mw" in df.columns:
        df["generation_mw"] = pd.to_numeric(df["generation_mw"], errors="coerce")
    else:
        logger.error("No generation_mw column found")
        raise typer.Exit(code=1)

    # Create BA mapping (ba_code -> ba_id)
    ba_to_id, metadata = create_ba_mapping(df)
    df["ba_id"] = df["ba_code"].map(ba_to_id)

    # Handle capacity data
    if capacity_method == "estimate":
        logger.info("Estimating capacity from maximum generation")
        capacity_estimates = estimate_capacity_from_generation(df)
        df["capacity_mwp"] = df["ba_code"].map(capacity_estimates)
    else:
        logger.warning(f"Capacity method '{capacity_method}' not fully implemented in this script")
        # Fallback to estimate
        capacity_estimates = estimate_capacity_from_generation(df)
        df["capacity_mwp"] = df["ba_code"].map(capacity_estimates)

    # Validate capacity data
    if df["capacity_mwp"].isna().any():
        logger.warning("Some BAs have missing capacity data, filling with estimates")
        missing_bas = df[df["capacity_mwp"].isna()]["ba_code"].unique()
        for ba in missing_bas:
            ba_gen = df[df["ba_code"] == ba]["generation_mw"].max()
            estimated_capacity = ba_gen * 1.15 if not pd.isna(ba_gen) else 100.0
            df.loc[df["ba_code"] == ba, "capacity_mwp"] = estimated_capacity

    # Ensure capacity >= generation (with small tolerance)
    df["capacity_mwp"] = df[["capacity_mwp", "generation_mw"]].max(axis=1) * 1.01

    # Select and reorder columns
    columns_to_keep = ["ba_id", "datetime_gmt", "generation_mw", "capacity_mwp"]
    if "ba_code" in df.columns:
        columns_to_keep.append("ba_code")
    if "ba_name" in df.columns:
        columns_to_keep.append("ba_name")
    if "latitude" in df.columns:
        columns_to_keep.append("latitude")
    if "longitude" in df.columns:
        columns_to_keep.append("longitude")

    df_processed = df[columns_to_keep].copy()

    # Set index to match UK format: (ba_id, datetime_gmt)
    df_processed = df_processed.set_index(["ba_id", "datetime_gmt"])

    # Convert to xarray Dataset
    ds_processed = xr.Dataset.from_dataframe(df_processed)

    # Ensure datetime_gmt is datetime64[ns] (no timezone)
    if "datetime_gmt" in ds_processed.coords:
        ds_processed.coords["datetime_gmt"] = ds_processed.coords["datetime_gmt"].astype(np.datetime64)

    # Apply chunking like UK implementation
    ds_processed = ds_processed.chunk({"ba_id": 1, "datetime_gmt": 1000})

    # Save to Zarr
    os.makedirs(output_folder, exist_ok=True)
    filename = f"combined_eia_{range_start.date()}_{range_end.date()}.zarr"
    output_path = os.path.join(output_folder, filename)
    ds_processed.to_zarr(output_path, mode="w", consolidated=True)

    logger.info(f"Successfully saved combined EIA dataset to {output_path}")
    logger.info(f"Dataset contains {len(ba_to_id)} BAs for period {range_start.date()} to {range_end.date()}")
    
    # Also save metadata CSV
    metadata_path = os.path.join(output_folder, f"us_ba_metadata_{range_start.date()}_{range_end.date()}.csv")
    metadata.to_csv(metadata_path, index=False)
    logger.info(f"BA metadata saved to {metadata_path}")


if __name__ == "__main__":
    typer.run(main)

