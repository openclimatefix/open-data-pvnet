"""
Generate Combined GSP Data Script

This script fetches PVLive data for all GSPs (0-318) and combines them into a single Zarr dataset.

Usage:
    python src/open_data_pvnet/scripts/generate_combined_gsp.py --start-year 2023 --end-year 2024 --output-folder data

Requirements:
    - pvlive-api
    - pandas
    - xarray
    - zarr
    - typer

The script will:
1. Fetch data for each GSP ID from PVLive API
2. Add gsp_id column to each dataset
3. Combine all datasets into a single DataFrame
4. Convert to xarray Dataset and save as Zarr format
5. Output file: combined_gsp_{start_date}_{end_date}.zarr

Note: Some GSP IDs may not exist and will be skipped with a warning message.
"""

import pandas as pd
import xarray as xr
from datetime import datetime
import pytz
import os
import typer
import logging

from src.open_data_pvnet.scripts.fetch_pvlive_data import PVLiveData

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(
    start_year: int = typer.Option(2020, help="Start year for data collection"),
    end_year: int = typer.Option(2025, help="End year for data collection"),
    output_folder: str = typer.Option("data", help="Output folder for the zarr dataset")
):
    """
    Generate combined GSP data for all GSPs and save as a zarr dataset.
    """
    range_start = datetime(start_year, 1, 1, tzinfo=pytz.UTC)
    range_end = datetime(end_year, 1, 1, tzinfo=pytz.UTC)

    data_source = PVLiveData()

    all_dataframes = []

    # Changed range to start from 0 to include gsp_id=0
    for gsp_id in range(0, 319):  
        logging.info(f"Processing GSP ID {gsp_id}")
        df = data_source.get_data_between(
            start=range_start,
            end=range_end,
            entity_id=gsp_id,
            extra_fields="capacity_mwp,installedcapacity_mwp"
        )
        
        if df is not None and not df.empty:
            # Add gsp_id column to the dataframe
            df["gsp_id"] = gsp_id
            all_dataframes.append(df)
        else:
            logging.warning(f"No data available for GSP ID {gsp_id}")

    # Concatenate all dataframes
    if all_dataframes:
        df_pv = pd.concat(all_dataframes, ignore_index=True)
    else:
        logging.error("No data retrieved for any GSP IDs - terminating")
        return

    df_pv.rename(columns={"datetime": "datetime_gmt"}, inplace=True)
    df_pv["datetime_gmt"] = pd.to_datetime(df_pv["datetime_gmt"], utc=True).dt.tz_convert(None)
    df_pv = df_pv.set_index(["gsp_id", "datetime_gmt"])

    xr_pv = xr.Dataset.from_dataframe(df_pv)
    xr_pv = xr_pv.chunk({"gsp_id": 1, "datetime_gmt": 1000})

    os.makedirs(output_folder, exist_ok=True)
    filename = f"combined_gsp_{range_start.date()}_{range_end.date()}.zarr"
    output_path = os.path.join(output_folder, filename)
    xr_pv.to_zarr(output_path, mode="w", consolidated=True)

    logging.info(f"Successfully saved combined GSP dataset to {output_path}")
    logging.info(f"Dataset contains GSPs 0-318 for period {range_start.date()} to {range_end.date()}")


if __name__ == "__main__":
    typer.run(main)
