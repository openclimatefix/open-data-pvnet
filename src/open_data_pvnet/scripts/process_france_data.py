
import pandas as pd
import xarray as xr
import os
import logging
import numpy as np
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

base_dir = os.getcwd()
metadata_file_dir = os.path.join(os.path.dirname(base_dir), "configs")
parent_3_levels_up = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
generation_data_dir = os.path.join(parent_3_levels_up, "tmp")
output_dir = os.path.join(parent_3_levels_up, "data")
start_yr = 2020
end_yr = 2024

def process_location_csv(generation_data_dir, region, year) -> pd.DataFrame:
    """Process the location CSV for a given region and year.
    Args:
        generation_data_dir (str): The directory where the generation CSV files are stored.
        region (str): The name of the region.
        year (int): The year for which to process the data.
    """
    df = pd.read_csv(
        os.path.join(generation_data_dir, f"eCO2mix_RTE_{region}_Annuel_{year}.csv"),
        low_memory=False,
    )
    # Keep only solar related columns and date/time
    df = df[["Date", "Heures", "Solaire", "TCH Solaire (%)"]]

    # generate a datetime column from the date and time column
    # Convert all time to UTC
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Heures"])

    # Handle DST transitions
    # - nonexistent (spring): Mark as NaT and drop
    # - ambiguous (fall): Keep first occurrence (DST/summer time)
    df["datetime"] = df["datetime"].dt.tz_localize(
        "Europe/Paris", ambiguous=True, nonexistent="NaT"
    )

    # Drop any NaT values created during spring forward transition
    df = df.dropna(subset=["datetime"])

    # Convert to UTC
    df["datetime"] = df["datetime"].dt.tz_convert("UTC")

    # Create capacity column from TCH
    df["capacity_mwp"] = (
        ((df["Solaire"] / df["TCH Solaire (%)"]) * 100).replace([np.inf, -np.inf], np.nan).round(1)
    )
    # keep only every 0 and 30 minute data of each hour as only these are filled with data
    df = df[df["datetime"].dt.minute.isin([0, 30])]

    # Ensure time is monotonic and no duplicate
    # Check if monotonic increasing
    if not df["datetime"].is_monotonic_increasing:
        logger.warning("Datetime column is not monotonic increasing. Sorting by datetime.")
        df = df.sort_values("datetime")
    else:
        logger.info("Datetime is monotonic increasing.")

    # Check for duplicates
    duplicates = df[df.duplicated(subset=["datetime"], keep=False)]
    if len(duplicates) > 0:
        duplicate_times = df[df.duplicated(subset=["datetime"], keep="first")]["datetime"].unique()
        logger.warning(f"Found {len(duplicates)} duplicate timestamps:")
        for dt in duplicate_times:
            logger.warning(f"  Duplicate: {dt}")
        df = df.drop_duplicates(subset=["datetime"], keep="first")
    else:
        logger.info("No duplicate timestamps found.")

    # Ensure all 30-minute timesteps are present
    df = df.set_index("datetime")
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="30min")
    missing_timestamps = full_range.difference(df.index)

    if len(missing_timestamps) > 0:
        logger.warning(f"Found {len(missing_timestamps)} missing 30-minute timesteps:")
        for dt in missing_timestamps[:20]:  # Show first 20 to avoid excessive logging
            logger.warning(f"  Missing: {dt}")
        if len(missing_timestamps) > 20:
            logger.warning(f"  ... and {len(missing_timestamps) - 20} more missing timesteps")
        df = df.reindex(full_range)
        df.index.name = "datetime"
    else:
        logger.info("All 30-minute timesteps are present.")

    # Reset index to make datetime a column again
    df = df.reset_index()

    # rename solaiure
    df = df.rename(columns={"Solaire": "generation_mw", "TCH Solaire (%)": "tch_solaire_percent"})
    df.drop(columns=["Date", "Heures"], inplace=True)
    # Forward-fill then backward-fill solar capacity at nighttimes
    df["capacity_mwp"] = (
        ((df["generation_mw"] / df["tch_solaire_percent"]) * 100)
        .replace([np.inf, -np.inf], np.nan)
        .round(1)
    )
    df["capacity_mwp"] = df["capacity_mwp"].ffill().bfill()
    df.set_index("datetime", inplace=True)

    return df





# Create a France wide aggregate
def create_France_aggregate(generation_data_dir, region_list, year_list) -> pd.DataFrame:
    """Create a France-wide aggregate DataFrame from regional generation data.
    Args:
        generation_data_dir (str): The directory where the generation CSV files are stored.
        region_list (list): List of regions to include in the aggregate.
        year_list (list): List of years to include in the aggregate.
    """
    france_df = None

    for region in region_list:
        for year in year_list:
            logger.info(f"Processing {region} {year}")
            df = process_location_csv(generation_data_dir, region, year)
            df["region"] = region  # Add region column for later aggregation

            if france_df is None:
                france_df = df
            else:
                france_df = pd.concat([france_df, df], ignore_index=True)

    # Now we have a DataFrame with all regions and years. We can create an aggregate by summing generation and capacity across regions for each timestamp.
    france_aggregate = (
        france_df.groupby("datetime")
        .agg({"generation_mw": "sum", "capacity_mwp": "sum"})
        .reset_index()
    )

    return france_aggregate


def create_xarray_dataset(
    generation_data_dir,
    region_list,
    year_list,
    metadata_file=f"{metadata_file_dir}/admin_region_lat_lon.csv",
) -> xr.Dataset:
    """Create an xarray Dataset combining all regions with their lat/lon coordinates.

    Args:
        generation_data_dir (str): The directory where the generation CSV files are stored.
        region_list (list): List of regions to include.
        year_list (list): List of years to include.
        metadata_file (str): Path to CSV file with columns: region, latitude, longitude

    Returns:
        xr.Dataset: Dataset with dimensions (location_id, time_utc)
    """
    # Load metadata - if not absolute path, assume it's relative to generation_data_dir parent
    if not os.path.isabs(metadata_file):
        metadata_path = os.path.join(os.path.dirname(generation_data_dir), metadata_file)
    else:
        metadata_path = metadata_file

    metadata = pd.read_csv(metadata_path)
    metadata = metadata.set_index("region")

    # Collect data for all regions
    all_data = []
    location_ids = []
    latitudes = []
    longitudes = []

    for region in region_list:
        logger.info(f"Processing region: {region}")

        # Collect data across all years for this region
        region_dfs = []
        for year in year_list:
            try:
                df = process_location_csv(generation_data_dir, region, year)
                region_dfs.append(df)
            except FileNotFoundError:
                logger.warning(f"File not found for {region} {year}, skipping")
                continue

        # Concatenate all years for this region
        if region_dfs:
            region_df = pd.concat(region_dfs)
            region_df = region_df.sort_index()  # Sort by datetime

            all_data.append(region_df)
            location_ids.append(region)
            latitudes.append(metadata.loc[region, "latitude"])
            longitudes.append(metadata.loc[region, "longitude"])

    # Create common time index (union of all timestamps)
    all_times = pd.DatetimeIndex([])
    for df in all_data:
        all_times = all_times.union(df.index)
    all_times = all_times.sort_values()

    # Create a complete 30-minute grid to ensure no gaps
    min_time = all_times.min()
    max_time = all_times.max()
    complete_range = pd.date_range(start=min_time, end=max_time, freq="30min")

    logger.info(f"Original union has {len(all_times)} timestamps")
    logger.info(f"Complete 30-min grid has {len(complete_range)} timestamps")
    logger.info(f"Missing {len(complete_range) - len(all_times)} timestamps in union")

    # Use complete range instead of union
    all_times = complete_range

    # Convert to timezone-naive datetime64[ns] for zarr serialization
    if hasattr(all_times, "tz") and all_times.tz is not None:
        all_times = all_times.tz_localize(None)
    elif isinstance(all_times, pd.Index):
        # If it's a regular Index, convert to DatetimeIndex and remove timezone
        all_times = pd.DatetimeIndex(all_times).tz_localize(None)

    # Reindex all dataframes to common time index
    generation_data = []
    capacity_data = []

    for df in all_data:
        # Remove timezone from dataframe index to match all_times
        df_naive = df.copy()
        df_naive.index = df_naive.index.tz_localize(None)

        df_reindexed = df_naive.reindex(all_times)
        generation_data.append(df_reindexed["generation_mw"].values)
        capacity_data.append(df_reindexed["capacity_mwp"].values)

    # Stack into 2D arrays (location, time)
    generation_array = np.stack(generation_data, axis=0)
    capacity_array = np.stack(capacity_data, axis=0)

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            "generation_mw": (["location_id", "time_utc"], generation_array),
            "capacity_mwp": (["location_id", "time_utc"], capacity_array),
        },
        coords={
            "location_id": location_ids,
            "time_utc": all_times,
            "latitude": ("location_id", latitudes),
            "longitude": ("location_id", longitudes),
        },
    )

    # Add attributes
    ds.attrs["description"] = (
        "France Réseau de Transport d’Électricité (RTE) solar generation data for PVNet training"
    )
    ds.attrs["source"] = "https://www.rte-france.com/eco2mix"
    ds.attrs["schema"] = "ocf-data-sampler generation format"
    ds.attrs["time_resolution"] = "0.5 hour"
    ds.attrs["date_range"] = f"{all_times.min().isoformat()} to {all_times.max().isoformat()}"
    ds.attrs["created"] = pd.Timestamp.now().isoformat()

    ds["generation_mw"].attrs["units"] = "MW"
    ds["generation_mw"].attrs["long_name"] = "Solar generation"

    ds["capacity_mwp"].attrs["units"] = "MWp"
    ds["capacity_mwp"].attrs["long_name"] = "Solar capacity"

    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process France RTE solar generation data and create xarray dataset"
    )
    parser.add_argument(
        "--generation-data-dir",
        type=str,
        default="downloads",
        help="Directory containing the generation CSV files (default: downloads)",
    )
    parser.add_argument(
        "--start-yr", type=int, default=2020, help="Start year for data processing (default: 2020)"
    )
    parser.add_argument(
        "--end-yr", type=int, default=2024, help="End year for data processing (default: 2024)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="france_solar_combined.zarr",
        help="Output file name (default: france_solar_combined.zarr)",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=f"{metadata_file_dir}/admin_region_lat_lon.csv",
        help="Metadata CSV file with region lat/lon (default: configs/admin_region_lat_lon.csv)",
    )

    args = parser.parse_args()

    # Convert to absolute path if relative
    if not os.path.isabs(args.generation_data_dir):
        generation_data_dir = os.path.join(os.getcwd(), args.generation_data_dir)
    else:
        generation_data_dir = args.generation_data_dir

    # Define regions and years
    admin_region_list = [
        "Auvergne-Rhône-Alpes",
        "Bourgogne-Franche-Comté",
        "Bretagne",
        "Centre-Val-de-Loire",
        "Grand-Est",
        "Hauts-de-France",
        "Ile-de-France",
        "Normandie",
        "Nouvelle-Aquitaine",
        "Occitanie",
        "Pays-de-la-Loire",
        "PACA",
    ]

    year_list = list(range(args.start_yr, args.end_yr + 1))

    logger.info(f"Processing data from {generation_data_dir}")
    logger.info(f"Years: {args.start_yr} to {args.end_yr}")

    # Create xarray dataset
    ds = create_xarray_dataset(
        generation_data_dir, admin_region_list, year_list, args.metadata_file
    )

    # Save to Zarr
    output_file = os.path.join(output_dir, args.output)
    ds.to_zarr(output_file, mode="w")
    logger.info(f"Saved dataset to {output_file}")

    # Display info
    print(ds)
