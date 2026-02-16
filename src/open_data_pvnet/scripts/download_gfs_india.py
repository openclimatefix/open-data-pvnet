"""
Download and process NOAA GFS data for India region.

Uses Herbie for efficient byte-range downloads (.idx-based) from NOAA S3.
Downloads only the specific variables needed, not full GRIB2 files.
Converts to Zarr format matching the OCF GFS schema used by open-data-pvnet.

Usage:
    # Test with 1 day of data
    python download_gfs_india.py --year 2024 --months 1 --max-days 1

    # Download Jan 2024
    python download_gfs_india.py --year 2024 --months 1

    # Download full year and merge
    python download_gfs_india.py --year 2024 --months 1 2 3 4 5 6 7 8 9 10 11 12 --merge

    # Dry run (verify file availability without downloading)
    python download_gfs_india.py --year 2024 --months 1 --dry-run

Requirements:
    pip install herbie-data xarray cfgrib eccodes numpy pandas zarr s3fs
"""

import argparse
import logging
import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# OCF channel → GFS GRIB search term mapping
# --------------------------------------------------------------------------- #
# Verified against GFS pgrb2.0p25 inventory for 2024 data.
# Each entry: (ocf_channel_name, herbie_search_regex, xarray_var_name)

OCF_CHANNELS = {
    "dlwrf": {
        "search": ":DLWRF:surface",
        "description": "Downward long-wave radiation flux [W/m²]",
    },
    "dswrf": {
        "search": ":DSWRF:surface",
        "description": "Downward short-wave radiation flux [W/m²]",
    },
    "hcc": {
        "search": ":HCDC:high cloud layer:(?!.*ave)",
        "description": "High cloud cover [%]",
    },
    "lcc": {
        "search": ":LCDC:low cloud layer:(?!.*ave)",
        "description": "Low cloud cover [%]",
    },
    "mcc": {
        "search": ":MCDC:middle cloud layer:(?!.*ave)",
        "description": "Medium cloud cover [%]",
    },
    "prate": {
        "search": ":PRATE:surface:(?!.*ave)",
        "description": "Precipitation rate [kg/m²/s]",
    },
    "r": {
        "search": ":RH:850 mb",
        "description": "Relative humidity at 850 hPa [%]",
    },
    "t": {
        "search": ":TMP:2 m above ground",
        "description": "2-metre temperature [K]",
    },
    "tcc": {
        "search": ":TCDC:entire atmosphere:(?!.*ave)",
        "description": "Total cloud cover [%]",
    },
    "u10": {
        "search": ":UGRD:10 m above ground",
        "description": "10-metre U-wind [m/s]",
    },
    "u100": {
        "search": ":UGRD:100 m above ground",
        "description": "100-metre U-wind [m/s]",
    },
    "v10": {
        "search": ":VGRD:10 m above ground",
        "description": "10-metre V-wind [m/s]",
    },
    "v100": {
        "search": ":VGRD:100 m above ground",
        "description": "100-metre V-wind [m/s]",
    },
    "vis": {
        "search": ":VIS:surface",
        "description": "Visibility [m]",
    },
}

# India bounding box with 1° buffer
INDIA_LAT_MIN = 5.0
INDIA_LAT_MAX = 39.0
INDIA_LON_MIN = 67.0
INDIA_LON_MAX = 99.0

# GFS forecast hours to download (17 steps: 0-48h at 3h intervals)
FORECAST_HOURS = list(range(0, 49, 3))

# GFS initialization hours (4x daily)
INIT_HOURS = [0, 6, 12, 18]


def download_single_variable(
    date_str: str,
    init_hour: int,
    fxx: int,
    ocf_name: str,
    search_term: str,
) -> xr.DataArray | None:
    """
    Download a single variable for one forecast step using Herbie byte-range.

    Returns DataArray subset to India, or None on failure.
    """
    from herbie import Herbie

    try:
        H = Herbie(
            date_str,
            model="gfs",
            fxx=fxx,
            product="pgrb2.0p25",
            verbose=False,
        )

        # Download only this variable via byte-range
        ds = H.xarray(search_term, verbose=False)

        if ds is None or len(ds.data_vars) == 0:
            return None

        # Get the data variable (first one)
        var_name = list(ds.data_vars)[0]
        da = ds[var_name].load()

        # Handle GFS longitude convention (0-360 → subset India)
        if float(da.longitude.max()) > 180:
            da_india = da.sel(
                latitude=slice(INDIA_LAT_MAX, INDIA_LAT_MIN),
                longitude=slice(INDIA_LON_MIN, INDIA_LON_MAX),
            )
        else:
            da_india = da.sel(
                latitude=slice(INDIA_LAT_MAX, INDIA_LAT_MIN),
                longitude=slice(INDIA_LON_MIN, INDIA_LON_MAX),
            )

        # Drop extra coords from cfgrib (time, step, etc.)
        keep_dims = {"latitude", "longitude"}
        drop_coords = [c for c in da_india.coords if c not in keep_dims]
        da_india = da_india.drop_vars(drop_coords, errors="ignore")

        # Rename to OCF channel name
        da_india.name = ocf_name

        return da_india.astype(np.float32)

    except Exception as e:
        logger.debug(f"    {ocf_name} f{fxx:03d}: {e}")
        return None


def process_single_init_time(
    date: datetime,
    init_hour: int,
    channels: list[str] | None = None,
) -> xr.Dataset | None:
    """
    Process all forecast steps for a single GFS init time.

    Downloads all 14 OCF channels for each forecast step,
    combines into a Dataset with dims (step, latitude, longitude).

    Args:
        date: Date to process
        init_hour: GFS initialization hour (0, 6, 12, 18)
        channels: Optional subset of channels to download

    Returns:
        xr.Dataset with dims (init_time_utc, step, latitude, longitude)
        and 14 data variables, or None if no data.
    """
    date_str = date.strftime("%Y-%m-%d")
    init_time = pd.Timestamp(date_str) + pd.Timedelta(hours=init_hour)

    if channels is None:
        channels = list(OCF_CHANNELS.keys())

    logger.info(f"Processing {init_time} ({len(channels)} channels × "
                f"{len(FORECAST_HOURS)} steps)")

    step_datasets = []

    for fxx in FORECAST_HOURS:
        step_vars = {}

        for ch_name in channels:
            spec = OCF_CHANNELS[ch_name]
            da = download_single_variable(
                date_str, init_hour, fxx, ch_name, spec["search"]
            )
            if da is not None:
                step_vars[ch_name] = da

        if not step_vars:
            logger.warning(f"  Step f{fxx:03d}: no variables extracted")
            continue

        step_ds = xr.Dataset(step_vars)
        step_td = np.timedelta64(fxx, "h")
        step_ds = step_ds.expand_dims({"step": [step_td]})
        step_datasets.append(step_ds)

        n_ok = len(step_vars)
        n_total = len(channels)
        logger.info(f"  f{fxx:03d}: {n_ok}/{n_total} channels OK")

    if not step_datasets:
        logger.warning(f"  No valid steps for {init_time}")
        return None

    combined = xr.concat(step_datasets, dim="step")
    combined = combined.expand_dims({"init_time_utc": [init_time]})

    logger.info(f"  ✓ {init_time}: {len(step_datasets)} steps, "
                f"{len(combined.data_vars)} channels")
    return combined


def process_month(
    year: int,
    month: int,
    output_dir: str,
    max_days: int | None = None,
    channels: list[str] | None = None,
    dry_run: bool = False,
) -> str | None:
    """
    Process one month of GFS data for India and save as Zarr.

    Args:
        year: Year to process
        month: Month to process (1-12)
        output_dir: Directory for output Zarr files
        max_days: Limit number of days (for testing)
        channels: Optional channel subset
        dry_run: If True, only verify data availability

    Returns:
        Path to output Zarr file, or None.
    """
    from herbie import Herbie

    # Date range
    start = datetime(year, month, 1)
    end = datetime(year + (month // 12), (month % 12) + 1, 1)
    dates = []
    current = start
    while current < end:
        dates.append(current)
        current += timedelta(days=1)

    if max_days:
        dates = dates[:max_days]

    n_init = len(dates) * len(INIT_HOURS)
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}"
                f"Processing {year}-{month:02d}: "
                f"{len(dates)} days, {n_init} init times")

    if dry_run:
        # Just check availability for first day
        for init_hour in INIT_HOURS:
            try:
                H = Herbie(
                    dates[0].strftime("%Y-%m-%d"),
                    model="gfs", fxx=0, product="pgrb2.0p25", verbose=False,
                )
                inv = H.inventory()
                logger.info(f"  {dates[0].strftime('%Y-%m-%d')} {init_hour:02d}Z: "
                            f"{len(inv)} GRIB messages available")
            except Exception as e:
                logger.warning(f"  {dates[0].strftime('%Y-%m-%d')} {init_hour:02d}Z: "
                               f"unavailable ({e})")
        return None

    all_datasets = []

    for date in dates:
        for init_hour in INIT_HOURS:
            try:
                ds = process_single_init_time(date, init_hour, channels)
                if ds is not None:
                    all_datasets.append(ds)
            except Exception as e:
                logger.error(f"Failed {date.strftime('%Y-%m-%d')} "
                             f"{init_hour:02d}Z: {e}")

    if not all_datasets:
        logger.warning(f"No data processed for {year}-{month:02d}")
        return None

    # Combine all init times
    logger.info(f"Combining {len(all_datasets)} init times...")
    combined = xr.concat(all_datasets, dim="init_time_utc")
    combined = combined.sortby("init_time_utc")

    # Ensure latitude is descending (N→S, matching OCF convention)
    if combined.latitude[0] < combined.latitude[-1]:
        combined = combined.isel(latitude=slice(None, None, -1))

    # Save as Zarr
    output_path = os.path.join(output_dir, f"india_gfs_{year}_{month:02d}.zarr")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving {output_path}...")
    logger.info(f"  Dims: {dict(combined.dims)}")
    logger.info(f"  Channels: {list(combined.data_vars)}")
    lat_min, lat_max = float(combined.latitude.min()), float(combined.latitude.max())
    lon_min, lon_max = float(combined.longitude.min()), float(combined.longitude.max())
    logger.info(f"  Lat: {lat_min:.1f} to {lat_max:.1f}")
    logger.info(f"  Lon: {lon_min:.1f} to {lon_max:.1f}")

    combined.to_zarr(output_path, mode="w", consolidated=True)
    logger.info(f"✓ Saved: {output_path}")

    return output_path


def merge_monthly_zarrs(zarr_paths: list[str], output_path: str) -> str:
    """Merge monthly Zarr files into a single yearly Zarr."""
    logger.info(f"Merging {len(zarr_paths)} monthly files → {output_path}")

    datasets = [xr.open_zarr(p) for p in zarr_paths]
    combined = xr.concat(datasets, dim="init_time_utc")
    combined = combined.sortby("init_time_utc")
    combined.to_zarr(output_path, mode="w", consolidated=True)

    logger.info(f"✓ Merged: {combined.dims['init_time_utc']} init times")
    return output_path


def validate_zarr(zarr_path: str) -> bool:
    """Validate output Zarr matches OCF GFS schema."""
    logger.info(f"Validating {zarr_path}...")
    ds = xr.open_zarr(zarr_path)

    # Check dims
    required_dims = {"init_time_utc", "step", "latitude", "longitude"}
    actual_dims = set(ds.dims)
    assert required_dims.issubset(actual_dims), \
        f"Missing dims: {required_dims - actual_dims}"

    # Check channels
    expected_channels = set(OCF_CHANNELS.keys())
    actual_channels = set(ds.data_vars)
    missing = expected_channels - actual_channels
    if missing:
        logger.warning(f"  Missing channels: {missing}")
    else:
        logger.info(f"  ✓ All 14 channels present")

    # Check lat/lon bounds cover India
    assert float(ds.latitude.min()) <= 8.0, "Latitude min should cover southern India"
    assert float(ds.latitude.max()) >= 36.0, "Latitude max should cover northern India"
    assert float(ds.longitude.min()) <= 70.0, "Longitude min should cover western India"
    assert float(ds.longitude.max()) >= 96.0, "Longitude max should cover eastern India"
    logger.info(f"  ✓ Spatial coverage OK")

    # Check data types
    for var in ds.data_vars:
        assert ds[var].dtype == np.float32, \
            f"{var} dtype should be float32, got {ds[var].dtype}"
    logger.info(f"  ✓ Data types OK (float32)")

    logger.info(f"  ✓ Validation passed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download NOAA GFS data for India → OCF-compatible Zarr"
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--months", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="data/gfs_india")
    parser.add_argument("--max-days", type=int, default=None,
                        help="Limit days per month (testing)")
    parser.add_argument("--channels", type=str, nargs="+", default=None,
                        help="Subset of channels to download")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge", action="store_true",
                        help="Merge monthly Zarrs into yearly file")
    parser.add_argument("--validate", type=str, default=None,
                        help="Validate an existing Zarr file")

    args = parser.parse_args()

    if args.validate:
        validate_zarr(args.validate)
        return

    monthly_paths = []
    for month in args.months:
        path = process_month(
            year=args.year,
            month=month,
            output_dir=args.output_dir,
            max_days=args.max_days,
            channels=args.channels,
            dry_run=args.dry_run,
        )
        if path:
            monthly_paths.append(path)

    if args.merge and len(monthly_paths) > 1:
        yearly = os.path.join(args.output_dir, f"india_gfs_{args.year}.zarr")
        merge_monthly_zarrs(monthly_paths, yearly)
        validate_zarr(yearly)
    elif monthly_paths:
        validate_zarr(monthly_paths[-1])


if __name__ == "__main__":
    main()
