"""
Download and process NOAA GFS data for India region.

Two download modes:
  1. NOMADS GRIB filter (fast) — Selects specific variables + India subregion
     in a single HTTP request. Returns ~100-200KB per file vs 300MB full GRIB.
     Only available for last ~10 days of data.
  2. Herbie byte-range (fallback) — For historical data from S3.
     Uses .idx index files to download specific variables.

Output: OCF-compatible Zarr with dims (init_time_utc, step, latitude, longitude)
and 14 data variables matching existing GFS schema.

Usage:
    # Fast mode — recent data via NOMADS filter (recommended for testing)
    python download_gfs_india.py --year 2026 --months 2 --max-days 1

    # Historical data via Herbie S3 byte-range
    python download_gfs_india.py --year 2024 --months 1 --max-days 1 --source herbie

    # Parallel downloads (10 workers)
    python download_gfs_india.py --year 2024 --months 1 --max-days 3 --workers 10

Requirements:
    pip install xarray cfgrib eccodes numpy pandas zarr requests
    pip install herbie-data  # only needed for --source herbie
"""

import argparse
import logging
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# OCF channel mapping
# --------------------------------------------------------------------------- #

# NOMADS uses different parameter names than GRIB shortnames
# Format: ocf_name -> (nomads_var_param, herbie_search_regex, description)
OCF_CHANNELS = {
    "dlwrf": {
        "nomads": "DLWRF",
        "search": ":DLWRF:surface",
        "level": "surface",
    },
    "dswrf": {
        "nomads": "DSWRF",
        "search": ":DSWRF:surface",
        "level": "surface",
    },
    "hcc": {
        "nomads": "HCDC",
        "search": ":HCDC:high cloud layer:(?!.*ave)",
        "level": "high_cloud_layer",
    },
    "lcc": {
        "nomads": "LCDC",
        "search": ":LCDC:low cloud layer:(?!.*ave)",
        "level": "low_cloud_layer",
    },
    "mcc": {
        "nomads": "MCDC",
        "search": ":MCDC:middle cloud layer:(?!.*ave)",
        "level": "middle_cloud_layer",
    },
    "prate": {
        "nomads": "PRATE",
        "search": ":PRATE:surface:(?!.*ave)",
        "level": "surface",
    },
    "r": {
        "nomads": "RH",
        "search": ":RH:850 mb",
        "level": "850_mb",
    },
    "t": {
        "nomads": "TMP",
        "search": ":TMP:2 m above ground",
        "level": "2_m_above_ground",
    },
    "tcc": {
        "nomads": "TCDC",
        "search": ":TCDC:entire atmosphere:(?!.*ave)",
        "level": "entire_atmosphere_(considered_as_a_single_layer)",
    },
    "u10": {
        "nomads": "UGRD",
        "search": ":UGRD:10 m above ground",
        "level": "10_m_above_ground",
    },
    "u100": {
        "nomads": "UGRD",
        "search": ":UGRD:100 m above ground",
        "level": "100_m_above_ground",
    },
    "v10": {
        "nomads": "VGRD",
        "search": ":VGRD:10 m above ground",
        "level": "10_m_above_ground",
    },
    "v100": {
        "nomads": "VGRD",
        "search": ":VGRD:100 m above ground",
        "level": "100_m_above_ground",
    },
    "vis": {
        "nomads": "VIS",
        "search": ":VIS:surface",
        "level": "surface",
    },
}

# India bounding box (with 1° buffer)
INDIA_LAT_MIN = 5.0
INDIA_LAT_MAX = 39.0
INDIA_LON_MIN = 67.0
INDIA_LON_MAX = 99.0

# GFS forecast hours (17 steps: 0-48h at 3h intervals)
FORECAST_HOURS = list(range(0, 49, 3))

# GFS initialization hours (4x daily)
INIT_HOURS = [0, 6, 12, 18]

# NOMADS base URL
NOMADS_BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"


# --------------------------------------------------------------------------- #
# NOMADS GRIB Filter — fast, subregion-aware downloads
# --------------------------------------------------------------------------- #

def build_nomads_url(date: datetime, init_hour: int, fxx: int) -> str:
    """
    Build NOMADS GRIB filter URL for India-subset GFS download.

    Downloads ALL 14 OCF variables + India subregion in a single request.
    Returns ~100-200KB GRIB file instead of 300MB full global file.
    """
    date_str = date.strftime("%Y%m%d")

    params = {
        "dir": f"/gfs.{date_str}/{init_hour:02d}/atmos",
        "file": f"gfs.t{init_hour:02d}z.pgrb2.0p25.f{fxx:03d}",
        # Subregion — India with buffer
        "subregion": "",
        "toplat": str(INDIA_LAT_MAX),
        "bottomlat": str(INDIA_LAT_MIN),
        "leftlon": str(INDIA_LON_MIN),
        "rightlon": str(INDIA_LON_MAX),
    }

    # Add all variable selections
    nomads_vars = set()
    for spec in OCF_CHANNELS.values():
        nomads_vars.add(spec["nomads"])
    for var in sorted(nomads_vars):
        params[f"var_{var}"] = "on"

    # Add level selections
    nomads_levels = set()
    for spec in OCF_CHANNELS.values():
        nomads_levels.add(spec["level"])
    for level in sorted(nomads_levels):
        params[f"lev_{level}"] = "on"

    # Build URL manually (NOMADS is finicky about param order)
    param_str = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{NOMADS_BASE}?{param_str}"


def download_nomads_step(
    date: datetime,
    init_hour: int,
    fxx: int,
    tmp_dir: str,
    timeout: int = 60,
) -> str | None:
    """Download a single forecast step via NOMADS grib filter."""
    url = build_nomads_url(date, init_hour, fxx)
    fname = f"gfs_{date.strftime('%Y%m%d')}_{init_hour:02d}z_f{fxx:03d}.grib2"
    local_path = os.path.join(tmp_dir, fname)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 1000:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            size_kb = len(resp.content) / 1024
            logger.debug(f"  Downloaded f{fxx:03d}: {size_kb:.0f} KB")
            return local_path
        else:
            logger.debug(f"  f{fxx:03d}: HTTP {resp.status_code} or empty")
            return None
    except Exception as e:
        logger.debug(f"  f{fxx:03d}: download failed ({e})")
        return None


def extract_variables_from_grib(grib_path: str) -> dict[str, xr.DataArray]:
    """Extract OCF variables from a subsetted GRIB file."""
    variables = {}

    for ocf_name, spec in OCF_CHANNELS.items():
        try:
            ds = xr.open_dataset(
                grib_path,
                engine="cfgrib",
                backend_kwargs={
                    "filter_by_keys": {
                        "shortName": spec["nomads"].lower()
                        if spec["nomads"] not in ("HCDC", "LCDC", "MCDC")
                        else spec["nomads"].lower(),
                    },
                    "errors": "ignore",
                },
            )

            if len(ds.data_vars) == 0:
                continue

            var_name = list(ds.data_vars)[0]
            da = ds[var_name].load()

            # Drop extra coords
            keep = {"latitude", "longitude"}
            drop = [c for c in da.coords if c not in keep]
            da = da.drop_vars(drop, errors="ignore")
            da.name = ocf_name

            variables[ocf_name] = da.astype(np.float32)
            ds.close()

        except Exception:
            pass

    return variables


# --------------------------------------------------------------------------- #
# Herbie byte-range downloads — for historical S3 data
# --------------------------------------------------------------------------- #

def download_herbie_step(
    date_str: str,
    init_hour: int,
    fxx: int,
    channels: list[str],
) -> dict[str, xr.DataArray]:
    """Download variables via Herbie byte-range from S3."""
    from herbie import Herbie

    variables = {}
    try:
        H = Herbie(
            date_str,
            model="gfs",
            fxx=fxx,
            product="pgrb2.0p25",
            verbose=False,
        )

        for ch_name in channels:
            spec = OCF_CHANNELS[ch_name]
            try:
                ds = H.xarray(spec["search"], verbose=False)
                if ds is None or len(ds.data_vars) == 0:
                    continue

                var_name = list(ds.data_vars)[0]
                da = ds[var_name].load()

                # Subset to India
                if float(da.longitude.max()) > 180:
                    da = da.sel(
                        latitude=slice(INDIA_LAT_MAX, INDIA_LAT_MIN),
                        longitude=slice(INDIA_LON_MIN, INDIA_LON_MAX),
                    )
                else:
                    da = da.sel(
                        latitude=slice(INDIA_LAT_MAX, INDIA_LAT_MIN),
                        longitude=slice(INDIA_LON_MIN, INDIA_LON_MAX),
                    )

                keep = {"latitude", "longitude"}
                drop = [c for c in da.coords if c not in keep]
                da = da.drop_vars(drop, errors="ignore")
                da.name = ch_name
                variables[ch_name] = da.astype(np.float32)

            except Exception:
                pass

    except Exception as e:
        logger.warning(f"  Herbie init failed for f{fxx:03d}: {e}")

    return variables


# --------------------------------------------------------------------------- #
# Core processing pipeline
# --------------------------------------------------------------------------- #

def process_single_init_time(
    date: datetime,
    init_hour: int,
    source: str = "nomads",
    workers: int = 6,
    channels: list[str] | None = None,
) -> xr.Dataset | None:
    """
    Process all forecast steps for a single GFS init time.

    Args:
        date: Date to process
        init_hour: Init hour (0, 6, 12, 18)
        source: "nomads" (fast, recent data) or "herbie" (historical S3)
        workers: Number of parallel download workers
        channels: Channel subset (default: all 14)

    Returns:
        xr.Dataset with dims (init_time_utc, step, latitude, longitude)
    """
    if channels is None:
        channels = list(OCF_CHANNELS.keys())

    init_time = pd.Timestamp(date.strftime("%Y-%m-%d")) + pd.Timedelta(
        hours=init_hour
    )
    logger.info(f"Processing {init_time} [{source}] "
                f"({len(channels)}ch × {len(FORECAST_HOURS)}steps, "
                f"{workers} workers)")

    step_datasets = []

    if source == "nomads":
        # NOMADS: download subsetted GRIB files in parallel
        with tempfile.TemporaryDirectory(prefix="gfs_india_") as tmp_dir:
            # Parallel download
            grib_paths = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        download_nomads_step, date, init_hour, fxx, tmp_dir
                    ): fxx
                    for fxx in FORECAST_HOURS
                }
                for future in as_completed(futures):
                    fxx = futures[future]
                    try:
                        path = future.result()
                        if path:
                            grib_paths[fxx] = path
                    except Exception as e:
                        logger.debug(f"  f{fxx:03d}: {e}")

            logger.info(f"  Downloaded {len(grib_paths)}/{len(FORECAST_HOURS)} steps")

            # Extract variables from each GRIB (parallel)
            for fxx in FORECAST_HOURS:
                if fxx not in grib_paths:
                    continue
                variables = extract_variables_from_grib(grib_paths[fxx])
                if not variables:
                    continue

                step_ds = xr.Dataset(variables)
                step_td = np.timedelta64(fxx, "h")
                step_ds = step_ds.expand_dims({"step": [step_td]})
                step_datasets.append(step_ds)

    else:
        # Herbie: byte-range downloads from S3
        date_str = date.strftime("%Y-%m-%d")
        for fxx in FORECAST_HOURS:
            variables = download_herbie_step(date_str, init_hour, fxx, channels)
            if not variables:
                logger.debug(f"  f{fxx:03d}: no variables")
                continue

            step_ds = xr.Dataset(variables)
            step_td = np.timedelta64(fxx, "h")
            step_ds = step_ds.expand_dims({"step": [step_td]})
            step_datasets.append(step_ds)

            n_ok = len(variables)
            logger.info(f"  f{fxx:03d}: {n_ok}/{len(channels)} channels OK")

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
    source: str = "nomads",
    workers: int = 6,
    channels: list[str] | None = None,
    dry_run: bool = False,
) -> str | None:
    """
    Process one month of GFS data for India and save as Zarr.

    Args:
        year: Year to process
        month: Month to process (1-12)
        output_dir: Directory for output Zarr files
        max_days: Limit days per month (testing)
        source: "nomads" or "herbie"
        workers: Parallel download workers
        channels: Channel subset
        dry_run: Verify availability without downloading

    Returns:
        Path to output Zarr, or None.
    """
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
                f"{year}-{month:02d}: {len(dates)} days, {n_init} init times "
                f"[{source}, {workers} workers]")

    if dry_run:
        if source == "nomads":
            url = build_nomads_url(dates[0], 0, 3)
            try:
                resp = requests.head(url, timeout=10)
                logger.info(f"  NOMADS: HTTP {resp.status_code}")
            except Exception as e:
                logger.warning(f"  NOMADS: {e}")
        return None

    all_datasets = []

    for date in dates:
        for init_hour in INIT_HOURS:
            try:
                ds = process_single_init_time(
                    date, init_hour, source, workers, channels
                )
                if ds is not None:
                    all_datasets.append(ds)
            except Exception as e:
                logger.error(f"Failed {date.strftime('%Y-%m-%d')} "
                             f"{init_hour:02d}Z: {e}")

    if not all_datasets:
        logger.warning(f"No data processed for {year}-{month:02d}")
        return None

    logger.info(f"Combining {len(all_datasets)} init times...")
    combined = xr.concat(all_datasets, dim="init_time_utc")
    combined = combined.sortby("init_time_utc")

    # Ensure latitude descending (N→S, matching OCF convention)
    if combined.latitude[0] < combined.latitude[-1]:
        combined = combined.isel(latitude=slice(None, None, -1))

    # Save as Zarr
    output_path = os.path.join(output_dir, f"india_gfs_{year}_{month:02d}.zarr")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving {output_path}...")
    logger.info(f"  Dims: {dict(combined.dims)}")
    logger.info(f"  Channels: {list(combined.data_vars)}")
    lat_min = float(combined.latitude.min())
    lat_max = float(combined.latitude.max())
    lon_min = float(combined.longitude.min())
    lon_max = float(combined.longitude.max())
    logger.info(f"  Lat: {lat_min:.1f} to {lat_max:.1f}")
    logger.info(f"  Lon: {lon_min:.1f} to {lon_max:.1f}")

    combined.to_zarr(output_path, mode="w", consolidated=True)
    logger.info(f"✓ Saved: {output_path}")

    return output_path


def merge_monthly_zarrs(zarr_paths: list[str], output_path: str) -> str:
    """Merge monthly Zarr files into a single yearly Zarr."""
    logger.info(f"Merging {len(zarr_paths)} files → {output_path}")
    datasets = [xr.open_zarr(p) for p in zarr_paths]
    combined = xr.concat(datasets, dim="init_time_utc")
    combined = combined.sortby("init_time_utc")
    combined.to_zarr(output_path, mode="w", consolidated=True)
    logger.info(f"✓ Merged: {combined.dims['init_time_utc']} init times")
    return output_path


def validate_zarr(zarr_path: str) -> bool:
    """Validate Zarr matches OCF GFS schema."""
    logger.info(f"Validating {zarr_path}...")
    ds = xr.open_zarr(zarr_path)

    required_dims = {"init_time_utc", "step", "latitude", "longitude"}
    actual_dims = set(ds.dims)
    assert required_dims.issubset(actual_dims), \
        f"Missing dims: {required_dims - actual_dims}"

    expected = set(OCF_CHANNELS.keys())
    actual = set(ds.data_vars)
    missing = expected - actual
    if missing:
        logger.warning(f"  Missing channels: {missing}")
    else:
        logger.info("  ✓ All 14 channels present")

    assert float(ds.latitude.min()) <= 8.0
    assert float(ds.latitude.max()) >= 36.0
    assert float(ds.longitude.min()) <= 70.0
    assert float(ds.longitude.max()) >= 96.0
    logger.info("  ✓ Spatial coverage OK (India)")

    for var in ds.data_vars:
        assert ds[var].dtype == np.float32
    logger.info("  ✓ float32 types OK")

    logger.info("  ✓ Validation passed")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download NOAA GFS data for India → OCF Zarr"
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--months", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="data/gfs_india")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--source", choices=["nomads", "herbie"], default="nomads",
                        help="nomads=GRIB filter (fast, recent), "
                             "herbie=S3 byte-range (historical)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel download workers")
    parser.add_argument("--channels", type=str, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--validate", type=str, default=None)

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
            source=args.source,
            workers=args.workers,
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
