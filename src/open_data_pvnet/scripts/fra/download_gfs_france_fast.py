"""
Download and process NOAA GFS data for France region.
Adopted from Raakshass:feature/india-solar-pipeline & Sharkyii:germany in open-data-pvnet repo.

Two download modes:
  1. NOMADS GRIB filter (fast) — Selects specific variables + France subregion
     in a single HTTP request. Returns ~100-200KB per file vs 300MB full GRIB.
     Only available for last ~10 days of data.
  2. Herbie byte-range (fallback) — For historical data from S3.
     Uses .idx index files to download specific variables.

Output: OCF-compatible Zarr with dims (init_time_utc, step, latitude, longitude)
and 14 data variables matching existing GFS schema.

Usage:
    # Fast mode — recent data via NOMADS filter (recommended for testing)
    python src/open_data_pvnet/scripts/fra/download_gfs_france_fast.py --year 2026 --months 2 --max-days 1

    # Historical data via Herbie S3 byte-range
    python src/open_data_pvnet/scripts/fra/download_gfs_france_fast.py --year 2024 --months 1 --max-days 1 --source herbie

    # Parallel downloads (10 workers)
    python src/open_data_pvnet/scripts/fra/download_gfs_france_fast.py --year 2024 --months 1 --max-days 3 --workers 10

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

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Define France latitude and longitude bounds
FRANCE_MIN_LAT, FRANCE_MAX_LAT = 42.0, 51.5
FRANCE_MIN_LON, FRANCE_MAX_LON = -5.5, 9.0

# --------------------------------------------------------------------------- #
# OCF channel mapping
# --------------------------------------------------------------------------- #

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
        "search": ":[HT]CDC:high cloud layer:",
        "level": "high_cloud_layer",
    },
    "lcc": {
        "nomads": "LCDC",
        "search": ":[LT]CDC:low cloud layer:",
        "level": "low_cloud_layer",
    },
    "mcc": {
        "nomads": "MCDC",
        "search": ":[MT]CDC:middle cloud layer:",
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
        "search": ":TCDC:entire atmosphere:",
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
    """Build NOMADS GRIB filter URL for France-subset GFS download."""
    date_str = date.strftime("%Y%m%d")

    params = {
        "dir": f"/gfs.{date_str}/{init_hour:02d}/atmos",
        "file": f"gfs.t{init_hour:02d}z.pgrb2.0p25.f{fxx:03d}",
        "subregion": "",
        "toplat": str(FRANCE_MAX_LAT),
        "bottomlat": str(FRANCE_MIN_LAT),
        "leftlon": str(FRANCE_MIN_LON),
        "rightlon": str(FRANCE_MAX_LON),
    }

    nomads_vars = set()
    for spec in OCF_CHANNELS.values():
        nomads_vars.add(spec["nomads"])
    for var in sorted(nomads_vars):
        params[f"var_{var}"] = "on"

    nomads_levels = set()
    for spec in OCF_CHANNELS.values():
        nomads_levels.add(spec["level"])
    for level in sorted(nomads_levels):
        params[f"lev_{level}"] = "on"

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
            return local_path
        return None
    except Exception:
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
                        "shortName": spec["nomads"].lower(),
                    },
                    "errors": "ignore",
                },
            )
            if len(ds.data_vars) == 0:
                continue
            var_name = list(ds.data_vars)[0]
            da = ds[var_name].load()
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

def download_herbie_variable(herbie_obj, ch_name: str) -> tuple[str, xr.DataArray] | None:
    """Download a single variable using an initialized Herbie object."""
    spec = OCF_CHANNELS[ch_name]
    try:
        ds = herbie_obj.xarray(spec["search"], verbose=False)
        if ds is None or len(ds.data_vars) == 0:
            return None
        var_name = list(ds.data_vars)[0]
        da = ds[var_name].load()
        # Subset to France
        da = da.sel(
            latitude=slice(FRANCE_MAX_LAT, FRANCE_MIN_LAT),
            longitude=slice(FRANCE_MIN_LON, FRANCE_MAX_LON),
        )
        keep = {"latitude", "longitude"}
        drop = [c for c in da.coords if c not in keep]
        da = da.drop_vars(drop, errors="ignore")
        da.name = ch_name
        return ch_name, da.astype(np.float32)
    except Exception:
        return None

def download_herbie_step(
    date_str: str,
    init_hour: int,
    fxx: int,
    channels: list[str],
    workers: int = 4,
) -> dict[str, xr.DataArray]:
    """Download variables via Herbie byte-range from S3 in parallel."""
    from herbie import Herbie
    variables = {}
    try:
        H = Herbie(
            date_str,
            model="gfs",
            fxx=fxx,
            product="pgrb2.0p25",
            verbose=False,
            priority=['aws', 'nomads', 'google', 'azure']
        )
        # Parallelize variable downloads within the forecast step
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(download_herbie_variable, H, ch): ch for ch in channels}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    ch_name, da = result
                    variables[ch_name] = da
    except Exception as e:
        logger.warning(f"  Herbie failed for f{fxx:03d}: {e}")
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
    """Process all forecast steps for a single GFS init time."""
    if channels is None:
        channels = list(OCF_CHANNELS.keys())
    init_time = pd.Timestamp(date.strftime("%Y-%m-%d")) + pd.Timedelta(hours=init_hour)
    logger.info(f"Processing {init_time} [{source}]")
    
    step_datasets = {}
    
    if source == "nomads":
        with tempfile.TemporaryDirectory(prefix="gfs_germany_") as tmp_dir:
            grib_paths = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(download_nomads_step, date, init_hour, fxx, tmp_dir): fxx
                    for fxx in FORECAST_HOURS
                }
                for future in as_completed(futures):
                    fxx = futures[future]
                    try:
                        path = future.result()
                        if path: grib_paths[fxx] = path
                    except Exception: pass
            for fxx in sorted(grib_paths.keys()):
                variables = extract_variables_from_grib(grib_paths[fxx])
                if not variables: continue
                ds = xr.Dataset(variables).expand_dims({"step": [np.timedelta64(fxx, "h")]})
                step_datasets[fxx] = ds
    else:
        date_str = date.strftime("%Y-%m-%d")
        # Parallelize forecast steps for Herbie too
        with ThreadPoolExecutor(max_workers=max(1, workers // 2)) as executor:
            futures = {
                executor.submit(download_herbie_step, date_str, init_hour, fxx, channels): fxx
                for fxx in FORECAST_HOURS
            }
            for future in as_completed(futures):
                fxx = futures[future]
                variables = future.result()
                if variables:
                    ds = xr.Dataset(variables).expand_dims({"step": [np.timedelta64(fxx, "h")]})
                    step_datasets[fxx] = ds

    if not step_datasets: return None
    
    # Sort by step before concat
    sorted_steps = [step_datasets[fxx] for fxx in sorted(step_datasets.keys())]
    combined = xr.concat(sorted_steps, dim="step")
    combined = combined.expand_dims({"init_time_utc": [init_time]})
    return combined


def process_month(
    year: int, month: int, output_dir: str, 
    max_days: int = None, source: str = "nomads", 
    workers: int = 6, channels: list[str] = None
) -> str | None:
    """Process one month of GFS data for Germany and save as Zarr."""
    start = datetime(year, month, 1)
    if month == 12: end = datetime(year + 1, 1, 1)
    else: end = datetime(year, month + 1, 1)
    
    # Calculate total days in the month
    total_days = (end - start).days
    
    # Default max_days to total days in month if not specified
    if max_days is None:
        max_days = total_days
    
    dates = []
    current = start
    while current < end:
        dates.append(current)
        current += timedelta(days=1)
    dates = dates[:max_days]  # Always slice, using calculated max_days
    
    logger.info(f"Processing {year}-{month:02d} [{source}]")
    all_datasets = []
    
    # Generate all (date, init_hour) pairs
    init_pairs = []
    for date in dates:
        for init_hour in INIT_HOURS:
            init_pairs.append((date, init_hour))
            
    pbar = None
    if tqdm:
        pbar = tqdm(total=len(init_pairs), desc=f"Downloading {year}-{month:02d}")
        
    def _process_one(pair):
        d, h = pair
        try:
            return process_single_init_time(d, h, source, workers, channels)
        except Exception as e:
            logger.error(f"Failed {d.strftime('%Y-%m-%d')} {h:02d}Z: {e}")
            return None

    # Parallelize at the init_time level for historical data
    # We use fewer workers here if the inner loop is also parallelized
    top_workers = 1 if source == "nomads" else max(1, workers // 4)
    if source == "herbie":
        # For historical data, top-level parallelism is more efficient
        logger.info(f"Using {top_workers} concurrent initialization times for historical download")
        with ThreadPoolExecutor(max_workers=top_workers) as executor:
            futures = {executor.submit(_process_one, p): p for p in init_pairs}
            for future in as_completed(futures):
                ds = future.result()
                if ds is not None:
                    all_datasets.append(ds)
                if pbar: pbar.update(1)
    else:
        # NOMADS is usually fast enough and supports its own internal parallelism
        for pair in init_pairs:
            ds = _process_one(pair)
            if ds is not None:
                all_datasets.append(ds)
            if pbar: pbar.update(1)
                
    if pbar: pbar.close()
    
    if not all_datasets:
        logger.warning(f"No data captured for {year}-{month:02d}")
        return None
    
    combined = xr.concat(all_datasets, dim="init_time_utc").sortby("init_time_utc")
    if combined.latitude[0] < combined.latitude[-1]:
        combined = combined.isel(latitude=slice(None, None, -1))
        
    output_path = os.path.join(output_dir, f"france_gfs_{year}_{month:02d}.zarr")
    os.makedirs(output_dir, exist_ok=True)
    combined.to_zarr(output_path, mode="w", consolidated=True, zarr_version=2)
    logger.info(f"✓ Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download NOAA GFS data for France → OCF Zarr")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--months", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, default="data/fra/gfs")
    parser.add_argument("--max-days", type=int, default=None, 
                        help="Maximum number of days to process from start of month (default: all days in the month)")
    parser.add_argument("--source", choices=["nomads", "herbie"], default="nomads")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for steps/variables")
    parser.add_argument("--channels", type=str, nargs="+", default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    
    monthly_paths = []
    for month in args.months:
        path = process_month(args.year, month, args.output_dir, args.max_days, args.source, args.workers, args.channels)
        if path: monthly_paths.append(path)
    
    if args.merge and len(monthly_paths) > 1:
        yearly = os.path.join(args.output_dir, f"france_gfs_{args.year}.zarr")
        datasets = [xr.open_zarr(p) for p in monthly_paths]
        xr.concat(datasets, dim="init_time_utc").sortby("init_time_utc").to_zarr(yearly, mode="w", consolidated=True, zarr_version=2)
        logger.info(f"✓ Merged: {yearly}")

if __name__ == "__main__":
    main()