import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr

OUTPUT_DIR = Path("./gfs_data")
OUTPUT_DIR.mkdir(exist_ok=True)

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 3, 1)

LAT_MIN, LAT_MAX = 47, 55
LON_MIN, LON_MAX = 6, 15

CYCLES = [0, 6, 12, 18]
FORECAST_HOURS = list(range(0, 25, 3))

PV_ZARR_PATH = Path("./gsp_2023.zarr")

DESIRED_VARS = [
    "dlwrf", "dswrf", "hcc", "lcc", "mcc",
    "prate", "r", "sde", "t", "tcc",
    "u10", "u100", "v10", "v100", "vis"
]

TARGET_PATTERNS = {
    "dswrf": "DSWRF",
    "dlwrf": "DLWRF",
    "t": "TMP",
    "r": "RH",
    "tcc": "TCDC",
    "hcc": "HCDC",
    "mcc": "MCDC",
    "lcc": "LCDC",
    "prate": "PRATE",
    "u10": "UGRD:10",
    "v10": "VGRD:10",
    "u100": "UGRD:100",
    "v100": "VGRD:100",
    "vis": "VIS",
}


def get_byte_ranges(idx_url: str) -> Optional[List[Dict[str, Union[int, str]]]]:
    try:
        response = requests.get(idx_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None

    lines = response.text.splitlines()
    records = []
    
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 5:
            continue
            
        try:
            offset = int(parts[1])
        except ValueError:
            continue
            
        var_lvl = f"{parts[3]}:{parts[4]}"
        next_offset = None
        
        if i + 1 < len(lines):
            try:
                next_offset = int(lines[i + 1].split(":")[1])
            except (ValueError, IndexError):
                next_offset = None
                
        records.append({"offset": offset, "var_lvl": var_lvl, "next": next_offset})
        
    return records


def build_var_mapping(records: List[Dict], patterns: Dict[str, str]) -> Dict[str, str]:
    mapping = {}
    for key, pattern in patterns.items():
        found = None
        for record in records:
            if pattern.lower() in record["var_lvl"].lower():
                found = record["var_lvl"]
                break
        if found:
            mapping[key] = found
    return mapping


def download_smart(date: datetime, cycle: int, fhour: int, target_patterns: Dict[str, str]) -> Optional[Path]:
    date_str = date.strftime("%Y%m%d")
    cycle_str = f"{cycle:02d}"
    fhour_str = f"{fhour:03d}"

    base_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.f{fhour_str}"
    idx_url = f"{base_url}.idx"
    filename = OUTPUT_DIR / f"gfs_{date_str}_{cycle_str}z_f{fhour_str}.grib2"

    records = get_byte_ranges(idx_url)
    var_mapping = build_var_mapping(records, target_patterns)
    print(f"Downloading {filename.name}...")

    for attempt in range(3):
        try:
            with open(filename, "wb") as f:
                for var_lvl in var_mapping.values():
                    rec = next((r for r in records if r["var_lvl"] == var_lvl), None)
                    if not rec:
                        continue
                    
                    start = rec["offset"]
                    end = (rec["next"] - 1) if rec["next"] else ""
                    range_header = f"bytes={start}-{end}"
                    
                    with requests.get(base_url, headers={"Range": range_header}, stream=True, timeout=30) as r:
                        r.raise_for_status()
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
            return filename
        except Exception as e:
            if filename.exists():
                filename.unlink(missing_ok=True)
            time.sleep(2)
            
    return None


def load_pv_from_zarr(zarr_path: Path) -> Optional[pd.Series]:
    if not zarr_path.exists():
        print(f"PV Zarr not found at {zarr_path}, SDE channel will be NaN")
        return None

    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
        # Sum over gsp_id to get total generation for mapping, or take mean? 
        # Typically SDE maps to the generation profile. 
        # Since we have 1 GSP for Germany, sum or mean is fine.
        da = ds["generation_mw"].sum(dim="gsp_id")
        
        # Convert to series
        series = da.to_series()
        # Ensure UTC
        if series.index.tz is None:
             series.index = series.index.tz_localize("UTC")
        else:
             series.index = series.index.tz_convert("UTC")
             
        return series
    except Exception as e:
        print(f"Error loading PV Zarr: {e}")
        return None


def build_dataset() -> xr.Dataset:
    all_data = {}
    init_times = []
    steps = []
    
    current = START_DATE
    gfs_patterns = {k: v for k, v in TARGET_PATTERNS.items() if k != "sde"}

    while current <= END_DATE:
        for cycle in CYCLES:
            init_time = current.replace(hour=cycle, minute=0, second=0, microsecond=0)
            
            for fhour in FORECAST_HOURS:
                grib_path = download_smart(current, cycle, fhour, target_patterns=gfs_patterns)
                if not grib_path:
                    continue

                try:
                    filter_levels = [
                        {"typeOfLevel": "surface"},
                        {"typeOfLevel": "heightAboveGround", "level": 2},
                        {"typeOfLevel": "heightAboveGround", "level": 10},
                        {"typeOfLevel": "heightAboveGround", "level": 100},
                        {"typeOfLevel": "atmosphere"},
                        {"typeOfLevel": "entireAtmosphere"},
                        {"typeOfLevel": "lowCloudLayer"},
                        {"typeOfLevel": "middleCloudLayer"},
                        {"typeOfLevel": "highCloudLayer"},
                    ]

                    arrays = {}
                    for filters in filter_levels:
                        try:
                            ds_lvl = xr.open_dataset(
                                grib_path,
                                engine="cfgrib",
                                backend_kwargs={"indexpath": "", "filter_by_keys": filters}
                            )
                            
                            for var in ds_lvl.data_vars:
                                if "latitude" not in ds_lvl[var].coords or "longitude" not in ds_lvl[var].coords:
                                    continue

                                data = ds_lvl[var].sel(
                                    latitude=slice(LAT_MAX, LAT_MIN),
                                    longitude=slice(LON_MIN, LON_MAX)
                                )

                                grib_short = ds_lvl[var].attrs.get("GRIB_shortName", "").lower()
                                grib_std = ds_lvl[var].attrs.get("shortName", "").lower()
                                
                                matched = None
                                for key, patt in gfs_patterns.items():
                                    if patt.lower() in grib_short or patt.lower() in grib_std:
                                        matched = key
                                        break
                                
                                if matched:
                                    arrays[matched] = data.values
                            ds_lvl.close()
                        except Exception:
                            continue

                    if arrays:
                        key = (init_time, fhour)
                        all_data[key] = arrays
                        if init_time not in init_times:
                            init_times.append(init_time)
                        if fhour not in steps:
                            steps.append(fhour)

                except Exception as e:
                    print(f"Error processing {grib_path}: {e}")
                    
        current += timedelta(days=1)

    if not all_data:
        raise RuntimeError("No GFS data extracted.")

    init_times.sort()
    steps.sort()
    
    sample_key = next(iter(all_data.keys()))
    sample_arrays = list(all_data[sample_key].values())[0]
    lat_size, lon_size = sample_arrays.shape

    sde_matrix = None
    try:
        pv_series = load_pv_from_zarr(PV_ZARR_PATH)
        if pv_series is not None:
            n_init = len(init_times)
            n_steps = len(steps)
            sde_matrix = np.full((n_init, n_steps), np.nan, dtype=np.float32)

            for i, it in enumerate(init_times):
                for j, fh in enumerate(steps):
                    valid_time = (it + pd.Timedelta(hours=int(fh))).to_pydatetime()
                    valid_time = pd.Timestamp(valid_time).tz_localize("UTC") # Ensure UTC for comparison if needed

                    if valid_time in pv_series.index:
                        sde_val = pv_series.loc[valid_time]
                    else:
                        # Nearest lookup
                        try:
                            idx = pv_series.index.get_indexer([valid_time], method="nearest")[0]
                            sde_val = pv_series.iloc[idx]
                        except:
                            sde_val = np.nan
                    sde_matrix[i, j] = float(sde_val)
    except Exception as e:
        print(f"PV data integration skipped: {e}")

    n_init, n_steps = len(init_times), len(steps)
    data_buffer = np.full((n_init, n_steps, len(DESIRED_VARS), lat_size, lon_size), np.nan, dtype=np.float32)

    for (it, fh), arrays in all_data.items():
        i = init_times.index(it)
        j = steps.index(fh)
        
        for k, ch in enumerate(DESIRED_VARS):
            if ch == "sde":
                continue
            if ch in arrays:
                try:
                    data_buffer[i, j, k, :, :] = arrays[ch]
                except ValueError:
                    continue

    if "sde" in DESIRED_VARS and sde_matrix is not None:
        idx_sde = DESIRED_VARS.index("sde")
        for i in range(n_init):
            for j in range(n_steps):
                data_buffer[i, j, idx_sde, :, :] = sde_matrix[i, j]

    coords = {
        "init_time_utc": init_times,
        "step": [np.timedelta64(int(h), "h") for h in steps],
        "latitude": np.linspace(LAT_MAX, LAT_MIN, lat_size),
        "longitude": np.linspace(LON_MIN, LON_MAX, lon_size)
    }

    data_vars = {}
    for k, ch in enumerate(DESIRED_VARS):
        data_vars[ch] = (["init_time_utc", "step", "latitude", "longitude"], data_buffer[:, :, k, :, :])

    ds = xr.Dataset(data_vars, coords=coords)
    zarr_path = OUTPUT_DIR / "processed_dataset.zarr"
    ds.to_zarr(zarr_path, mode="w", consolidated=True)
    
    print(f"Dataset saved to {zarr_path}")
    return ds


if __name__ == "__main__":
    build_dataset()