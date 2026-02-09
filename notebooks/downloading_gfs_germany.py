import os
import requests
import xarray as xr
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path

LAT_MIN, LAT_MAX = 47, 55
LON_MIN, LON_MAX = 6, 15

VARIABLES = {
    "dswrf": "DSWRF:surface",
    "t": "TMP:2 m above ground",
    "r": "RH:2 m above ground",
    "tcc": "TCDC:entire atmosphere",
    "u10": "UGRD:10 m above ground",
    "v10": "VGRD:10 m above ground",
}

OUTPUT_DIR = Path("./germany_gfs_data")
OUTPUT_DIR.mkdir(exist_ok=True)

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 1, 1)
CYCLES = [0, 6, 12, 18]
FORECAST_HOURS = [0, 3, 6, 9, 12, 15, 18, 21, 24]


def get_byte_ranges(idx_url):
    r = requests.get(idx_url)
    if r.status_code != 200:
        return None
    
    lines = r.text.splitlines()
    records = []
    
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 5:
            continue
        
        offset = int(parts[1])
        var_lvl = f"{parts[3]}:{parts[4]}"
        next_offset = int(lines[i+1].split(":")[1]) if i+1 < len(lines) else ""
        
        records.append({
            "offset": offset,
            "var_lvl": var_lvl,
            "next": next_offset
        })
    
    return records


def download_grib(date, cycle, fhour):
    date_str = date.strftime("%Y%m%d")
    cycle_str = f"{cycle:02d}"
    fhour_str = f"{fhour:03d}"
    
    base_url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.f{fhour_str}"
    idx_url = base_url + ".idx"
    filename = OUTPUT_DIR / f"gfs_{date_str}_{cycle_str}z_f{fhour_str}.grib2"
    
    if filename.exists() and filename.stat().st_size > 1000:
        return filename
    
    records = get_byte_ranges(idx_url)
    if not records:
        return None
    
    print(f"Downloading {filename.name}...")
    
    with open(filename, "wb") as f:
        for var_name, var_pattern in VARIABLES.items():
            record = next((r for r in records if r["var_lvl"] == var_pattern), None)
            
            if record:
                range_header = f"bytes={record['offset']}-{record['next']-1 if record['next'] else ''}"
                r = requests.get(base_url, headers={"Range": range_header}, timeout=30)
                r.raise_for_status()
                
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
    
    return filename


def process_grib(grib_path):
    try:
        ds = xr.open_dataset(grib_path, engine="cfgrib")
        data = ds.sel(latitude=slice(LAT_MAX, LAT_MIN), longitude=slice(LON_MIN, LON_MAX))
        ds.close()
        return data
    except:
        return None


def main():
    print("=" * 50)
    print("GFS WEATHER DATA DOWNLOADER")
    print("=" * 50)
    
    all_data = {}
    init_times = []
    steps = []
    
    current = START_DATE
    while current <= END_DATE:
        for cycle in CYCLES:
            init_time = current.replace(hour=cycle)
            
            for fhour in FORECAST_HOURS:
                grib_path = download_grib(current, cycle, fhour)
                
                if grib_path:
                    data = process_grib(grib_path)
                    if data:
                        all_data[(init_time, fhour)] = data
                        if init_time not in init_times:
                            init_times.append(init_time)
                        if fhour not in steps:
                            steps.append(fhour)
        
        current += timedelta(days=1)
    
    if not all_data:
        print("No data downloaded")
        return
    
    init_times = sorted(init_times)
    steps = sorted(steps)
    
    # Assembly
    sample_key = list(all_data.keys())[0]
    lat_size = len(all_data[sample_key].latitude)
    lon_size = len(all_data[sample_key].longitude)
    channels = list(VARIABLES.keys())
    
    data_array = np.full((len(init_times), len(steps), len(channels), lat_size, lon_size), np.nan, dtype=np.float32)
    for (it, fh), data in all_data.items():
        it_idx = init_times.index(it)
        fh_idx = steps.index(fh)
        for i, ch in enumerate(channels):
            # Map the variable name from GFS to our channels
            grib_var = VARIABLES[ch].split(":")[0].lower()
            # Find the actual variable name in xarray (sometimes it's different)
            for var in data.data_vars:
                if var.lower() == grib_var:
                    data_array[it_idx, fh_idx, i] = data[var].values
                    break

    ds = xr.Dataset(
        {ch: (["init_time_utc", "step", "latitude", "longitude"], data_array[:, :, i]) for i, ch in enumerate(channels)},
        coords={
            "init_time_utc": init_times,
            "step": [np.timedelta64(h, "h") for h in steps],
            "latitude": np.linspace(LAT_MAX, LAT_MIN, lat_size),
            "longitude": np.linspace(LON_MIN, LON_MAX, lon_size),
        }
    )

    zarr_path = Path(r"c:\Users\SNEH\OneDrive\Desktop\GSOC\PR\Streamed\open-data-pvnet\notebooks\germany_gfs_2023.zarr")
    print(f"\nSaving to {zarr_path}")
    
    # Robust save for Windows
    if zarr_path.exists():
        import shutil
        shutil.rmtree(zarr_path, ignore_errors=True)
        
    ds.to_zarr(zarr_path, mode="w", consolidated=True)
    print("Done!")


if __name__ == "__main__":
    main()
