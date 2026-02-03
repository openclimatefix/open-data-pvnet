import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr

# ---------------- CONFIGURATION ----------------
BASE_URL = "https://www.smard.de/app/chart_data"
FILTER_ID = 4068
REGION = "DE"
OUTPUT_DIR = Path("./germany_pv_data")
OUTPUT_ZARR = Path("./gsp_2023.zarr")

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

GSP_ID = "germany_total"
CAPACITY_MWP = np.nan
INSTALLED_CAPACITY_MWP = np.nan
RESOLUTION = "quarterhour"
REQUEST_SLEEP = 0.25

OUTPUT_DIR.mkdir(exist_ok=True)


def get_timestamps(filter_id: int, region: str, resolution: str) -> List[int]:
    url = f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    if isinstance(data, dict) and "timestamps" in data:
        return data["timestamps"]
    return data


def get_chunk(filter_id: int, region: str, resolution: str, timestamp: int) -> Optional[List[Union[int, float]]]:
    url = f"{BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{timestamp}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict) and "series" in data:
        return data["series"]
    if isinstance(data, list):
        return data
    return None


def download_smard_data(start: str, end: str, resolution: str) -> pd.DataFrame:
    timestamps = get_timestamps(FILTER_ID, REGION, resolution)
    if not timestamps:
        raise RuntimeError("No timestamps index returned from SMARD.")

    start_ts = int(pd.to_datetime(start).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end).timestamp() * 1000)
    buckets = [ts for ts in timestamps if start_ts <= ts <= end_ts]

    print(f"Found {len(buckets)} buckets in date range.")

    rows = []
    for ts in buckets:
        try:
            chunk = get_chunk(FILTER_ID, REGION, resolution, ts)
            if not chunk:
                continue

            for entry in chunk:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    ts_ms = int(entry[0])
                    val = entry[1]
                elif isinstance(entry, dict):
                    ts_ms = int(entry.get("timestamp_ms", entry.get("timestamp", 0)))
                    val = entry.get("value") or entry.get("generation_mw") or entry.get("generation")
                else:
                    continue

                try:
                    v = float(val) if val is not None else np.nan
                except (ValueError, TypeError):
                    v = np.nan
                
                rows.append((ts_ms, v))
                
            time.sleep(REQUEST_SLEEP)
            
        except requests.RequestException as e:
            print(f"Failed to fetch bucket {ts}: {e}")
            time.sleep(REQUEST_SLEEP)

    if not rows:
        raise RuntimeError("No data downloaded for the specified range.")

    df = pd.DataFrame(rows, columns=["timestamp_ms", "generation_mw"])
    df = df.drop_duplicates("timestamp_ms").sort_values("timestamp_ms").reset_index(drop=True)
    df["datetime_gmt"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df

def save_to_zarr(df: pd.DataFrame, output_path: Path) -> xr.Dataset:
    n_rows = len(df)
    print(f"Processing {n_rows} rows for Zarr output...")

    # Ensure sorted by time
    df = df.sort_values("datetime_gmt")
    
    # Prepare coordinates
    # UK data structure: coords=(gsp_id, datetime_gmt)
    times = df["datetime_gmt"].values # datetime64[ns]
    gsp_ids = np.array([GSP_ID], dtype="U20")
    
    n_times = len(times)
    n_gsp = len(gsp_ids)
    
    # Values reshape to (gsp_id, datetime_gmt) -> (1, ntimes)
    gen_values = df["generation_mw"].astype(np.float32).fillna(np.nan).values.reshape(n_gsp, n_times)
    
    cap_val = np.float32(CAPACITY_MWP) if not np.isnan(CAPACITY_MWP) else np.nan
    inst_val = np.float32(INSTALLED_CAPACITY_MWP) if not np.isnan(INSTALLED_CAPACITY_MWP) else np.nan
    
    cap_arr = np.full((n_gsp, n_times), cap_val, dtype=np.float32)
    inst_arr = np.full((n_gsp, n_times), inst_val, dtype=np.float32)
    
    ds = xr.Dataset(
        data_vars={
            "capacity_mwp": (["gsp_id", "datetime_gmt"], cap_arr),
            "generation_mw": (["gsp_id", "datetime_gmt"], gen_values),
            "installedcapacity_mwp": (["gsp_id", "datetime_gmt"], inst_arr),
        },
        coords={
            "gsp_id": gsp_ids,
            "datetime_gmt": times,
        },
        attrs={
            "description": "PV generation data from SMARD",
            "source": "Bundesnetzagentur SMARD",
            "created": datetime.now().isoformat() + "Z",
        },
    )

    if output_path.exists():
        shutil.rmtree(output_path)

    ds.to_zarr(str(output_path), mode="w", consolidated=True)
    print(f"Dataset saved to {output_path}")
    return ds



if __name__ == "__main__":
    try:
        data_df = download_smard_data(START_DATE, END_DATE, RESOLUTION)
        dataset = save_to_zarr(data_df, OUTPUT_ZARR)
        print(dataset)
    except Exception as e:
        print(f"Process failed: {e}")