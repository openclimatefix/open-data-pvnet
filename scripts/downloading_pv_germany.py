"""Download PV generation data from SMARD API for Germany."""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import xarray as xr

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://www.smard.de/app/chart_data"
FILTER_ID = 4068
REGION = "DE"


def get_timestamps(res="quarterhour"):
    """Get available timestamps from SMARD API."""
    url = f"{BASE_URL}/{FILTER_ID}/{REGION}/index_{res}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["timestamps"]


def get_chunk(ts, res="quarterhour"):
    """Download a data chunk from SMARD API."""
    url = f"{BASE_URL}/{FILTER_ID}/{REGION}/{FILTER_ID}_{REGION}_{res}_{ts}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["series"]


def download_pv_data(start_date: str, end_date: str, output_dir: Path):
    """Download PV generation data for date range."""
    logger.info(f"Downloading PV data from {start_date} to {end_date}")
    
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    timestamps = get_timestamps()
    timestamps = [ts for ts in timestamps if start_ts <= ts <= end_ts]
    
    logger.info(f"Found {len(timestamps)} timestamps to download")
    
    data = []
    for i, ts in enumerate(timestamps, 1):
        try:
            chunk = get_chunk(ts)
            if chunk:
                data.extend(chunk)
                logger.info(f"[{i}/{len(timestamps)}] Downloaded")
            else:
                logger.warning(f"[{i}/{len(timestamps)}] No data")
        except Exception as e:
            logger.error(f"[{i}/{len(timestamps)}] Failed: {e}")
        time.sleep(0.3)
    
    if not data:
        logger.error("No data downloaded")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["timestamp_ms", "generation_mw"])
    df["datetime_gmt"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df = df.drop_duplicates("timestamp_ms").sort_values("datetime_gmt")
    
    logger.info(f"Downloaded {len(df)} records")
    
    # Save CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "germany_pv.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")
    
    # Convert to Zarr
    df['gsp_id'] = 'DE'
    ds = xr.Dataset(
        {
            'generation_mw': (['datetime_gmt', 'gsp_id'], 
                             df.pivot_table(index='datetime_gmt', columns='gsp_id', 
                                          values='generation_mw').values)
        },
        coords={
            'datetime_gmt': df['datetime_gmt'].unique(),
            'gsp_id': df['gsp_id'].unique()
        }
    )
    
    zarr_path = output_dir / "germany_pv_2021.zarr"
    ds.to_zarr(zarr_path, mode="w", consolidated=True, zarr_version=2)
    logger.info(f"Saved Zarr to {zarr_path}")
    
    return zarr_path


def main():
    parser = argparse.ArgumentParser(description="Download PV generation data from SMARD API")
    parser.add_argument('--start-date', type=str, default='2021-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2021-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='./data/germany/generation', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    download_pv_data(args.start_date, args.end_date, output_dir)


if __name__ == '__main__':
    main()
