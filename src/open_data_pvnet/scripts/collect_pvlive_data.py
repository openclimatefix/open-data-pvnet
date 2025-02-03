import pandas as pd
import logging
from datetime import datetime
from fetch_pvlive_data import PVLiveData
from open_data_pvnet.utils.data_uploader import upload_to_huggingface
import pytz
import xarray as xr
import numpy as np
import os

logger = logging.getLogger(__name__)

def collect_pvlive_data(
    year: int,
    month: int,
    day: int,
    hour: int,
    overwrite: bool = False,
):
    pv = PVLiveData()

    start = datetime(year, month, day, hour, 0, 0, tzinfo=pytz.utc)
    end = datetime(year, month, day, hour, 0, 0, tzinfo=pytz.utc)

    data = pv.get_data_between(start=start, end=end, extra_fields="capacity_mwp")
    df = pd.DataFrame(data)

    df["datetime_gmt"] = pd.to_datetime(df["datetime_gmt"], utc=True)
    df["datetime_gmt"] = df["datetime_gmt"].dt.tz_convert(None)

    ds = xr.Dataset.from_dataframe(df)

    ds["datetime_gmt"] = ds["datetime_gmt"].astype(np.datetime64)

    local_path = os.path.join(os.path.dirname(__file__), "..", "target_data", "target_data.nc")

    if not overwrite and os.path.exists(local_path):
        logger.info(f"File {local_path} already exists and overwrite is set to False.")
        return None

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    ds.to_netcdf(local_path)

    logger.info(f"PVlive data stored successfully in {local_path}")

    return local_path
    


def process_pvlive_data(
    year: int,
    month: int,
    day: int,
    hour: int,
    region: str,
    overwrite: bool = False,
    archive_type: str = "zarr.zip",
):
    local_path = collect_pvlive_data(year, month, day, hour, region, overwrite, archive_type)

    if not local_path:
        logger.error("Failed to collect PVlive data.")
        return
    
    upload_to_huggingface(local_path, year, month, day, overwrite, archive_type)

    logger.info(f"PVlive data for {year}-{month:02d}-{day:02d} at hour {hour:02d} uploaded successfully.")



