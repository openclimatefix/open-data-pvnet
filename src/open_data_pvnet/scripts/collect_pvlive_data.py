import pandas as pd
import logging
from open_data_pvnet.utils.env_loader import PROJECT_BASE
from pathlib import Path
from datetime import datetime
from fetch_pvlive_data import PVLiveData
from open_data_pvnet.utils.config_loader import load_config
from open_data_pvnet.utils.data_uploader import upload_to_huggingface
from open_data_pvnet.utils.data_converters import convert_nc_to_zarr
import pytz
import xarray as xr
import numpy as np
import os

logger = logging.getLogger(__name__)

# Define the configuration paths for UK and Global
CONFIG_PATHS = {
    "uk": PROJECT_BASE / "src/open_data_pvnet/configs/pvlive_data_config.yaml",
}

def collect_pvlive_data(
    year: int,
    month: int,
    day: int,
    hour: int,
    overwrite: bool = False,
):
    config_path = CONFIG_PATHS["uk"]
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    local_path = PROJECT_BASE / config["input_data"]["local_data"] / "target_data.nc"
    
    logger.info(f"Downloading PVlive data to {local_path}")
    print(f"Downloading PVlive data to {local_path}")

    pv = PVLiveData()

    start = datetime(year, month, day, hour, 0, 0, tzinfo=pytz.utc)
    end = datetime(year, month, day, hour, 0, 0, tzinfo=pytz.utc)

    data = pv.get_data_between(start=start, end=end, extra_fields="capacity_mwp")
    df = pd.DataFrame(data)

    df["datetime_gmt"] = pd.to_datetime(df["datetime_gmt"], utc=True)
    df["datetime_gmt"] = df["datetime_gmt"].dt.tz_convert(None)

    ds = xr.Dataset.from_dataframe(df)

    ds["datetime_gmt"] = ds["datetime_gmt"].astype(np.datetime64)

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
    overwrite: bool = False,
):
    local_path = collect_pvlive_data(year, month, day, hour, overwrite)
    if not local_path:
        logger.error("Failed to collect PVlive data.")
        return
    
    local_path = Path(local_path)

    local_path = local_path.parent
    output_dir = local_path / "zarr"
    convert_nc_to_zarr(local_path, output_dir, overwrite)

    upload_to_huggingface(config_path=CONFIG_PATHS["uk"],folder_name=local_path, year=year, month=month, day=day,overwrite=overwrite)

    logger.info(f"PVlive data for {year}-{month:02d}-{day:02d} at hour {hour:02d} uploaded successfully.")


if __name__ == "__main__":
    process_pvlive_data(2021, 1, 1, 0, overwrite=True)
