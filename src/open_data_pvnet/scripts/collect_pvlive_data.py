import pandas as pd
import logging
from open_data_pvnet.utils.env_loader import PROJECT_BASE
from pathlib import Path
from datetime import datetime, timedelta
from open_data_pvnet.scripts.fetch_pvlive_data import PVLiveData
from open_data_pvnet.utils.config_loader import load_config
from open_data_pvnet.utils.data_uploader import upload_to_huggingface
from open_data_pvnet.utils.data_converters import convert_nc_to_zarr
import pytz
import os
import calendar
import shutil

logger = logging.getLogger(__name__)

# Define the configuration paths for UK and Global
CONFIG_PATHS = {
    "uk": PROJECT_BASE / "src/open_data_pvnet/configs/pvlive_data_config.yaml",
}

def collect_pvlive_data(
    year: int,
    month: int,
    overwrite: bool = False,
):
    config_path = CONFIG_PATHS["uk"]
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    local_path = PROJECT_BASE / config["input_data"]["local_data"] / f"target_data_{year}_{month:02d}.nc"
    
    logger.info(f"Downloading PVlive data to {local_path}")
    print(f"Downloading PVlive data to {local_path}")

    pv = PVLiveData()

    start = datetime(year, month, 1, 0, 0, 0, tzinfo=pytz.utc)
    end = datetime(year, month, calendar.monthrange(year, month)[1], 23, 59, 59, tzinfo=pytz.utc)

    df = pv.get_data_between(start=start, end=end, period=30, extra_fields="capacity_mwp,installedcapacity_mwp")
    df = pd.DataFrame(df).reset_index()

    df["datetime_gmt"] = pd.to_datetime(df["datetime_gmt"], utc=True)
    df["datetime_gmt"] = df["datetime_gmt"].dt.tz_convert(None)
    
    ds = df.set_index(["gsp_id", "datetime_gmt"]).to_xarray()
    ds = ds.transpose("gsp_id", "datetime_gmt")

    if not overwrite and os.path.exists(local_path):
        logger.info(f"File {local_path} already exists and overwrite is set to False.")
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        ds.to_netcdf(local_path)
        logger.info(f"PVLive data stored successfully in {local_path}")

    return local_path

def process_pvlive_data(
    year: int,
    month: int,
    overwrite: bool = False,
):
    local_path = collect_pvlive_data(year, month, overwrite)
    if not local_path:
        logger.error(f"Failed to collect PVlive data for {year}-{month:02d}.")
        return
    
    local_path = Path(local_path)

    local_path = local_path.parent
    output_dir = local_path / "zarr" / f"target_data_{year}_{month:02d}"
    convert_nc_to_zarr(local_path, output_dir, overwrite)

    upload_to_huggingface(config_path=CONFIG_PATHS["uk"], folder_name=output_dir.name, year=year, month=month, overwrite=overwrite)

    # Remove the local files
    os.remove(local_path / f"target_data_{year}_{month:02d}.nc")
    ## remove contents in the folder as well
    
    shutil.rmtree(local_path)
    logger.info(f"PVlive data for {year}-{month:02d} uploaded successfully.")


if __name__ == "__main__":
    process_pvlive_data(2023, 1, overwrite=True)