import pandas as pd
import logging
from datetime import datetime
from fetch_pvlive_data import PVLiveData
import pytz
import xarray as xr
import numpy as np
import os

logger = logging.getLogger(__name__)

pv = PVLiveData()

start = datetime(2020, 1, 1, 0, 0, 0, 0, tzinfo=pytz.UTC)
end = datetime(2025, 1, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)

data = pv.get_data_between(start=start, end=end, extra_fields="capacity_mwp")
df = pd.DataFrame(data)

df["datetime_gmt"] = pd.to_datetime(df["datetime_gmt"], utc=True)
df["datetime_gmt"] = df["datetime_gmt"].dt.tz_convert(None)

ds = xr.Dataset.from_dataframe(df)

ds["datetime_gmt"] = ds["datetime_gmt"].astype(np.datetime64)

local_path = os.path.join(os.path.dirname(__file__), "..", "data", "target_data.nc")

os.makedirs(os.path.dirname(local_path), exist_ok=True)
ds.to_netcdf(local_path)

logger.info(f"Data successfully stored in {local_path}")
