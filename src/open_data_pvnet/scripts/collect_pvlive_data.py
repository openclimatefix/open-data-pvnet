import pandas as pd
import logging
from datetime import datetime
from fetch_pvlive_data import PVLiveData
import pytz
import xarray as xr
import numpy as np
import os
import s3fs  # Import s3fs to interact with S3

# Initialize logger for logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize PVLiveData object to fetch PV data
pv = PVLiveData()

# Define the time range for data collection (from 2020 to 2025)
start = datetime(2020, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
end = datetime(2025, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)

# Fetch data using PVLive API, including the installed capacity in the output
data = pv.get_data_between(start=start, end=end, extra_fields="capacity_mwp")
df = pd.DataFrame(data)

# Process the data:
# - Convert datetime to UTC and remove timezone info
# - Sort data based on the datetime
df["datetime_gmt"] = pd.to_datetime(df["datetime_gmt"], utc=True)
df["datetime_gmt"] = df["datetime_gmt"].dt.tz_convert(None)
df = df.sort_values("datetime_gmt")

# Rename columns to match the expected dimension names
df = df.rename(columns={
    "datetime_gmt": "time_utc",
    "generation_mw": "gsp_pv_generation_mw",  # Generation data
    "capacity_mwp": "gsp_installedcapacity_mwp"  # Installed capacity data
})

# Add a gsp_id column with value 0 (this can be modified based on actual GSP ID)
df["gsp_id"] = 0

# Set index for easier access (time and gsp_id)
df = df.set_index(["time_utc", "gsp_id"])

# Convert the pandas DataFrame to an xarray Dataset
ds = xr.Dataset.from_dataframe(df)

# Rearrange dimensions so that the time dimension comes first
ds = ds.transpose("time_utc", "gsp_id")

# Saving the data to a local path
local_path = os.path.join(os.path.dirname(__file__), "..", "data", "gsp.zarr")

# Create the directories if they do not already exist
os.makedirs(os.path.dirname(local_path), exist_ok=True)

# Save the dataset as a Zarr file to the local path
ds.to_zarr(local_path, consolidated=True, mode="w")

logger.info(f"Zarr dataset successfully stored at: {local_path}")


# ---- Upload to S3 ----

# S3 bucket path - replace with your actual S3 bucket path
bucket_path = 's3://your-bucket-name/path/to/gsp.zarr'

# Initialize the S3 filesystem using s3fs
fs = s3fs.S3FileSystem()

# Open an S3 file and save the dataset in Zarr format to S3 (consolidated for better performance)
with fs.open(bucket_path, 'wb') as f:
    ds.to_zarr(f, consolidated=True, mode="w")

logger.info(f"Zarr dataset successfully stored at: {bucket_path}")
