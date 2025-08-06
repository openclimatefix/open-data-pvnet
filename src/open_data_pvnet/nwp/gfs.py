import logging
import xarray as xr

logger = logging.getLogger(__name__)

def process_gfs_data(year, month):
    """
    Download and preprocess GFS data for the given year/month:
      1. Open the NetCDF/GRIB file
      2. Re-map longitudes to 0–360 if needed
      3. Sort coords so they’re monotonic
    Returns an xarray.Dataset.
    """
    logger.info(f"Downloading GFS data for {year}-{month:02d}")

    # 1) construct path or URL to your downloaded file
    #    adjust this to wherever you store your GFS files
    netcdf_path = f"/data/gfs/{year}/{month:02d}/gfs_latest.nc"

    # 2) open dataset
    ds = xr.open_dataset(netcdf_path)

    # 3a) ensure all longitudes sit in 0–360 rather than –180..180
    lon = ds.coords["longitude"]
    if lon.min() < 0:
        ds = ds.assign_coords(longitude=((lon + 360) % 360))

    # 3b) make sure lat & lon coords are monotonic increasing
    if not ds.latitude.is_monotonic_increasing:
        ds = ds.sortby("latitude")
    if not ds.longitude.is_monotonic_increasing:
        ds = ds.sortby("longitude")

    return ds
