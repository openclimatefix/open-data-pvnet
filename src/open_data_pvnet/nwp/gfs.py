import logging
from pathlib import Path
import xarray as xr
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from open_data_pvnet.utils.env_loader import PROJECT_BASE
from open_data_pvnet.utils.config_loader import load_config

logger = logging.getLogger(__name__)

def fetch_gfs_data(year, month, day, hour, config):
    """Downloads GFS GRIB2 files from NOAA S3 bucket."""
    s3_bucket = config.get("s3_bucket", "noaa-gfs-bdp-pds")
    local_output_dir = Path(PROJECT_BASE) / config["local_output_dir"] / "raw" / f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    
    interval_end = config.get("interval_end_minutes", 1080)
    resolution = config.get("time_resolution_minutes", 180)
    steps = range(0, (interval_end // 60) + 1, resolution // 60)
    
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    downloaded_files = []
    
    for step in steps:
        # Key format: gfs.20231201/00/atmos/gfs.t00z.pgrb2.0p25.f000
        s3_key = f"gfs.{year}{month:02d}{day:02d}/{hour:02d}/atmos/gfs.t{hour:02d}z.pgrb2.0p25.f{step:03d}"
        filename = Path(s3_key).name
        local_path = local_output_dir / filename
        
        if not local_path.exists():
            logger.info(f"Downloading {s3_key} from {s3_bucket}")
            try:
                s3.download_file(s3_bucket, s3_key, str(local_path))
                downloaded_files.append(local_path)
            except Exception as e:
                logger.error(f"Failed to download {s3_key}: {e}")
        else:
            downloaded_files.append(local_path)
            
    return downloaded_files

def convert_grib_to_zarr(files, output_path, config):
    """Converts downloaded GRIB files to a single Zarr dataset."""
    datasets = []
    needed_channels = config.get("channels", [])
    
    for f in files:
        try:
            # GFS GRIB files often contain multiple 'hypercubes' (e.g. surface vs atmosphere)
            # cfgrib handles this by returning a list of datasets if we use open_datasets (not available in xarray directly easily)
            # or we can try to merge them.
            # Simpler checks: open with default, if it errors about multiple, we might need specific backends.
            # Let's try xarray's open_dataset with backend_kwargs to define filter_keys if needed, or just iterate.
            # For now, simplest: use cfgrib directly to open all datasets? No, want xarray.
            # We will use open_dataset and catch errors? No.
            # Actually, `xr.open_dataset(..., engine='cfgrib')` explicitly fails if multiple messages.
            # We should use `xr.open_mfdataset`? No.
            
            # Use cfgrib.open_datasets to get all parts, then merge
            import cfgrib
            grib_datasets = cfgrib.open_datasets(str(f))
            
            # Merge variables from all parts of the GRIB file
            merged_ds = xr.merge(grib_datasets, compat='override')
            
            # Filter channels
            # Mapping might be needed. GFS names in cfgrib might be 't2m', 'u10' etc.
            # We'll select what matches or log warnings
            # This part is tricky without knowing exact mapping.
            # For this MVP, let's keep all variables but subset time/step if needed.
            
            # Add a step/time dimension if missing or ensure it's correct
            # GRIB files usually have valid_time.
            
            datasets.append(merged_ds)
            
        except Exception as e:
            logger.error(f"Error processing {f}: {e}")
            
    if not datasets:
        return None
        
    # Concatenate along step/time
    # GFS files are ONE step per file.
    full_ds = xr.concat(datasets, dim="step") # or valid_time?
    
    # Save to Zarr
    full_ds.to_zarr(output_path, mode="w")
    return output_path

def process_gfs_data(year, month, day, hour=None, region="global", overwrite=False):
    logger.info(f"Processing GFS data for {year}-{month} {day} {hour} region={region}")
    
    if region == "us":
        config_path = PROJECT_BASE / "src/open_data_pvnet/configs/gfs_us_data_config.yaml"
    elif region == "global":
        config_path = PROJECT_BASE / "src/open_data_pvnet/configs/gfs_data_config.yaml"
    else:
        raise ValueError(f"Invalid region for GFS: {region}")

    config = load_config(config_path)
    
    # Check if download is needed
    if region == "us" and "s3_bucket" in config["input_data"]["nwp"]["gfs"]:
        # US Archive Mode: Download from NOAA
        files = fetch_gfs_data(year, month, day, hour or 0, config["input_data"]["nwp"]["gfs"])
        
        # Convert
        local_output_dir = Path(PROJECT_BASE) / config["input_data"]["nwp"]["gfs"]["local_output_dir"]
        zarr_dir = local_output_dir / "zarr" / f"{year}-{month:02d}-{day:02d}-{hour or 0:02d}"
        
        if not zarr_dir.exists() or overwrite:
            convert_grib_to_zarr(files, zarr_dir, config["input_data"]["nwp"]["gfs"])
            logger.info(f"Converted GFS data to {zarr_dir}")
            
            # Cleanup raw ???
            # shutil.rmtree(files[0].parent)
            
    else:
        # Existing global logic?
        pass
