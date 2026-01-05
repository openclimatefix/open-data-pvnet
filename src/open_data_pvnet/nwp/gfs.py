import logging
from pathlib import Path
import xarray as xr
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import shutil
from open_data_pvnet.utils.env_loader import PROJECT_BASE
from open_data_pvnet.utils.config_loader import load_config

logger = logging.getLogger(__name__)

def fetch_gfs_data(year, month, day, hour, config):
    """Downloads GFS GRIB2 files from NOAA S3 bucket."""
    s3_bucket = config.get("s3_bucket", "noaa-gfs-bdp-pds")
    
    # Determine output directory, default to a tmp location if not specified
    output_dir_rel = config.get("local_output_dir", "tmp/gfs/data")
    local_output_dir = Path(PROJECT_BASE) / output_dir_rel / "raw" / f"{year}-{month:02d}-{day:02d}-{hour:02d}"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    
    interval_end = config.get("interval_end_minutes", 1080)
    resolution = config.get("time_resolution_minutes", 180)
    # Generate steps (e.g., 0, 3, 6 ... hours)
    steps = range(0, (interval_end // 60) + 1, resolution // 60)
    
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    downloaded_files = []
    
    for step in steps:
        # Key format: gfs.20231201/00/atmos/gfs.t00z.pgrb2.0p25.f000
        # This structure matches the NOAA GFS bucket
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
    # Import cfgrib here to avoid hard dependency at module level
    import cfgrib
    
    datasets = []
    needed_channels = set(config.get("channels", []))
    
    for f in files:
        try:
            # GFS GRIB files often contain multiple 'hypercubes' (variable groups with different dims)
            # cfgrib.open_datasets handles this by returning a list of xarray Datasets
            grib_datasets = cfgrib.open_datasets(str(f))
            
            if not grib_datasets:
                logger.warning(f"No datasets found in {f}")
                continue

            # Merge variables from all parts of the GRIB file
            # compat='override' is often necessary if coordinates differ slightly due to precision
            merged_ds = xr.merge(grib_datasets, compat='override')
            
            # Filter channels
            if needed_channels:
                available_vars = set(merged_ds.data_vars)
                # Keep only what is in needed_channels (intersection)
                vars_to_keep = available_vars.intersection(needed_channels)
                
                if not vars_to_keep:
                    logger.warning(f"No matching channels found in {f}. Available: {available_vars}. Requested: {needed_channels}")
                    # Decide whether to continue empty or skip. Keeping empty might break downstream.
                    # We'll skip this file's contribution if it lacks all desired data.
                    # Or we could just warn.
                else:
                    merged_ds = merged_ds[list(vars_to_keep)]
            
            # GFS files are usually one 'step' per file
            # We assume the list of files is ordered by step
            datasets.append(merged_ds)
            
        except Exception as e:
            logger.error(f"Error processing {f}: {e}")
            continue
            
    if not datasets:
        logger.error("No GRIB datasets could be processed.")
        return None
        
    try:
        # Concatenate along step/time
        # Note: GRIB files often load with a 'step' dimension if valid_time is different but ref_time is same
        full_ds = xr.concat(datasets, dim="step")
        
        # Save to Zarr
        full_ds.to_zarr(output_path, mode="w")
        logger.info(f"Successfully saved Zarr to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error during final concat/save to Zarr: {e}")
        return None

def process_gfs_data(year, month, day, hour=None, region="global", overwrite=False):
    logger.info(f"Processing GFS data for {year}-{month} {day} {hour} region={region}")
    
    if region == "us":
        config_path = PROJECT_BASE / "src/open_data_pvnet/configs/gfs_us_data_config.yaml"
    elif region == "global":
        config_path = PROJECT_BASE / "src/open_data_pvnet/configs/gfs_data_config.yaml"
    else:
        raise ValueError(f"Invalid region for GFS: {region}")
    
    if not config_path.exists():
         raise FileNotFoundError(f"Config file not found at {config_path}")

    config = load_config(config_path)
    # Extract GFS specific config
    gfs_config = config.get("input_data", {}).get("nwp", {}).get("gfs", {})
    
    if not gfs_config:
        logger.error("No GFS configuration found in input_data.nwp.gfs")
        return

    # Fetch data
    # (Hour defaults to 00 if None, typical for daily run start)
    target_hour = hour if hour is not None else 0
    files = fetch_gfs_data(year, month, day, target_hour, gfs_config)
    
    if not files:
        logger.error("No files were downloaded.")
        return

    # Determine Output Path
    output_dir_rel = gfs_config.get("local_output_dir", f"tmp/gfs/{region}")
    local_output_dir = Path(PROJECT_BASE) / output_dir_rel
    zarr_dir = local_output_dir / "zarr" / f"{year}-{month:02d}-{day:02d}-{target_hour:02d}"
    
    if not zarr_dir.exists() or overwrite:
        result = convert_grib_to_zarr(files, zarr_dir, gfs_config)
        
        if result:
            # Cleanup raw files to save space
            raw_dir = files[0].parent
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
                logger.info(f"Cleaned up raw files in {raw_dir}")
    else:
        logger.info(f"Output Zarr already exists at {zarr_dir}. Use overwrite=True to replace.")
