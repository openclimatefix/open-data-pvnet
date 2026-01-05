"""
Test EIA Data Compatibility with ocf-data-sampler

This script tests if the preprocessed EIA data can be loaded by ocf-data-sampler
and matches the expected format.

Usage:
    python src/open_data_pvnet/scripts/test_eia_sampler_compatibility.py \
        --data-path src/open_data_pvnet/data/target_eia_data_processed.zarr \
        --config-path src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/us_configuration.yaml
"""

import argparse
import logging
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_zarr_structure(data_path: str) -> bool:
    """Test basic Zarr structure and format."""
    logger.info(f"Testing Zarr structure: {data_path}")
    
    try:
        ds = xr.open_dataset(data_path, engine="zarr")
        logger.info(f"✅ Successfully opened Zarr dataset")
        logger.info(f"   Dimensions: {dict(ds.dims)}")
        logger.info(f"   Variables: {list(ds.data_vars)}")
        logger.info(f"   Coordinates: {list(ds.coords)}")
        
        # Check required dimensions
        required_dims = ["ba_id", "datetime_gmt"]
        missing_dims = [d for d in required_dims if d not in ds.dims]
        if missing_dims:
            logger.error(f"❌ Missing required dimensions: {missing_dims}")
            return False
        logger.info(f"✅ Required dimensions present: {required_dims}")
        
        # Check required variables
        required_vars = ["generation_mw", "capacity_mwp"]
        missing_vars = [v for v in required_vars if v not in ds.data_vars]
        if missing_vars:
            logger.error(f"❌ Missing required variables: {missing_vars}")
            return False
        logger.info(f"✅ Required variables present: {required_vars}")
        
        # Check datetime_gmt format
        if "datetime_gmt" in ds.coords:
            dt_coord = ds.coords["datetime_gmt"]
            if not np.issubdtype(dt_coord.dtype, np.datetime64):
                logger.warning(f"⚠️  datetime_gmt is not datetime64: {dt_coord.dtype}")
            else:
                logger.info(f"✅ datetime_gmt is datetime64: {dt_coord.dtype}")
        
        # Check ba_id format
        if "ba_id" in ds.coords:
            ba_coord = ds.coords["ba_id"]
            if not np.issubdtype(ba_coord.dtype, np.integer):
                logger.warning(f"⚠️  ba_id is not integer: {ba_coord.dtype}")
            else:
                logger.info(f"✅ ba_id is integer: {ba_coord.dtype}")
        
        # Check data ranges
        gen_data = ds["generation_mw"]
        cap_data = ds["capacity_mwp"]
        
        logger.info(f"   Generation range: {float(gen_data.min().values):.2f} - {float(gen_data.max().values):.2f} MW")
        logger.info(f"   Capacity range: {float(cap_data.min().values):.2f} - {float(cap_data.max().values):.2f} MW")
        
        # Check that capacity >= generation (with tolerance)
        if (gen_data > cap_data * 1.1).any():
            logger.warning("⚠️  Some generation values exceed capacity (may be acceptable)")
        else:
            logger.info("✅ Capacity >= generation (with tolerance)")
        
        ds.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to open/validate Zarr: {e}", exc_info=True)
        return False


def test_ocf_data_sampler_compatibility(data_path: str, config_path: str) -> bool:
    """Test compatibility with ocf-data-sampler."""
    logger.info(f"Testing ocf-data-sampler compatibility")
    
    try:
        from ocf_data_sampler.config import load_yaml_configuration
        from ocf_data_sampler.torch_datasets.utils.valid_time_periods import find_valid_time_periods
        
        # Load configuration
        config = load_yaml_configuration(config_path)
        logger.info(f"✅ Loaded configuration from {config_path}")
        
        # Load data
        ds = xr.open_dataset(data_path, engine="zarr")
        logger.info(f"✅ Loaded data from {data_path}")
        
        # Test find_valid_time_periods
        # This is the key function that ocf-data-sampler uses
        try:
            valid_times = find_valid_time_periods({"gsp": ds}, config)
            logger.info(f"✅ find_valid_time_periods succeeded")
            logger.info(f"   Found {len(valid_times)} valid time periods")
            if len(valid_times) > 0:
                logger.info(f"   First valid time: {valid_times.iloc[0]}")
                logger.info(f"   Last valid time: {valid_times.iloc[-1]}")
            return True
        except Exception as e:
            logger.error(f"❌ find_valid_time_periods failed: {e}", exc_info=True)
            return False
        finally:
            ds.close()
            
    except ImportError as e:
        logger.warning(f"⚠️  ocf-data-sampler not available: {e}")
        logger.warning("   Install with: pip install ocf-data-sampler")
        return False
    except Exception as e:
        logger.error(f"❌ ocf-data-sampler compatibility test failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test EIA data compatibility with ocf-data-sampler"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to preprocessed EIA Zarr file"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/us_configuration.yaml",
        help="Path to ocf-data-sampler configuration file"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Testing EIA Data Compatibility with ocf-data-sampler")
    logger.info("=" * 60)
    
    # Test 1: Zarr structure
    structure_ok = test_zarr_structure(args.data_path)
    
    # Test 2: ocf-data-sampler compatibility
    sampler_ok = test_ocf_data_sampler_compatibility(args.data_path, args.config_path)
    
    logger.info("=" * 60)
    if structure_ok and sampler_ok:
        logger.info("✅ All tests passed! Data is compatible with ocf-data-sampler.")
        return 0
    elif structure_ok:
        logger.warning("⚠️  Basic structure OK, but ocf-data-sampler test failed or skipped.")
        return 1
    else:
        logger.error("❌ Tests failed. Data format needs correction.")
        return 1


if __name__ == "__main__":
    exit(main())


