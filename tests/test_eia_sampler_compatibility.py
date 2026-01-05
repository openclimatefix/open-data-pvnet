import pytest
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@pytest.fixture
def data_path():
    return "src/open_data_pvnet/data/target_eia_data_processed.zarr"

@pytest.fixture
def config_path():
    return "src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/us_configuration.yaml"


def test_zarr_structure(data_path):
    """Test basic Zarr structure and format."""
    if not Path(data_path).exists():
        pytest.skip(f"Data file not found at {data_path}")

    logger.info(f"Testing Zarr structure: {data_path}")
    
    try:
        ds = xr.open_dataset(data_path, engine="zarr")
        
        # Check required dimensions
        required_dims = ["ba_id", "datetime_gmt"]
        missing_dims = [d for d in required_dims if d not in ds.dims]
        if missing_dims:
            pytest.fail(f"Missing required dimensions: {missing_dims}")
        
        # Check required variables
        required_vars = ["generation_mw", "capacity_mwp"]
        missing_vars = [v for v in required_vars if v not in ds.data_vars]
        if missing_vars:
            pytest.fail(f"Missing required variables: {missing_vars}")
        
        # Check datetime_gmt format
        if "datetime_gmt" in ds.coords:
            dt_coord = ds.coords["datetime_gmt"]
            if not np.issubdtype(dt_coord.dtype, np.datetime64):
                logger.warning(f"datetime_gmt is not datetime64: {dt_coord.dtype}")

        # Check ba_id format
        if "ba_id" in ds.coords:
            ba_coord = ds.coords["ba_id"]
            if not np.issubdtype(ba_coord.dtype, np.integer):
                logger.warning(f"ba_id is not integer: {ba_coord.dtype}")
        
        # Check data ranges
        gen_data = ds["generation_mw"]
        cap_data = ds["capacity_mwp"]
        
        # Check that capacity >= generation (with tolerance)
        if (gen_data > cap_data * 1.1).any():
            logger.warning("Some generation values exceed capacity")
        
        ds.close()
        
    except Exception as e:
        pytest.fail(f"Failed to open/validate Zarr: {e}")


def test_ocf_data_sampler_compatibility(data_path, config_path):
    """Test compatibility with ocf-data-sampler."""
    if not Path(data_path).exists():
        pytest.skip(f"Data file not found at {data_path}")

    logger.info(f"Testing ocf-data-sampler compatibility")
    
    try:
        from ocf_data_sampler.config import load_yaml_configuration
        
        # Load configuration
        config = load_yaml_configuration(config_path)
        
        # Load data
        ds = xr.open_dataset(data_path, engine="zarr")
        
        # Test find_valid_time_periods
        try:
             # Importing here to avoiding top level failure if package is missing
            from ocf_data_sampler.torch_datasets.utils.valid_time_periods import find_valid_time_periods
            valid_times = find_valid_time_periods({"gsp": ds}, config)
            
            logger.info(f"Found {len(valid_times)} valid time periods")
            
        except ImportError:
             pytest.skip("ocf_data_sampler not installed or internal path changed")
        except Exception as e:
            pytest.fail(f"find_valid_time_periods failed: {e}")
        finally:
            ds.close()
            
    except ImportError as e:
        pytest.skip(f"ocf-data-sampler not available: {e}")
    except Exception as e:
        pytest.fail(f"ocf-data-sampler compatibility test failed: {e}")
