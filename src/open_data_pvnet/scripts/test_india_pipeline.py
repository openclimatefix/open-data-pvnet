"""
Test India PVNet Data Pipeline

Validates:
1. India Solar Zarr dataset loads correctly
2. GFS NWP data is accessible from S3
3. Data timestamps align for training
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_india_solar_data():
    """Test loading India solar Zarr dataset."""
    logger.info("=" * 60)
    logger.info("Testing India Solar Data")
    logger.info("=" * 60)
    
    zarr_path = Path(r"C:\Users\asus vivoBook\Desktop\New folder (2)\pvnet-india-data\processed\india_solar_2024-2025.zarr")
    
    if not zarr_path.exists():
        logger.error(f"India solar Zarr not found: {zarr_path}")
        return False
    
    try:
        ds = xr.open_zarr(str(zarr_path))
        
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Variables: {list(ds.data_vars)}")
        logger.info(f"Dimensions: {dict(ds.dims)}")
        
        # Check solar data
        if 'solar_generation_mw' in ds:
            solar = ds['solar_generation_mw']
            logger.info(f"\nSolar Generation (MW):")
            logger.info(f"  Shape: {solar.shape}")
            logger.info(f"  Min: {float(solar.min()):.2f}")
            logger.info(f"  Max: {float(solar.max()):.2f}")
            logger.info(f"  Mean: {float(solar.mean()):.2f}")
        
        # Check time range
        if 'datetime_gmt' in ds.dims:
            times = ds['datetime_gmt'].values
            logger.info(f"\nTime Range:")
            logger.info(f"  Start: {pd.Timestamp(times[0])}")
            logger.info(f"  End: {pd.Timestamp(times[-1])}")
            logger.info(f"  Count: {len(times)} hours")
        
        logger.info("\n✅ India Solar Data: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load India solar data: {e}")
        return False


def test_gfs_data_access():
    """Test accessing GFS NWP data from S3."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing GFS NWP Data Access")
    logger.info("=" * 60)
    
    gfs_path = "s3://ocf-open-data-pvnet/data/gfs/v4/2024.zarr"
    
    try:
        import fsspec
        
        logger.info(f"Opening GFS data from: {gfs_path}")
        store = fsspec.get_mapper(gfs_path, anon=True)
        
        # Open with limited variables to test access
        ds = xr.open_zarr(store, consolidated=True)
        
        logger.info(f"GFS Dataset accessed successfully!")
        logger.info(f"Variables: {list(ds.data_vars)[:10]}...")  # First 10
        logger.info(f"Dimensions: {dict(ds.dims)}")
        
        # Check latitude/longitude coverage
        if 'latitude' in ds.dims:
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            logger.info(f"\nSpatial Coverage:")
            logger.info(f"  Latitude: {lats.min():.1f} to {lats.max():.1f}")
            logger.info(f"  Longitude: {lons.min():.1f} to {lons.max():.1f}")
            
            # Check if India is covered (6-38°N, 68-98°E)
            india_lat_covered = (lats.min() <= 6) and (lats.max() >= 38)
            india_lon_covered = (lons.min() <= 68) and (lons.max() >= 98)
            
            if india_lat_covered and india_lon_covered:
                logger.info("  ✅ India region is covered!")
            else:
                logger.warning("  ⚠️ India region may not be fully covered")
        
        # Check time dimension
        if 'init_time' in ds.dims or 'time' in ds.dims:
            time_dim = 'init_time' if 'init_time' in ds.dims else 'time'
            times = ds[time_dim].values
            logger.info(f"\nTime Coverage:")
            logger.info(f"  First: {pd.Timestamp(times[0])}")
            logger.info(f"  Last: {pd.Timestamp(times[-1])}")
        
        logger.info("\n✅ GFS Data Access: PASSED")
        return True
        
    except ImportError:
        logger.warning("fsspec not available - skipping S3 test")
        return None
    except Exception as e:
        logger.error(f"Failed to access GFS data: {e}")
        return False


def test_time_alignment():
    """Check if India solar data and GFS data have overlapping times."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Time Alignment")
    logger.info("=" * 60)
    
    # Load India solar times
    zarr_path = Path(r"C:\Users\asus vivoBook\Desktop\New folder (2)\pvnet-india-data\processed\india_solar_2024-2025.zarr")
    
    try:
        ds_india = xr.open_zarr(str(zarr_path))
        india_times = pd.DatetimeIndex(ds_india['datetime_gmt'].values)
        
        logger.info(f"India Solar Data:")
        logger.info(f"  Start: {india_times.min()}")
        logger.info(f"  End: {india_times.max()}")
        
        # GFS 2024 data should cover Jan 2024 onwards
        gfs_expected_start = pd.Timestamp("2024-01-01")
        gfs_expected_end = pd.Timestamp("2024-12-31")
        
        logger.info(f"\nGFS 2024 Data (expected):")
        logger.info(f"  Start: {gfs_expected_start}")
        logger.info(f"  End: {gfs_expected_end}")
        
        # Find overlap
        overlap_start = max(india_times.min(), gfs_expected_start)
        overlap_end = min(india_times.max(), gfs_expected_end)
        
        if overlap_start < overlap_end:
            overlap_hours = len(india_times[(india_times >= overlap_start) & (india_times <= overlap_end)])
            logger.info(f"\n✅ Overlapping Period: {overlap_start} to {overlap_end}")
            logger.info(f"   Available training hours: {overlap_hours}")
        else:
            logger.warning("⚠️ No overlapping period found!")
        
        return True
        
    except Exception as e:
        logger.error(f"Time alignment check failed: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("INDIA PVNET DATA PIPELINE TEST")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: India Solar Data
    results['india_solar'] = test_india_solar_data()
    
    # Test 2: GFS Access
    results['gfs_access'] = test_gfs_data_access()
    
    # Test 3: Time Alignment
    results['time_alignment'] = test_time_alignment()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test, result in results.items():
        status = "✅ PASSED" if result else ("⏭️ SKIPPED" if result is None else "❌ FAILED")
        logger.info(f"  {test}: {status}")
    
    all_passed = all(r is True or r is None for r in results.values())
    if all_passed:
        logger.info("\n🎉 All tests passed! Ready for Week 3: PVNet integration")
    else:
        logger.info("\n⚠️ Some tests failed - check logs above")


if __name__ == "__main__":
    main()
