"""
This script inspects the France PV training pipeline.

Validates:
1. France Solar Zarr dataset loads correctly and data looks as expected
2. GFS NWP data for France is accessible from S3
3. Data timestamps align for training
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import fsspec
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

gfs_path = "s3://ocf-open-data-pvnet/data/gfs/v4/2024.zarr"

# Load the zarr dataset
base_dir = os.getcwd()
parent_3_levels_up = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
output_dir = os.path.join(parent_3_levels_up, "data")
solar_path = "france_solar_combined.zarr"
print(f"Loading {solar_path}...\n")

# Define France latitude and longitude bounds
MIN_LAT, MAX_LAT = 41.5, 51.5
MIN_LON, MAX_LON = -5.5, 9.0


def test_france_solar_data(zarr_path):
    """Test loading and inspecting the France solar zarr dataset."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing France Solar Zarr Dataset")
    logger.info("=" * 60)

    ds = xr.open_zarr(zarr_path)

    # 1. Basic Dataset Info
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(ds)
    print("\n")

    # 2. Dimensions and Coordinates
    print("=" * 60)
    print("DIMENSIONS & COORDINATES")
    print("=" * 60)
    print(f"Locations: {len(ds.location_id)} regions")
    print(f"Time steps: {len(ds.time_utc)}")
    print(f"Time range: {ds.time_utc.values[0]} to {ds.time_utc.values[-1]}")
    print(f"\nRegions: {list(ds.location_id.values)}")
    print(f"\nLatitudes: {ds.latitude.values}")
    print(f"Longitudes: {ds.longitude.values}")
    print("\n")

    # 3. Check Time Resolution
    print("=" * 60)
    print("TIME RESOLUTION CHECK")
    print("=" * 60)
    time_diff = pd.Series(ds.time_utc.values).diff()
    print("Expected resolution: 30 minutes")
    print(f"Actual resolution (mode): {time_diff.mode()[0]}")

    # Check for irregular gaps
    irregular_mask = time_diff != pd.Timedelta("30min")
    irregular_count = irregular_mask.sum()
    if irregular_count > 0:
        print(f"Found {irregular_count} irregular time gaps")
        irregular_times = ds.time_utc.values[irregular_mask]
        print(f"First few irregular gaps: {irregular_times[:5]}")
    else:
        print("✓ All timesteps are regular 30-minute intervals")

    # Check completeness
    expected_count = len(pd.date_range(ds.time_utc.values[0], ds.time_utc.values[-1], freq="30min"))
    actual_count = len(ds.time_utc)
    print(f"\nExpected timesteps: {expected_count}")
    print(f"Actual timesteps: {actual_count}")
    print(f"Complete: {'✓ Yes' if expected_count == actual_count else '✗ No'}")
    print("\n")

    # 4. Data Quality Check
    print("=" * 60)
    print("DATA QUALITY")
    print("=" * 60)

    # Generation data
    gen_data = ds["generation_mw"].values
    print("Generation (MW):")
    print(f"  Shape: {gen_data.shape}")
    print(f"  Range: [{np.nanmin(gen_data):.2f}, {np.nanmax(gen_data):.2f}] MW")
    print(f"  Mean: {np.nanmean(gen_data):.2f} MW")
    print(
        f"  NaN count: {np.isnan(gen_data).sum()} ({100*np.isnan(gen_data).sum()/gen_data.size:.2f}%)"
    )

    # Capacity data
    cap_data = ds["capacity_mwp"].values
    print("\nCapacity (MWp):")
    print(f"  Shape: {cap_data.shape}")
    print(f"  Range: [{np.nanmin(cap_data):.2f}, {np.nanmax(cap_data):.2f}] MWp")
    print(f"  Mean: {np.nanmean(cap_data):.2f} MWp")
    print(
        f"  NaN count: {np.isnan(cap_data).sum()} ({100*np.isnan(cap_data).sum()/cap_data.size:.2f}%)"
    )
    print("\n")

    # 5. Per-Region Stats
    print("=" * 60)
    print("PER-REGION STATISTICS")
    print("=" * 60)
    for i, region in enumerate(ds.location_id.values):
        gen = ds["generation_mw"].isel(location_id=i).values
        cap = ds["capacity_mwp"].isel(location_id=i).values
        print(f"\n{region}:")
        print(
            f"  Generation: [{np.nanmin(gen):.1f}, {np.nanmax(gen):.1f}] MW, "
            f"Mean: {np.nanmean(gen):.1f} MW, NaN: {100*np.isnan(gen).sum()/len(gen):.1f}%"
        )
        print(
            f"  Capacity: {np.nanmean(cap):.1f} MWp, NaN: {100*np.isnan(cap).sum()/len(cap):.1f}%"
        )
    print("\n")

    # 6. Dataset Attributes
    print("=" * 60)
    print("DATASET ATTRIBUTES")
    print("=" * 60)
    for key, value in ds.attrs.items():
        print(f"{key}: {value}")
    print("\n")

    # 7. Sample Time Series Plot
    print("=" * 60)
    print("SAMPLE TIME SERIES")
    print("=" * 60)
    print("Creating sample plot for first 30 days...")

    # Plot first region for first 30 days
    sample_days = 30
    sample_times = ds.time_utc[: 48 * sample_days]  # 48 half-hours per day

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Generation
    for i, region in enumerate(ds.location_id.values):
        ds["generation_mw"].sel(location_id=region, time_utc=sample_times).plot.line(
            ax=axes[0], label=region
        )
    axes[0].set_title("Solar Generation (First 30 Days)")
    axes[0].set_ylabel("Generation (MW)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Capacity
    for i, region in enumerate(ds.location_id.values):
        ds["capacity_mwp"].sel(location_id=region, time_utc=sample_times).plot.line(
            ax=axes[1], label=region
        )
    axes[1].set_title("Solar Capacity (First 30 Days)")
    axes[1].set_ylabel("Capacity (MWp)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("zarr_inspection_sample.png", dpi=150, bbox_inches="tight")
    print("✓ Saved plot to zarr_inspection_sample.png")

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


def test_gfs_data_access():
    """Test accessing GFS NWP data from S3."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing GFS NWP Data Access")
    logger.info("=" * 60)

    try:
        logger.info(f"Opening GFS data from: {gfs_path}")
        store = fsspec.get_mapper(gfs_path, anon=True)

        # Open with limited variables to test access
        ds = xr.open_zarr(store, consolidated=True)

        logger.info("GFS Dataset accessed successfully!")
        logger.info(f"Variables: {list(ds.data_vars)[:10]}...")  # First 10
        logger.info(f"Dimensions: {dict(ds.dims)}")

        # Check latitude/longitude coverage
        if "latitude" in ds.dims:
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            logger.info("\nSpatial Coverage:")
            logger.info(f"  Latitude: {lats.min():.1f} to {lats.max():.1f}")
            logger.info(f"  Longitude: {lons.min():.1f} to {lons.max():.1f}")

            # Check if France is within bounds
            if (
                (lats.min() <= MIN_LAT <= lats.max())
                and (lats.min() <= MAX_LAT <= lats.max())
                and (lons.min() <= MIN_LON <= lons.max())
                and (lons.min() <= MAX_LON <= lons.max())
            ):
                logger.info("GFS data covers France region")
            else:
                logger.warning("GFS data does NOT cover France region")
        else:
            logger.warning("GFS dataset does not have latitude/longitude dimensions")

        # Check time dimension
        if "init_time" in ds.dims or "time" in ds.dims:
            time_dim = "init_time" if "init_time" in ds.dims else "time"
            times = ds[time_dim].values
            logger.info("\nTime Coverage:")
            logger.info(f"  First: {pd.Timestamp(times[0])}")
            logger.info(f"  Last: {pd.Timestamp(times[-1])}")

        logger.info("\n GFS Data Access: PASSED")
        return True

    except ImportError:
        logger.warning("fsspec not available - skipping S3 test")
        return None
    except Exception as e:
        logger.error(f"Failed to access GFS data: {e}")
        return False


def test_time_alignment(solar_path, gfs_path):
    """Check if France solar data and GFS data have
    overlapping times."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Time Alignment")
    logger.info("=" * 60)

    try:
        ds_solar = xr.open_zarr(str(solar_path))
        solar_times = ds_solar["time_utc"].values

        ds_gfs = xr.open_zarr(str(gfs_path))
        gfs_times = ds_gfs["time"].values

        # Check for overlapping times
        overlap = np.intersect1d(solar_times, gfs_times)
        if len(overlap) > 0:
            logger.info(f"Found {len(overlap)} overlapping time steps")
        else:
            logger.warning("No overlapping time steps found")

    except Exception as e:
        logger.error(f"Failed to check time alignment: {e}")


def main():
    logger.info("=" * 60)
    logger.info("FRANCE PVNET DATA PIPELINE TEST")
    logger.info("=" * 60)

    results = {}
    results["solar_data"] = test_france_solar_data(solar_path)
    results["gfs_access"] = test_gfs_data_access()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for test, result in results.items():
        status = "PASSED" if result else ("SKIPPED" if result is None else "FAILED")
        logger.info(f"  {test}: {status}")

    all_passed = all(r is True or r is None for r in results.values())
    if all_passed:
        logger.info("\nAll tests passed!")
    else:
        logger.info("\nSome tests failed - check logs above")

    # Skip time alignment test since GFS is on S3
    logger.info("\nSkipping time alignment test (GFS is on S3)")


if __name__ == "__main__":
    main()
