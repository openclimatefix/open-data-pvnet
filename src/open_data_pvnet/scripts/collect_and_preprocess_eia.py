"""
Collect and Preprocess EIA Data for ocf-data-sampler

This script combines EIA data collection and preprocessing into a single workflow.
It collects raw EIA data and immediately preprocesses it for ocf-data-sampler compatibility.

Usage:
    python src/open_data_pvnet/scripts/collect_and_preprocess_eia.py \
        --start 2020-01-01 \
        --end 2023-12-31 \
        --output-dir src/open_data_pvnet/data
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from open_data_pvnet.scripts.fetch_eia_data import EIAData
from open_data_pvnet.scripts.preprocess_eia_for_sampler import preprocess_eia_data
from open_data_pvnet.utils.env_loader import load_environment_variables
import pandas as pd
import xarray as xr
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Collect and preprocess EIA data for ocf-data-sampler"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date YYYY-MM-DD"
    )
    parser.add_argument(
        "--bas",
        nargs="+",
        default=None,
        help="List of BA codes (default: all major ISOs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="src/open_data_pvnet/data",
        help="Output directory for data files"
    )
    parser.add_argument(
        "--capacity-method",
        type=str,
        choices=["estimate", "file", "static"],
        default="estimate",
        help="Method for capacity data (default: estimate)"
    )
    parser.add_argument(
        "--capacity-file",
        type=str,
        default=None,
        help="Path to capacity data CSV (if --capacity-method=file)"
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection, only preprocess existing data"
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default=None,
        help="Path for raw EIA data (default: {output_dir}/target_eia_data.zarr)"
    )
    parser.add_argument(
        "--processed-output",
        type=str,
        default=None,
        help="Path for processed data (default: {output_dir}/target_eia_data_processed.zarr)"
    )
    
    # S3 Upload Arguments
    parser.add_argument("--upload-to-s3", action="store_true", help="Upload processed data to S3")
    parser.add_argument("--s3-bucket", default="ocf-open-data-pvnet", help="S3 Bucket name")
    parser.add_argument("--s3-prefix", default="data/us/eia", help="S3 Prefix")
    parser.add_argument("--s3-version", default="latest", help="Data version string")
    parser.add_argument("--dry-run", action="store_true", help="Simulate S3 upload")
    parser.add_argument("--public", action="store_true", help="Make S3 objects public-read")
    
    args = parser.parse_args()
    
    # Set up paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = args.raw_output or str(output_dir / "target_eia_data.zarr")
    processed_path = args.processed_output or str(output_dir / "target_eia_data_processed.zarr")
    metadata_path = str(output_dir / "us_ba_metadata.csv")
    
    # Step 1: Collect raw EIA data
    if not args.skip_collection:
        logger.info("=" * 60)
        logger.info("Step 1: Collecting EIA data")
        logger.info("=" * 60)
        
        try:
            load_environment_variables()
        except Exception as e:
            logger.warning(f"Could not load environment variables: {e}")
        
        # Use default BAs if not specified
        if args.bas is None:
            DEFAULT_BAS = ['CISO', 'ERCO', 'PJM', 'MISO', 'NYIS', 'ISNE', 'SWPP']
            bas = DEFAULT_BAS
        else:
            bas = args.bas
        
        eia = EIAData()
        if not eia.api_key:
            logger.error("EIA_API_KEY not set. Exiting.")
            return 1
        
        logger.info(f"Fetching data from {args.start} to {args.end} for BAs: {bas}")
        
        df = eia.get_hourly_solar_data(
            start_date=args.start,
            end_date=args.end,
            ba_codes=bas
        )
        
        if df.empty:
            logger.error("No data fetched.")
            return 1
        
        logger.info(f"Fetched {len(df)} rows.")
        
        # BA Centroids (Approximate)
        ba_centroids = {
            'CISO': {'latitude': 37.0, 'longitude': -120.0},
            'ERCO': {'latitude': 31.0, 'longitude': -99.0},
            'PJM': {'latitude': 40.0, 'longitude': -77.0},
            'MISO': {'latitude': 40.0, 'longitude': -90.0},
            'NYIS': {'latitude': 43.0, 'longitude': -75.0},
            'ISNE': {'latitude': 44.0, 'longitude': -71.0},
            'SWPP': {'latitude': 38.0, 'longitude': -98.0},
        }
        
        # Add coordinates
        df["latitude"] = df["ba_code"].map(lambda x: ba_centroids.get(x, {}).get('latitude', np.nan))
        df["longitude"] = df["ba_code"].map(lambda x: ba_centroids.get(x, {}).get('longitude', np.nan))
        
        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        # Ensure timestamp is timezone-naive UTC for Zarr compatibility
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert(None)
        
        # Set index
        df = df.set_index(["timestamp", "ba_code"])
        
        # Convert to xarray
        ds = xr.Dataset.from_dataframe(df)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(raw_path)), exist_ok=True)
        
        # Save to Zarr
        ds.to_zarr(raw_path, mode="w", consolidated=True)
        
        logger.info(f"✅ Raw EIA data collected: {raw_path}")
    else:
        logger.info("Skipping data collection (--skip-collection)")
        if not os.path.exists(raw_path):
            logger.error(f"Raw data file not found: {raw_path}")
            return 1
    
    # Step 3: Optional S3 Upload
    if args.upload_to_s3:
        logger.info("=" * 60)
        logger.info("Step 3: Uploading to S3")
        logger.info("=" * 60)
        
        from open_data_pvnet.scripts.upload_eia_to_s3 import upload_directory_to_s3
        
        # Upload processed data
        full_prefix = f"{args.s3_prefix}/{args.s3_version}"
        full_prefix = full_prefix.replace("//", "/") # Safety check
        
        logger.info(f"Uploading processed data to s3://{args.s3_bucket}/{full_prefix}")
        
        success = upload_directory_to_s3(
            local_dir=processed_path,
            bucket=args.s3_bucket,
            prefix=full_prefix,
            dry_run=args.dry_run,
            public=args.public
        )
        
        if success:
             # Also upload metadata
             if os.path.exists(metadata_path):
                 meta_prefix = f"{full_prefix}/{os.path.basename(metadata_path)}"
                 logger.info(f"Uploading metadata to s3://{args.s3_bucket}/{meta_prefix}")
                 from open_data_pvnet.scripts.upload_eia_to_s3 import upload_file, get_s3_client
                 s3_client = get_s3_client(args.dry_run)
                 upload_file(s3_client, metadata_path, args.s3_bucket, meta_prefix, args.dry_run, args.public)

             logger.info("✅ S3 upload completed")
        else:
            logger.error("❌ S3 upload failed")
            # Don't fail the whole script if upload fails, but warn user
    
    # Summary
    logger.info("=" * 60)
    logger.info("✅ Collection and preprocessing complete!")
    logger.info("=" * 60)
    logger.info(f"Raw data: {raw_path}")
    logger.info(f"Processed data: {processed_path}")
    logger.info(f"Metadata: {metadata_path}")
    if args.upload_to_s3:
         logger.info(f"S3 Target: s3://{args.s3_bucket}/{args.s3_prefix}/{args.s3_version}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Test compatibility: python src/open_data_pvnet/scripts/test_eia_sampler_compatibility.py \\")
    logger.info(f"   --data-path {processed_path}")
    logger.info("2. Update configuration files to use processed data")
    logger.info("3. Proceed with PVNet training setup")
    
    return 0


if __name__ == "__main__":
    exit(main())

