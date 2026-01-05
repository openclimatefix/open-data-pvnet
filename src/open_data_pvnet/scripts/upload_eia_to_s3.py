import logging
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from datetime import datetime
import yaml

# Add parent directory to path to import modules if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from open_data_pvnet.utils.env_loader import load_environment_variables

logger = logging.getLogger(__name__)

def load_s3_config(config_path: str = None) -> dict:
    """Load S3 configuration from yaml file."""
    if config_path is None:
        # Default to standard location
        config_path = str(Path(__file__).parent.parent / "configs" / "eia_s3_config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f).get("s3", {})
    return {}

def get_s3_client(dry_run: bool = False):
    """Get authenticated S3 client."""
    if dry_run:
        return None
    
    try:
        # Check for credentials
        session = boto3.Session()
        credentials = session.get_credentials()
        if not credentials:
            logger.error("No AWS credentials found. Please configure them via 'aws configure' or env vars.")
            return None
        return boto3.client("s3")
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return None

def check_bucket_access(s3_client, bucket: str) -> bool:
    """Check if bucket exists and is accessible."""
    if s3_client is None: return True # Dry run assumes access
    
    try:
        s3_client.head_bucket(Bucket=bucket)
        return True
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            logger.error(f"Bucket '{bucket}' does not exist.")
        elif error_code == 403:
            logger.error(f"Access denied to bucket '{bucket}'. check permissions.")
        else:
            logger.error(f"Error accessing bucket '{bucket}': {e}")
        return False

def upload_file(
    s3_client, 
    local_path: str, 
    bucket: str, 
    s3_key: str, 
    dry_run: bool = False,
    public: bool = False
) -> bool:
    """Upload a single file to S3."""
    if dry_run:
        logger.info(f"[DRY RUN] Would upload {local_path} to s3://{bucket}/{s3_key}")
        return True
    
    try:
        extra_args = {}
        if public:
            extra_args['ACL'] = 'public-read'
            
        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        s3_client.upload_file(local_path, bucket, s3_key, ExtraArgs=extra_args)
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path}: {e}")
        return False

import posixpath

def upload_directory_to_s3(
    local_dir: str,
    bucket: str,
    prefix: str,
    dry_run: bool = False,
    public: bool = False
) -> bool:
    """Upload a directory (e.g. Zarr store) to S3 recursively."""
    s3_client = get_s3_client(dry_run)
    if not dry_run and not s3_client:
        return False
        
    if not check_bucket_access(s3_client, bucket):
        return False

    local_path = Path(local_dir)
    if not local_path.exists():
        logger.error(f"Local path {local_dir} does not exist.")
        return False

    # Normalize prefix: remove leading/trailing slashes
    prefix = prefix.strip("/")
    failed_uploads = []
    
    # If it's a file
    if local_path.is_file():
        s3_key = posixpath.join(prefix, local_path.name)
        if not upload_file(s3_client, str(local_path), bucket, s3_key, dry_run, public):
            failed_uploads.append(str(local_path))
            
    # If it's a directory (Zarr)
    elif local_path.is_dir():
        for root, _, files in os.walk(local_path):
            for file in files:
                full_path = Path(root) / file
                relative_path = full_path.relative_to(local_path)
                
                # Ensure forward slashes for S3 key
                relative_path_str = str(relative_path).replace(os.sep, "/")
                s3_key = posixpath.join(prefix, local_path.name, relative_path_str)
                
                if not upload_file(s3_client, str(full_path), bucket, s3_key, dry_run, public):
                    failed_uploads.append(str(full_path))
    
    if failed_uploads:
        logger.error(f"❌ Failed to upload {len(failed_uploads)} files:")
        for f in failed_uploads[:10]: # Log first 10
            logger.error(f"  - {f}")
        if len(failed_uploads) > 10:
            logger.error(f"  ... and {len(failed_uploads) - 10} more.")
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Upload EIA data to S3")
    parser.add_argument("--input", required=True, help="Path to file or directory to upload (e.g. Zarr store)")
    parser.add_argument("--bucket", default="ocf-open-data-pvnet", help="S3 Bucket name")
    parser.add_argument("--prefix", default="data/us/eia", help="Base S3 prefix")
    parser.add_argument("--version", default="latest", help="Version string (e.g. v1, 2024-01-01)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate upload without actual transfer")
    parser.add_argument("--public", action="store_true", help="Set ACL to public-read")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Construct full prefix
    full_prefix = posixpath.join(args.prefix, args.version)
    
    logger.info("="*60)
    logger.info(f"S3 Upload Utility")
    logger.info(f"Input: {args.input}")
    logger.info(f"Target: s3://{args.bucket}/{full_prefix}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info(f"Public Access: {args.public}")
    logger.info("="*60)

    try:
        load_environment_variables()
    except Exception:
        pass

    if upload_directory_to_s3(
        local_dir=args.input,
        bucket=args.bucket,
        prefix=full_prefix,
        dry_run=args.dry_run,
        public=args.public
    ):
        logger.info("✅ Upload completed successfully.")
        return 0
    else:
        logger.error("❌ Upload failed.")
        return 1

if __name__ == "__main__":
    exit(main())
