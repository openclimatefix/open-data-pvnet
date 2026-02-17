"""Generate training samples from PV and GFS data."""

import argparse
import logging
import sys
from pathlib import Path

import xarray as xr
import yaml

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_samples(pv_ds: xr.Dataset, gfs_ds: xr.Dataset, num_samples: int):
    """Generate training samples from datasets."""
    logger.info(f"Generating {num_samples} samples")
    
    samples = []
    pv_time_dim = 'datetime_gmt' if 'datetime_gmt' in pv_ds.dims else 'time'
    gfs_time_dim = 'init_time_utc' if 'init_time_utc' in gfs_ds.dims else 'time'
    
    pv_steps = len(pv_ds[pv_time_dim])
    gfs_steps = len(gfs_ds[gfs_time_dim])
    
    for i in range(min(num_samples, pv_steps)):
        sample = {
            'pv_idx': i,
            'gfs_idx': i if i < gfs_steps else None
        }
        samples.append(sample)
    
    logger.info(f"Generated {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate training samples from PV and GFS data")
    parser.add_argument('--pv-zarr', type=str, required=True, help='Path to PV Zarr file')
    parser.add_argument('--gfs-zarr', type=str, required=True, help='Path to GFS Zarr file')
    parser.add_argument('--config', type=str,
                       default='src/open_data_pvnet/configs/PVNet_configs/datamodule/configuration/germany_configuration.yaml',
                       help='Path to model configuration')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--output-dir', type=str, default='./data/germany/samples', help='Output directory')
    
    args = parser.parse_args()
    
    try:
        pv_ds = xr.open_zarr(args.pv_zarr)
        gfs_ds = xr.open_zarr(args.gfs_zarr)
        
        logger.info(f"Loaded PV data: {dict(pv_ds.dims)}")
        logger.info(f"Loaded GFS data: {dict(gfs_ds.dims)}")
        
        samples = generate_samples(pv_ds, gfs_ds, args.num_samples)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Sample generation completed: {len(samples)} samples")
        logger.info(f"Output directory: {output_dir}")
        
    except Exception as e:
        logger.error(f"Sample generation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
