"""Germany solar forecasting pipeline orchestrator."""

import argparse
import logging
import sys
from pathlib import Path

import xarray as xr
import yaml

from germany_utils import (
    validate_pv_data,
    validate_gfs_data,
    calculate_normalization,
    check_temporal_alignment,
    save_normalization,
    print_zarr_summary,
    get_time_coord,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_data(pv_path: str, gfs_path: str):
    """Load PV and GFS datasets."""
    try:
        pv_ds = xr.open_zarr(pv_path)
        gfs_ds = xr.open_zarr(gfs_path)
        logger.info(f"Loaded PV data: {dict(pv_ds.dims)}")
        logger.info(f"Loaded GFS data: {dict(gfs_ds.dims)}")
        return pv_ds, gfs_ds
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None, None


def process_command(args):
    """Process and align data."""
    pv_ds, gfs_ds = load_data(args.pv_zarr, args.gfs_zarr)
    if pv_ds is None or gfs_ds is None:
        sys.exit(1)
    
    if not validate_pv_data(pv_ds) or not validate_gfs_data(gfs_ds):
        logger.error("Data validation failed")
        sys.exit(1)
    
    # Check alignment
    alignment_stats = check_temporal_alignment(pv_ds, gfs_ds)
    logger.info(f"Temporal overlap: {alignment_stats['overlap_days']} days")
    
    # Calculate normalization
    norm_constants = calculate_normalization(gfs_ds)
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_normalization(norm_constants, output_dir / 'normalization_constants.yaml')
    
    # Generate report
    report_lines = [
        "=" * 60,
        "Germany Data Processing Report",
        "=" * 60,
        "",
        "PV Data:",
        f"  Range: {alignment_stats['pv_start']} to {alignment_stats['pv_end']}",
        f"  Steps: {len(pv_ds[get_time_coord(pv_ds, is_pv=True)])}",
        "",
        "GFS Data:",
        f"  Range: {alignment_stats['gfs_start']} to {alignment_stats['gfs_end']}",
        f"  Steps: {len(gfs_ds[get_time_coord(gfs_ds, is_pv=False)])}",
        f"  Variables: {list(gfs_ds.data_vars)}",
        "",
        "Temporal Alignment:",
        f"  Overlap: {alignment_stats['overlap_days']} days",
        f"  From: {alignment_stats['overlap_start']}",
        f"  To: {alignment_stats['overlap_end']}",
        "=" * 60,
    ]
    
    report_text = "\n".join(report_lines)
    with open(output_dir / 'processing_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    logger.info("Processing completed successfully")


def test_command(args):
    """Run pipeline tests."""
    pv_ds, gfs_ds = load_data(args.pv_zarr, args.gfs_zarr)
    if pv_ds is None or gfs_ds is None:
        sys.exit(1)
    
    tests = []
    
    # Test 1: Data loading
    try:
        pv_time = get_time_coord(pv_ds, is_pv=True)
        gfs_time = get_time_coord(gfs_ds, is_pv=False)
        assert len(pv_ds[pv_time]) > 0 and len(gfs_ds[gfs_time]) > 0
        tests.append(("Data Loading", True))
        logger.info("✓ Data loading test passed")
    except Exception as e:
        tests.append(("Data Loading", False))
        logger.error(f"✗ Data loading test failed: {e}")
    
    # Test 2: Data validation
    pv_valid = validate_pv_data(pv_ds)
    gfs_valid = validate_gfs_data(gfs_ds)
    tests.append(("Data Validation", pv_valid and gfs_valid))
    
    # Test 3: Temporal alignment
    try:
        alignment = check_temporal_alignment(pv_ds, gfs_ds)
        assert alignment['overlap_days'] > 0
        tests.append(("Temporal Alignment", True))
        logger.info("✓ Temporal alignment test passed")
    except Exception as e:
        tests.append(("Temporal Alignment", False))
        logger.error(f"✗ Temporal alignment test failed: {e}")
    
    # Generate report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    report_lines = [
        "=" * 60,
        "Germany Pipeline Test Report",
        "=" * 60,
        "",
    ]
    for test_name, result in tests:
        status = "PASS" if result else "FAIL"
        report_lines.append(f"{test_name}: {status}")
    
    report_lines.extend(["", f"Total: {passed}/{total} tests passed", "=" * 60])
    
    report_text = "\n".join(report_lines)
    with open(output_dir / 'test_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    sys.exit(0 if passed == total else 1)


def inspect_command(args):
    """Inspect a Zarr file."""
    print_zarr_summary(args.zarr)


def main():
    parser = argparse.ArgumentParser(
        description="Germany solar forecasting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a Zarr file
  python germany_pipeline.py inspect --zarr data/germany/gfs/zarr/germany_gfs_2021_01.zarr
  
  # Process data
  python germany_pipeline.py process --pv-zarr data/germany/generation/germany_pv_2021.zarr \\
                                      --gfs-zarr data/germany/gfs/zarr/germany_gfs_2021_01.zarr
  
  # Run tests
  python germany_pipeline.py test --pv-zarr data/germany/generation/germany_pv_2021.zarr \\
                                   --gfs-zarr data/germany/gfs/zarr/germany_gfs_2021_01.zarr
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect a Zarr file')
    inspect_parser.add_argument('--zarr', type=str, required=True, help='Path to Zarr file')
    inspect_parser.set_defaults(func=inspect_command)
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process and align data')
    process_parser.add_argument('--pv-zarr', type=str, required=True, help='Path to PV Zarr file')
    process_parser.add_argument('--gfs-zarr', type=str, required=True, help='Path to GFS Zarr file')
    process_parser.add_argument('--output-dir', type=str, default='./data/germany/processed')
    process_parser.set_defaults(func=process_command)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run pipeline tests')
    test_parser.add_argument('--pv-zarr', type=str, required=True, help='Path to PV Zarr file')
    test_parser.add_argument('--gfs-zarr', type=str, required=True, help='Path to GFS Zarr file')
    test_parser.add_argument('--output-dir', type=str, default='./data/germany/tests')
    test_parser.set_defaults(func=test_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
