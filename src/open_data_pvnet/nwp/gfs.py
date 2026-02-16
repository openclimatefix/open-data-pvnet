"""
GFS NWP data processing for open-data-pvnet.

Downloads NOAA GFS forecast data using Herbie (byte-range downloads)
and converts to OCF-compatible Zarr format for PVNet training.

Supports region-specific processing (India, UK, etc.) with configurable
bounding boxes and channel selection.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def process_gfs_data(
    year: int,
    month: int,
    region: str = "india",
    output_dir: str | None = None,
    max_days: int | None = None,
) -> str:
    """
    Download and process GFS NWP data for a specific region and time period.

    Uses Herbie for efficient byte-range downloads from NOAA S3,
    extracting only the 14 OCF channels needed for PVNet.

    Args:
        year: Year to process
        month: Month to process (1-12)
        region: Target region ("india" or "uk")
        output_dir: Output directory for Zarr files. Defaults to data/gfs_{region}/
        max_days: Limit number of days (for testing)

    Returns:
        Path to the output Zarr file.

    Raises:
        ValueError: If region is not supported.
        RuntimeError: If no data could be processed.
    """
    if region not in ("india", "uk"):
        raise ValueError(f"Unsupported region: {region}. Use 'india' or 'uk'.")

    if output_dir is None:
        output_dir = f"data/gfs_{region}"

    # Import here to avoid requiring herbie as a top-level dependency
    from open_data_pvnet.scripts.download_gfs_india import process_month

    logger.info(f"Processing GFS data for {region}: {year}-{month:02d}")

    zarr_path = process_month(
        year=year,
        month=month,
        output_dir=output_dir,
        max_days=max_days,
    )

    if zarr_path is None:
        raise RuntimeError(
            f"No GFS data processed for {region} {year}-{month:02d}. "
            "Check network connectivity and NOAA S3 availability."
        )

    logger.info(f"GFS data saved to {zarr_path}")
    return zarr_path
