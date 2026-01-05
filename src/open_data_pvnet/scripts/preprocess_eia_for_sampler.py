"""
Preprocess EIA Data for ocf-data-sampler

This script converts raw EIA data collected by collect_eia_data.py into the format
expected by ocf-data-sampler, matching the UK GSP data structure.

UK GSP Format:
- Dimensions: (gsp_id, datetime_gmt) where gsp_id is int64
- Variables: generation_mw, capacity_mwp, installedcapacity_mwp
- Chunking: {"gsp_id": 1, "datetime_gmt": 1000}

US EIA Format (input):
- Dimensions: (timestamp, ba_code) where ba_code is string
- Variables: generation_mw, ba_name, latitude, longitude

US EIA Format (output):
- Dimensions: (ba_id, datetime_gmt) where ba_id is int64
- Variables: generation_mw, capacity_mwp
- Coordinates: ba_code, ba_name, latitude, longitude

Usage:
    python src/open_data_pvnet/scripts/preprocess_eia_for_sampler.py \
        --input src/open_data_pvnet/data/target_eia_data.zarr \
        --output src/open_data_pvnet/data/target_eia_data_processed.zarr \
        --metadata-output src/open_data_pvnet/data/us_ba_metadata.csv
"""

import pandas as pd
import xarray as xr
import numpy as np
import logging
import os
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def estimate_capacity_from_generation(
    df: pd.DataFrame, ba_col: str = "ba_code", gen_col: str = "generation_mw"
) -> pd.Series:
    """
    Estimate capacity from maximum historical generation.
    
    This is a simple heuristic: capacity ≈ max(generation) * safety_factor
    The safety factor accounts for the fact that max generation is typically
    less than installed capacity (due to weather, maintenance, etc.)
    
    Args:
        df: DataFrame with generation data
        ba_col: Column name for BA identifier
        gen_col: Column name for generation values
        
    Returns:
        Series with capacity estimates indexed by BA code
    """
    # Group by BA and find max generation
    max_gen = df.groupby(ba_col)[gen_col].max()
    
    # Apply safety factor (typically max generation is 70-90% of capacity)
    # Using 0.85 as a reasonable estimate
    safety_factor = 1.15  # 1/0.85 ≈ 1.15 to get capacity from max gen
    capacity = max_gen * safety_factor
    
    # Ensure minimum capacity (at least 100 MW for major BAs)
    capacity = capacity.clip(lower=100.0)
    
    logger.info(f"Estimated capacity for {len(capacity)} BAs")
    logger.debug(f"Capacity range: {capacity.min():.2f} - {capacity.max():.2f} MW")
    
    return capacity


def create_ba_mapping(df: pd.DataFrame, ba_col: str = "ba_code") -> tuple[dict, pd.DataFrame]:
    """
    Create mapping from BA codes to numeric IDs.
    
    Args:
        df: DataFrame with BA codes
        ba_col: Column name for BA codes
        
    Returns:
        Tuple of (ba_code_to_id dict, metadata DataFrame)
    """
    unique_bas = sorted(df[ba_col].unique())
    ba_to_id = {ba: idx for idx, ba in enumerate(unique_bas)}
    
    # Create metadata DataFrame
    metadata = pd.DataFrame({
        "ba_id": list(ba_to_id.values()),
        "ba_code": list(ba_to_id.keys()),
    })
    
    # Add coordinates if available
    if "latitude" in df.columns and "longitude" in df.columns:
        coords = df.groupby(ba_col)[["latitude", "longitude"]].first()
        metadata = metadata.merge(
            coords.reset_index(),
            on="ba_code",
            how="left"
        )
    
    # Add BA names if available
    if "ba_name" in df.columns:
        names = df.groupby(ba_col)["ba_name"].first()
        metadata = metadata.merge(
            names.reset_index(),
            on="ba_code",
            how="left"
        )
    
    logger.info(f"Created mapping for {len(ba_to_id)} BAs")
    
    return ba_to_id, metadata


def preprocess_eia_data(
    input_path: str,
    output_path: str,
    metadata_output_path: str = None,
    capacity_method: str = "estimate",
    capacity_file: str = None,
) -> str:
    """
    Preprocess EIA data to match ocf-data-sampler format.
    
    Args:
        input_path: Path to input EIA Zarr/NetCDF file
        output_path: Path to output processed Zarr file
        metadata_output_path: Path to save BA metadata CSV
        capacity_method: Method for capacity data ("estimate", "file", "static")
        capacity_file: Path to capacity data file (if method is "file")
        
    Returns:
        Path to output file
    """
    logger.info(f"Loading EIA data from {input_path}")
    
    if input_path.endswith(".zarr"):
        ds = xr.open_dataset(input_path, engine="zarr")
    else:
        ds = xr.open_dataset(input_path)
    
    df = ds.to_dataframe()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        if df.index.name in ["timestamp", "datetime_gmt"]:
            df = df.reset_index()
    
    logger.info(f"Loaded {len(df)} rows")
    
    if "ba_code" not in df.columns:
        if isinstance(ds.indexes.get("ba_code"), pd.Index):
            df = df.reset_index()
        else:
            raise ValueError("ba_code not found in dataset. Check input data format.")
    
    logger.info(f"Loaded {len(df)} rows for {df['ba_code'].nunique()} BAs")
    
    if "timestamp" in df.columns:
        df["datetime_gmt"] = pd.to_datetime(df["timestamp"], utc=True)
        df["datetime_gmt"] = df["datetime_gmt"].dt.tz_convert(None)
        df = df.drop(columns=["timestamp"])
    elif "datetime_gmt" not in df.columns:
        raise ValueError("No timestamp or datetime_gmt column found")
    
    if "generation_mw" in df.columns:
        df["generation_mw"] = pd.to_numeric(df["generation_mw"], errors="coerce")
    else:
        raise ValueError("No generation_mw column found")
    
    ba_to_id, metadata = create_ba_mapping(df)
    df["ba_id"] = df["ba_code"].map(ba_to_id)
    
    if capacity_method == "estimate":
        logger.info("Estimating capacity from maximum generation")
        capacity_estimates = estimate_capacity_from_generation(df)
        df["capacity_mwp"] = df["ba_code"].map(capacity_estimates)
    elif capacity_method == "file" and capacity_file:
        logger.info(f"Loading capacity from {capacity_file}")
        capacity_df = pd.read_csv(capacity_file)
        capacity_map = dict(zip(capacity_df["ba_code"], capacity_df["capacity_mwp"]))
        df["capacity_mwp"] = df["ba_code"].map(capacity_map)
    elif capacity_method == "static":
        logger.warning("Using static capacity values (not recommended)")
        df["capacity_mwp"] = 1000.0
    else:
        raise ValueError(f"Invalid capacity_method: {capacity_method}")
    
    if df["capacity_mwp"].isna().any():
        logger.warning("Some BAs have missing capacity data, filling with estimates")
        missing_bas = df[df["capacity_mwp"].isna()]["ba_code"].unique()
        for ba in missing_bas:
            ba_gen = df[df["ba_code"] == ba]["generation_mw"].max()
            estimated_capacity = ba_gen * 1.15 if not pd.isna(ba_gen) else 100.0
            df.loc[df["ba_code"] == ba, "capacity_mwp"] = estimated_capacity
    
    df["capacity_mwp"] = df[["capacity_mwp", "generation_mw"]].max(axis=1) * 1.01
    
    columns_to_keep = ["ba_id", "datetime_gmt", "generation_mw", "capacity_mwp"]
    if "ba_code" in df.columns:
        columns_to_keep.append("ba_code")
    if "ba_name" in df.columns:
        columns_to_keep.append("ba_name")
    if "latitude" in df.columns:
        columns_to_keep.append("latitude")
    if "longitude" in df.columns:
        columns_to_keep.append("longitude")
    
    df_processed = df[columns_to_keep].copy()
    
    df_processed = df_processed.set_index(["ba_id", "datetime_gmt"])
    
    ds_processed = xr.Dataset.from_dataframe(df_processed)
    
    # Ensure datetime_gmt is datetime64[ns] (no timezone)
    if "datetime_gmt" in ds_processed.coords:
        ds_processed.coords["datetime_gmt"] = ds_processed.coords["datetime_gmt"].astype(np.datetime64)
    
    # Apply chunking like UK implementation
    # We'll use: {"ba_id": 1, "datetime_gmt": 1000}
    try:
        import dask
        ds_processed = ds_processed.chunk({"ba_id": 1, "datetime_gmt": 1000})
    except ImportError:
        logger.warning("Dask not installed, skipping chunking. Performance may be affected.")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to Zarr with consolidated metadata
    logger.info(f"Saving processed data to {output_path}")
    ds_processed.to_zarr(output_path, mode="w", consolidated=True)
    
    # Save metadata CSV
    if metadata_output_path:
        metadata_dir = os.path.dirname(os.path.abspath(metadata_output_path))
        os.makedirs(metadata_dir, exist_ok=True)
        metadata.to_csv(metadata_output_path, index=False)
        logger.info(f"Saved BA metadata to {metadata_output_path}")
    
    logger.info(f"✅ Successfully preprocessed EIA data")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Dimensions: {dict(ds_processed.dims)}")
    logger.info(f"   Variables: {list(ds_processed.data_vars)}")
    
    return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess EIA data for ocf-data-sampler compatibility"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input EIA data file (Zarr or NetCDF)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output processed Zarr file path"
    )
    parser.add_argument(
        "--metadata-output",
        type=str,
        default=None,
        help="Output path for BA metadata CSV (optional)"
    )
    parser.add_argument(
        "--capacity-method",
        type=str,
        choices=["estimate", "file", "static"],
        default="estimate",
        help="Method for obtaining capacity data (default: estimate)"
    )
    parser.add_argument(
        "--capacity-file",
        type=str,
        default=None,
        help="Path to capacity data CSV file (if --capacity-method=file)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        preprocess_eia_data(
            input_path=args.input,
            output_path=args.output,
            metadata_output_path=args.metadata_output,
            capacity_method=args.capacity_method,
            capacity_file=args.capacity_file,
        )
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

