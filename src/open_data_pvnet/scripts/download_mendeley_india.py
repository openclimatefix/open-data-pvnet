"""
India PVNet Data Download Script

This script provides utilities for downloading and processing India solar generation data
from various sources for PVNet training.

Data Sources:
1. Mendeley Dataset (DOI: 10.17632/y58jknpgs8.2)
   - Hourly data from Grid-India NERLDC
   - Sep 2021 - Dec 2023
   - 5 regional grids (NR, WR, SR, ER, NER)

2. Kaggle Solar Power Generation (backup)
   - 15-min data from 2 Indian plants
   - 34 days of data
   - https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

Usage:
    # After manual download from Mendeley:
    python download_mendeley_india.py --process --input-dir raw_data/

    # For Kaggle data (requires kaggle API key):
    python download_mendeley_india.py --kaggle
"""

import os
import sys
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("c:/Users/asus vivoBook/Desktop/New folder (2)/pvnet-india-data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Mendeley dataset info
MENDELEY_DOI = "10.17632/y58jknpgs8.2"
MENDELEY_URL = f"https://data.mendeley.com/datasets/y58jknpgs8/2"

# India regional grid metadata
INDIA_REGIONS = {
    "NR": {"name": "Northern Region", "lat": 28.6139, "lon": 77.2090},
    "WR": {"name": "Western Region", "lat": 19.0760, "lon": 72.8777},
    "SR": {"name": "Southern Region", "lat": 13.0827, "lon": 80.2707},
    "ER": {"name": "Eastern Region", "lat": 22.5726, "lon": 88.3639},
    "NER": {"name": "North-Eastern Region", "lat": 26.1445, "lon": 91.7362},
}


def print_download_instructions():
    """Print manual download instructions for Mendeley dataset."""
    instructions = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MENDELEY INDIA DATASET DOWNLOAD GUIDE                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  The Mendeley API has known issues. Please download manually:               ║
║                                                                              ║
║  1. Open browser and go to:                                                  ║
║     {MENDELEY_URL:<55}║
║                                                                              ║
║  2. Click "Download all files" button (usually a ZIP archive)               ║
║                                                                              ║
║  3. Save to:                                                                 ║
║     {str(RAW_DIR):<55}║
║                                                                              ║
║  4. Extract the ZIP file in the same directory                              ║
║                                                                              ║
║  5. Run this script again with --process flag:                              ║
║     python download_mendeley_india.py --process                             ║
║                                                                              ║
║  Dataset DOI: {MENDELEY_DOI:<50}║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


def download_kaggle_dataset():
    """Download Kaggle solar power generation dataset as a backup."""
    try:
        import kaggle
        kaggle.api.authenticate()
        
        kaggle_dir = RAW_DIR / "kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Downloading Kaggle solar power generation dataset...")
        kaggle.api.dataset_download_files(
            "anikannal/solar-power-generation-data",
            path=str(kaggle_dir),
            unzip=True
        )
        logger.info(f"Dataset downloaded to {kaggle_dir}")
        return True
    except ImportError:
        logger.error("Kaggle package not installed. Run: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        logger.info("Ensure ~/.kaggle/kaggle.json has valid API credentials")
        return False


def find_mendeley_files(input_dir: Path) -> list:
    """Find Mendeley data files in the input directory."""
    extensions = ['.xlsx', '.xls', '.csv', '.parquet']
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f'*{ext}'))
        files.extend(input_dir.glob(f'**/*{ext}'))
    return files


def load_mendeley_data(file_path: Path) -> pd.DataFrame:
    """Load Mendeley data file into pandas DataFrame."""
    logger.info(f"Loading: {file_path}")
    
    if file_path.suffix in ['.xlsx', '.xls']:
        # Try reading Excel file
        df = pd.read_excel(file_path)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def process_mendeley_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process Mendeley data to standardize format for PVNet.
    
    Expected columns after processing:
    - datetime_gmt: Timestamp in UTC
    - region_id: Integer ID for each region (0-4)
    - generation_mw: Solar generation in MW
    - capacity_mw: Installed capacity in MW (if available)
    """
    logger.info("Processing Mendeley data...")
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Log available columns for inspection
    logger.info(f"Available columns: {list(df.columns)}")
    
    # TODO: Actual column mapping will depend on the downloaded file structure
    # This is a placeholder that will be updated after inspecting the actual data
    
    return df


def convert_to_zarr(df: pd.DataFrame, output_path: Path):
    """Convert processed DataFrame to Zarr format for PVNet."""
    logger.info(f"Converting to Zarr: {output_path}")
    
    # Convert to xarray Dataset
    ds = xr.Dataset.from_dataframe(df.set_index(['region_id', 'datetime_gmt']))
    
    # Chunk appropriately
    ds = ds.chunk({'region_id': 1, 'datetime_gmt': 1000})
    
    # Save to Zarr
    ds.to_zarr(str(output_path), mode='w', consolidated=True)
    logger.info(f"Saved Zarr dataset to {output_path}")


def validate_data(df: pd.DataFrame):
    """Validate data quality and print summary statistics."""
    logger.info("\n" + "="*60)
    logger.info("DATA VALIDATION REPORT")
    logger.info("="*60)
    
    # Basic stats
    logger.info(f"\nShape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"\nData types:\n{df.dtypes}")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"\nMissing values:\n{missing[missing > 0]}")
    else:
        logger.info("\n✓ No missing values")
    
    # Date range
    date_cols = df.select_dtypes(include=['datetime64']).columns
    for col in date_cols:
        logger.info(f"\n{col} range: {df[col].min()} to {df[col].max()}")
    
    # Numeric summaries
    logger.info(f"\nNumeric summary:\n{df.describe()}")


def main():
    parser = argparse.ArgumentParser(description="India PVNet Data Download & Processing")
    parser.add_argument('--process', action='store_true', help='Process downloaded Mendeley data')
    parser.add_argument('--kaggle', action='store_true', help='Download Kaggle backup dataset')
    parser.add_argument('--input-dir', type=str, default=str(RAW_DIR), help='Input directory with raw data')
    parser.add_argument('--validate', action='store_true', help='Validate existing data')
    
    args = parser.parse_args()
    
    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.kaggle:
        download_kaggle_dataset()
        return
    
    if args.process:
        input_dir = Path(args.input_dir)
        files = find_mendeley_files(input_dir)
        
        if not files:
            logger.error(f"No data files found in {input_dir}")
            print_download_instructions()
            return
        
        logger.info(f"Found {len(files)} data file(s)")
        
        all_data = []
        for f in files:
            try:
                df = load_mendeley_data(f)
                all_data.append(df)
            except Exception as e:
                logger.error(f"Failed to load {f}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            validate_data(combined_df)
            
            # Save intermediate CSV
            combined_df.to_csv(PROCESSED_DIR / "india_solar_combined.csv", index=False)
            logger.info(f"Saved combined data to {PROCESSED_DIR / 'india_solar_combined.csv'}")
    else:
        # Default: print download instructions
        print_download_instructions()


if __name__ == "__main__":
    main()
