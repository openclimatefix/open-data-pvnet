"""
India Data Processor v3 - Handle both SCADA-coded and readable column formats

- Jan 2024-Jun 2025: Readable columns (Timestamp, Demand (MW), Solar (MW))  
- Sep 2021-Dec 2023: SCADA codes that need mapping
"""

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(r"C:\Users\asus vivoBook\Desktop\New folder (2)\pvnet-india-data")
RAW_DIR = BASE_DIR / "raw" / "Electricity Demand, Solar and Wind Generation Data" / "Electricity Demand, Solar and Wind Generation Data"
PROCESSED_DIR = BASE_DIR / "processed"


def load_2024_2025_file(file_path: Path) -> pd.DataFrame:
    """Load the Jan 2024 - Jun 2025 file with readable columns."""
    logger.info(f"Loading 2024-2025 data: {file_path.name}")
    
    xl = pd.ExcelFile(file_path)
    df = xl.parse('Report', header=0)
    
    # Column mapping
    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if 'timestamp' in col_lower or 'time' in col_lower:
            col_map[col] = 'datetime'
        elif 'solar' in col_lower:
            col_map[col] = 'solar_generation_mw'
        elif 'wind' in col_lower:
            col_map[col] = 'wind_generation_mw'
        elif 'demand' in col_lower:
            col_map[col] = 'demand_mw'
    
    df = df.rename(columns=col_map)
    
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
    
    # Convert to numeric
    for col in ['solar_generation_mw', 'wind_generation_mw', 'demand_mw']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"  Loaded {len(df)} rows from 2024-2025 data")
    
    # Show columns found
    found = [c for c in ['solar_generation_mw', 'wind_generation_mw', 'demand_mw'] if c in df.columns]
    logger.info(f"  Columns: {found}")
    
    return df


def load_scada_file(file_path: Path, solar_col_idx: int = None, wind_col_idx: int = None) -> pd.DataFrame:
    """Load old SCADA-coded file with known column indices."""
    logger.info(f"Loading SCADA file: {file_path.name}")
    
    try:
        xl = pd.ExcelFile(file_path)
        if 'Sheet1' not in xl.sheet_names:
            logger.warning(f"  No Sheet1 in {file_path.name}")
            return pd.DataFrame()
        
        df = xl.parse('Sheet1', header=0)
        
        # First column is Time
        time_col = df.columns[0]
        df['datetime'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=['datetime'])
        
        # Use known column indices if provided
        result = {'datetime': df['datetime']}
        
        if solar_col_idx is not None and solar_col_idx < len(df.columns):
            result['solar_generation_mw'] = pd.to_numeric(df.iloc[:, solar_col_idx], errors='coerce')
        
        if wind_col_idx is not None and wind_col_idx < len(df.columns):
            result['wind_generation_mw'] = pd.to_numeric(df.iloc[:, wind_col_idx], errors='coerce')
        
        result_df = pd.DataFrame(result)
        logger.info(f"  Loaded {len(result_df)} rows")
        return result_df
        
    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {e}")
        return pd.DataFrame()


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Start with the 2024-2025 file which has clear columns
    file_2024_2025 = RAW_DIR / "January 2024- June 2025.xlsx"
    
    all_data = []
    
    if file_2024_2025.exists():
        df = load_2024_2025_file(file_2024_2025)
        if not df.empty and 'solar_generation_mw' in df.columns:
            all_data.append(df)
    
    # For now, skip the SCADA-coded files until we find the column mapping
    # TODO: Add SCADA column index mapping once identified
    
    logger.info(f"\nSuccessfully loaded {len(all_data)} files with solar data")
    
    if not all_data:
        logger.error("No valid data found!")
        return
    
    # Combine
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values('datetime').drop_duplicates(subset=['datetime'])
    
    logger.info(f"\nCombined dataset: {len(combined)} rows")
    logger.info(f"Date range: {combined['datetime'].min()} to {combined['datetime'].max()}")
    
    # Already hourly from the 2024-2025 file
    hourly = combined.copy()
    hourly = hourly.rename(columns={'datetime': 'datetime_gmt'})
    
    logger.info(f"Hourly dataset: {len(hourly)} rows")
    
    # Add region_id for PVNet compatibility (0 = All India)
    hourly['region_id'] = 0
    
    # Select only needed columns
    columns = ['region_id', 'datetime_gmt', 'solar_generation_mw']
    if 'wind_generation_mw' in hourly.columns:
        columns.append('wind_generation_mw')
    if 'demand_mw' in hourly.columns:
        columns.append('demand_mw')
    hourly = hourly[columns]
    
    # Save CSV
    csv_path = PROCESSED_DIR / "india_solar_hourly.csv"
    hourly.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")
    
    # Save Zarr
    hourly_indexed = hourly.set_index(['region_id', 'datetime_gmt'])
    ds = xr.Dataset.from_dataframe(hourly_indexed)
    ds.attrs = {
        'description': 'India All-India solar generation from Grid-India (POSOCO)',
        'source': 'Mendeley DOI: 10.17632/y58jknpgs8.2',
        'time_resolution': '1 hour',
        'date_range': f"{combined['datetime'].min()} to {combined['datetime'].max()}",
        'created': datetime.now().isoformat()
    }
    # Skip chunking for simpler export without dask
    
    zarr_path = PROCESSED_DIR / "india_solar_2024-2025.zarr"
    ds.to_zarr(str(zarr_path), mode='w', consolidated=True)
    logger.info(f"Saved Zarr: {zarr_path}")
    
    # Print summary stats
    logger.info("\n" + "="*60)
    logger.info("DATA SUMMARY")
    logger.info("="*60)
    for col in ['solar_generation_mw', 'wind_generation_mw', 'demand_mw']:
        if col in hourly.columns:
            data = hourly[col].dropna()
            logger.info(f"\n{col}:")
            logger.info(f"  Count: {len(data)}")
            logger.info(f"  Min: {data.min():.2f} MW")
            logger.info(f"  Max: {data.max():.2f} MW")
            logger.info(f"  Mean: {data.mean():.2f} MW")
            logger.info(f"  Missing: {hourly[col].isna().sum()} ({hourly[col].isna().mean()*100:.1f}%)")
    
    logger.info("\n✅ Processing complete!")
    logger.info("\nNOTE: Only Jan 2024-Jun 2025 data processed. SCADA-coded files (2021-2023) need column mapping.")


if __name__ == "__main__":
    main()
