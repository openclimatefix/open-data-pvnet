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
    
    # Rename columns to ocf-data-sampler expected format
    # See: https://github.com/openclimatefix/ocf-data-sampler/blob/main/ocf_data_sampler/load/generation.py
    hourly = combined.copy()
    hourly = hourly.rename(columns={
        'datetime': 'time_utc',
        'solar_generation_mw': 'generation_mw'
    })
    
    # Use location_id instead of region_id (ocf-data-sampler expects location_id)
    hourly['location_id'] = 0  # 0 = All India aggregate
    
    # Add required coordinates for ocf-data-sampler
    # India center approx: 20°N, 78°E (used for GFS NWP extraction)
    hourly['longitude'] = 78.0  # India center longitude
    hourly['latitude'] = 20.0   # India center latitude
    
    # Calculate capacity_mwp from peak observed values
    # India solar installed capacity ~70GW as of 2024
    # Use 95th percentile of observed values as proxy for capacity
    solar_capacity_mwp = hourly['generation_mw'].quantile(0.95)
    hourly['capacity_mwp'] = solar_capacity_mwp
    
    logger.info(f"Hourly dataset: {len(hourly)} rows")
    logger.info(f"  Estimated capacity: {solar_capacity_mwp:.0f} MWp")
    
    # Save CSV with original column names for reference
    csv_data = hourly.copy()
    csv_path = PROCESSED_DIR / "india_solar_hourly.csv"
    csv_data.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")
    
    # Create xarray Dataset with ocf-data-sampler schema
    # Dimensions: (time_utc, location_id)
    # Data Variables: generation_mw, capacity_mwp
    # Coordinates: time_utc, location_id, longitude, latitude
    
    time_utc = pd.to_datetime(hourly['time_utc']).values
    location_ids = hourly['location_id'].unique()
    
    # Create DataArrays
    generation_mw = xr.DataArray(
        data=hourly['generation_mw'].values.reshape(1, -1),  # (location, time)
        dims=['location_id', 'time_utc'],
        coords={
            'location_id': ('location_id', location_ids),
            'time_utc': ('time_utc', time_utc),
            'longitude': ('location_id', [78.0]),
            'latitude': ('location_id', [20.0]),
        }
    )
    
    capacity_mwp = xr.DataArray(
        data=np.full((1, len(time_utc)), solar_capacity_mwp),  # Same capacity for all times
        dims=['location_id', 'time_utc'],
        coords={
            'location_id': ('location_id', location_ids),
            'time_utc': ('time_utc', time_utc),
            'longitude': ('location_id', [78.0]),
            'latitude': ('location_id', [20.0]),
        }
    )
    
    ds = xr.Dataset({
        'generation_mw': generation_mw,
        'capacity_mwp': capacity_mwp,
    })
    
    ds.attrs = {
        'description': 'India All-India solar generation for PVNet training',
        'source': 'Mendeley DOI: 10.17632/y58jknpgs8.2 (Grid-India/POSOCO)',
        'schema': 'ocf-data-sampler generation format',
        'time_resolution': '1 hour',
        'location': 'All India aggregate',
        'date_range': f"{hourly['time_utc'].min()} to {hourly['time_utc'].max()}",
        'created': datetime.now().isoformat()
    }
    
    zarr_path = PROCESSED_DIR / "india_solar_2024-2025.zarr"
    ds.to_zarr(str(zarr_path), mode='w', consolidated=True)
    logger.info(f"Saved Zarr: {zarr_path}")
    
    # Verify the schema
    logger.info("\n=== Zarr Schema Verification ===")
    logger.info(f"Dimensions: {dict(ds.dims)}")
    logger.info("Coordinates:")
    for coord in ds.coords:
        logger.info(f"  {coord}: {ds.coords[coord].dtype}")
    logger.info("Data Variables:")
    for var in ds.data_vars:
        logger.info(f"  {var}: {ds.data_vars[var].dtype}, dims {ds.data_vars[var].dims}")
    
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
