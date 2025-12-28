import pandas as pd
import logging
from datetime import datetime
from open_data_pvnet.scripts.fetch_eia_data import EIAData
from open_data_pvnet.utils.env_loader import load_environment_variables
import xarray as xr
import numpy as np
import os
import argparse

logger = logging.getLogger(__name__)

# Major US ISOs/RTOs
DEFAULT_BAS = [
    'CISO', # CAISO
    'ERCO', # ERCOT
    'PJM',  # PJM
    'MISO', # MISO
    'NYIS', # NYISO
    'ISNE', # ISO-NE
    'SWPP', # SPP
]

def main():
    try:
        load_environment_variables()
    except Exception as e:
        logger.warning(f"Could not load environment variables: {e}")

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Collect EIA Solar Data")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument("--bas", nargs="+", default=DEFAULT_BAS, help="List of BA codes")
    parser.add_argument("--output", type=str, default="src/open_data_pvnet/data/target_eia_data.nc", help="Output path")
    
    args = parser.parse_args()
    
    eia = EIAData()
    if not eia.api_key:
        logger.error("EIA_API_KEY not set. Exiting.")
        return

    logger.info(f"Fetching data from {args.start} to {args.end} for BAs: {args.bas}")
    
    try:
        df = eia.get_hourly_solar_data(
            start_date=args.start,
            end_date=args.end,
            ba_codes=args.bas
        )
        
        if df.empty:
            logger.warning("No data fetched.")
            return

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
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Set index
        df = df.set_index(["timestamp", "ba_code"])
        
        # Convert to xarray
        ds = xr.Dataset.from_dataframe(df)
        
        # Ensure output directory exists
        output_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if output_path.endswith(".zarr"):
            ds.to_zarr(output_path, mode="w")
        else:
            # For NetCDF, handling MultiIndex might be tricky or supported depending on xarray version
            # Resetting index might be safer for basic NetCDF viewers, but pvnet might expect dims
            ds.to_netcdf(output_path)
            
        logger.info(f"Data successfully stored in {output_path}")

    except Exception as e:
        logger.error(f"Failed to collect data: {e}")
        raise

if __name__ == "__main__":
    main()
