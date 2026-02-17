import os
import logging
import argparse
import pandas as pd
import xarray as xr
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path

from open_data_pvnet.scripts.fetch_eia_data import EIAData

logger = logging.getLogger(__name__)

# US RTO/Region location metadata
US_RTO_LOCATIONS = {
    "CAISO": {"location_id": 1, "latitude": 36.7783, "longitude": -119.4179, "name": "California ISO"},
    "ERCOT": {"location_id": 2, "latitude": 31.9686, "longitude": -99.9018, "name": "Texas"},
    "PJM": {"location_id": 3, "latitude": 40.0583, "longitude": -76.3055, "name": "Mid-Atlantic"},
    "MISO": {"location_id": 4, "latitude": 41.8781, "longitude": -87.6298, "name": "Midwest ISO"},
    "NYISO": {"location_id": 5, "latitude": 42.6526, "longitude": -73.7562, "name": "New York ISO"},
    "ISONE": {"location_id": 6, "latitude": 42.3601, "longitude": -71.0589, "name": "New England ISO"},
    "SPP": {"location_id": 7, "latitude": 38.5767, "longitude": -92.1735, "name": "Southwest Power Pool"},
    "US48": {"location_id": 0, "latitude": 39.8283, "longitude": -98.5795, "name": "Continental US"},
}


class EIAPreprocessor:
    """Preprocessor to convert EIA solar data to ocf-data-sampler format."""
    def __init__(self, api_key: Optional[str] = None):
        """Initialize preprocessor with optional API key."""
        self.eia_data = EIAData(api_key=api_key)
        self.location_metadata = US_RTO_LOCATIONS
        
    def fetch_and_preprocess(
        self,
        start_date: str,
        end_date: str,
        regions: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        route: str = "electricity/rto/daily-fuel-type-data",
        frequency: str = "hourly",
    ) -> xr.Dataset:
        """Fetch EIA data and preprocess it. Defaults to US48 if no regions specified."""
        logger.info(f"Starting preprocessing for {start_date} to {end_date}")
        
        if regions is None:
            regions = ["US48"]
            
        all_datasets = []
        
        for region in regions:
            logger.info(f"Processing region: {region}")
            
            facets = {"fueltype": "SUN"}
            if region != "US48":
                facets["respondent"] = [region]
                
            df = self.eia_data.get_data(
                route=route,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency,
                data_cols=["value"],
                facets=facets,
                region=region if region == "US48" else None,
            )
            
            if df is None or df.empty:
                logger.warning(f"No data retrieved for region {region}")
                continue
                
            ds = self.transform_to_schema(df, region)
            all_datasets.append(ds)
            
        if not all_datasets:
            raise ValueError("No data retrieved for any region")
            
        combined_ds = xr.concat(all_datasets, dim="location_id")
        combined_ds = self.estimate_capacity(combined_ds)
        
        if not self.validate_data(combined_ds):
            raise ValueError("Data validation failed")
            
        logger.info("Preprocessing completed successfully")
        
        if output_path:
            self.save_to_zarr(combined_ds, output_path)
            
        return combined_ds
        
    def transform_to_schema(self, df: pd.DataFrame, region: str) -> xr.Dataset:
        """Transform raw EIA DataFrame to the required schema."""
        if region not in self.location_metadata:
            raise ValueError(f"Unknown region: {region}. Available: {list(self.location_metadata.keys())}")
            
        location_info = self.location_metadata[region]
        location_id = location_info["location_id"]
        
        if "period" in df.columns:
            df["time_utc"] = pd.to_datetime(df["period"], utc=True)
        elif "datetime_gmt" in df.columns:
            df["time_utc"] = pd.to_datetime(df["datetime_gmt"], utc=True)
        else:
            raise ValueError("No time column found in data")
            
        if "value" in df.columns:
            df["generation_mw"] = df["value"].astype(np.float32)
        else:
            raise ValueError("No 'value' column found in data")
            
        df_clean = df[["time_utc", "generation_mw"]].copy()
        df_clean = df_clean.drop_duplicates(subset=["time_utc"])
        df_clean = df_clean.set_index("time_utc")
        df_clean = df_clean.sort_index()
        
        ds = xr.Dataset.from_dataframe(df_clean)
        ds = ds.expand_dims({"location_id": [location_id]})
        ds = ds.assign_coords({
            "longitude": ("location_id", [location_info["longitude"]]),
            "latitude": ("location_id", [location_info["latitude"]]),
        })
        
        return ds
        
    def estimate_capacity(self, ds: xr.Dataset, percentile: float = 99.0) -> xr.Dataset:
        """Estimate capacity using percentile of generation values (default 99th to avoid outliers)."""
        logger.info(f"Estimating capacity using {percentile}th percentile of generation")
        
        capacity_values = []
        for loc_id in ds.location_id.values:
            gen_data = ds["generation_mw"].sel(location_id=loc_id)
            capacity = np.percentile(gen_data.values[~np.isnan(gen_data.values)], percentile)
            capacity_values.append(capacity)
            
        capacity_da = xr.DataArray(
            np.array(capacity_values, dtype=np.float32),
            dims=["location_id"],
            coords={"location_id": ds.location_id},
            name="capacity_mwp"
        )
        
        capacity_broadcast = capacity_da.broadcast_like(ds["generation_mw"])
        ds["capacity_mwp"] = capacity_broadcast
        
        return ds
        
    def add_location_metadata(self, ds: xr.Dataset) -> xr.Dataset:
        """Add location metadata. Currently handled in transform_to_schema."""
        return ds
        
    def validate_data(self, ds: xr.Dataset) -> bool:
        """Check if dataset has all required dims, vars, and coords."""
        logger.info("Validating dataset schema")
        
        required_dims = {"time_utc", "location_id"}
        if not required_dims.issubset(set(ds.dims)):
            logger.error(f"Missing required dimensions. Expected {required_dims}, got {set(ds.dims)}")
            return False
            
        required_vars = {"generation_mw", "capacity_mwp"}
        if not required_vars.issubset(set(ds.data_vars)):
            logger.error(f"Missing required variables. Expected {required_vars}, got {set(ds.data_vars)}")
            return False
            
        required_coords = {"time_utc", "location_id", "longitude", "latitude"}
        if not required_coords.issubset(set(ds.coords)):
            logger.error(f"Missing required coordinates. Expected {required_coords}, got {set(ds.coords)}")
            return False
            
        gen_missing = ds["generation_mw"].isnull().sum().values
        if gen_missing > 0:
            logger.warning(f"Found {gen_missing} missing values in generation_mw")
            
        if ds["generation_mw"].dtype != np.float32:
            logger.warning(f"generation_mw dtype is {ds['generation_mw'].dtype}, expected float32")
            
        logger.info("✅ Dataset validation passed")
        return True
        
    def save_to_zarr(self, ds: xr.Dataset, path: str, mode: str = "w"):
        """Save to Zarr, handling timezone conversion for compatibility."""
        logger.info(f"Saving dataset to {path}")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Zarr doesn't handle datetime64[ns, UTC] well, convert to timezone-naive datetime64[ns]
        ds_to_save = ds.copy()
        if "time_utc" in ds_to_save.coords:
            time_values = pd.DatetimeIndex(ds_to_save["time_utc"].values).tz_localize(None)
            ds_to_save["time_utc"] = time_values
        
        ds_to_save.to_zarr(path, mode=mode)
        logger.info(f"✅ Dataset saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess EIA solar generation data for ocf-data-sampler"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="List of region codes (e.g., CAISO ERCOT). Default: US48"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for Zarr file"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="EIA API key (optional, uses EIA_API_KEY env var if not provided)"
    )
    parser.add_argument(
        "--frequency",
        default="hourly",
        choices=["hourly", "daily"],
        help="Data frequency (default: hourly)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    preprocessor = EIAPreprocessor(api_key=args.api_key)
    
    try:
        ds = preprocessor.fetch_and_preprocess(
            start_date=args.start_date,
            end_date=args.end_date,
            regions=args.regions,
            output_path=args.output,
            frequency=args.frequency,
        )
        
        logger.info("Preprocessing completed successfully!")
        logger.info(f"Dataset shape: {ds.dims}")
        logger.info(f"Variables: {list(ds.data_vars)}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
