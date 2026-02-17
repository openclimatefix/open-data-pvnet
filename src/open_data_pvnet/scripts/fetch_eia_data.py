import os
import logging
import requests
import pandas as pd
import xarray as xr
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class EIAData:
    """
    Class to handle interactions with the EIA API v2.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EIA_API_KEY")
        if not self.api_key:
            logger.warning("EIA_API_KEY environment variable is not set. You must provide an API key to fetch data.")
        self.base_url = "https://api.eia.gov/v2"

    def get_data(
        self, 
        route: str, 
        start_date: str, 
        end_date: str, 
        frequency: str = "hourly", 
        data_cols: List[str] = ["value"],
        facets: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        length: int = 5000,
        region: str = "US48",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from the EIA API.
        
        Args:
            route: API route (e.g. 'electricity/rto/daily-fuel-type-data')
            frequency: Data frequency (e.g. 'daily', 'hourly')
            start_date: Start date string
            end_date: End date string
            data_cols: List of data columns to retrieve
            facets: Dictionary of facets to filter by
            offset: Pagination offset
            length: Number of results to return
            region: Region identifier (default: "US48")
            
        Returns:
            pd.DataFrame: Data returned from the API, or None if error/empty
        """
        if not self.api_key:
             raise ValueError("API Key is missing")

        if region:
            if facets is None:
                facets = {}
            if "respondent" not in facets and region == "US48":
                facets["respondent"] = ["US48"]

        url = f"{self.base_url}/{route}/data"
        
        params = {
            "api_key": self.api_key,
            "frequency": frequency,
            "start": start_date,
            "end": end_date,
            "offset": offset,
            "length": length,
        }
        
        for i, col in enumerate(data_cols):
            params[f"data[{i}]"] = col
            
        if facets:
            for key, value in facets.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        params[f"facets[{key}][{i}]"] = v
                else:
                    params[f"facets[{key}][]"] = value

        all_data = []
        
        try:
            current_offset = offset
            while True:
                # Create a fresh copy of params for each request to avoid mutating history
                request_params = params.copy()
                request_params["offset"] = current_offset
                
                logger.info(f"Fetching data from {url}, offset={current_offset}...")
                response = requests.get(url, params=request_params)
                response.raise_for_status()
                
                payload = response.json()
                if "response" in payload and "data" in payload["response"]:
                    data = payload["response"]["data"]
                    if not data:
                        logger.info("No more data returned from API.")
                        break
                    
                    all_data.extend(data)
                    
                    if len(data) < length:
                        break
                        
                    current_offset += length
                else:
                    logger.error(f"Unexpected API response format: {payload.keys()}")
                    break
            
            if not all_data:
                logger.warning("No data retrieved.")
                return None
                
            return pd.DataFrame(all_data)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if 'response' in locals() and response is not None:
                logger.error(f"Response: {response.text}")
            return None

    def get_dataset(
        self,
        route: str,
        start_date: str,
        end_date: str,
        frequency: str = "hourly",
        data_cols: List[str] = ["value"],
        facets: Optional[Dict[str, Any]] = None,
        region: str = "US48",
    ) -> Optional[xr.Dataset]:
        """
        Fetch data and convert to xarray Dataset compatible with ocf-data-sampler.
        
        Args:
            route: API route
            start_date: Start date string
            end_date: End date string
            frequency: Data frequency
            data_cols: List of data columns
            facets: Dictionary of facets
            region: Region identifier
            
        Returns:
            xr.Dataset: Dataset with datetime_gmt index, or None if no data
        """
        df = self.get_data(
            route=route,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            data_cols=data_cols,
            facets=facets,
            region=region
        )
        
        if df is None or df.empty:
            return None
            
        # Process for ocf-data-sampler format
        if "period" in df.columns:
            df["datetime_gmt"] = pd.to_datetime(df["period"], utc=True)
            df = df.drop(columns=["period"])
            
        index_cols = ["datetime_gmt"]
        if "respondent" in df.columns:
            index_cols.append("respondent")
        elif "region" in df.columns:
            index_cols.append("region")
            
        if not df.index.is_unique:
             df = df.drop_duplicates(subset=index_cols)
             
        df = df.set_index(index_cols)
        
        ds = xr.Dataset.from_dataframe(df)
        
        return ds

if __name__ == "__main__":
    # Basic test execution
    logging.basicConfig(level=logging.INFO)
    eia = EIAData()
    print("EIAData initialized. Set EIA_API_KEY and call get_data() to test.")
