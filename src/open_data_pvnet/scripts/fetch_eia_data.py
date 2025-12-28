import os
import logging
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, List, Union

logger = logging.getLogger(__name__)

class EIAData:
    """
    Class to fetch data from the EIA Open Data API.
    """
    BASE_URL = "https://api.eia.gov/v2"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EIA_API_KEY")
        if not self.api_key:
            logger.warning("EIA_API_KEY not found in environment variables.")

    def get_hourly_solar_data(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        ba_codes: Optional[List[str]] = None,
        timeout: int = 30
    ) -> pd.DataFrame:
        """
        Fetch hourly solar generation data for specific Balancing Authorities or all available.
        
        Args:
            start_date: Start date (inclusive) in 'YYYY-MM-DD' or 'YYYY-MM-DDTHH' format.
            end_date: End date (inclusive) in 'YYYY-MM-DD' or 'YYYY-MM-DDTHH' format.
            ba_codes: List of Balancing Authority codes (e.g., ['CISO', 'PJM']). If None, fetches for all.
            timeout: Request timeout in seconds.

        Returns:
            pd.DataFrame: DataFrame containing values, timestamps, and BA codes.
        """
        if not self.api_key:
            raise ValueError("API Key is required to fetch data.")

        # Ensure dates are strings in ISO format if they are datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.strftime("%Y-%m-%dT%H")
        if isinstance(end_date, datetime):
            end_date = end_date.strftime("%Y-%m-%dT%H")

        # Endpoint for hourly electricity generation by fuel type
        # Route: electricity/rto/fuel-type-data
        url = f"{self.BASE_URL}/electricity/rto/fuel-type-data/data/"

        params = {
            "api_key": self.api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[fueltype][]": "SUN",  # Solar
            "start": start_date,
            "end": end_date,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0,
            "length": 5000, # Max length per page
        }

        if ba_codes:
            # Add facets for respondent (BA)
            for ba in ba_codes:
                # Note: EIA API allows multiple values for a facet
                # But requests params dict with list value handles standard query string usually.
                # However, EIA might want 'facets[respondent][]': ['BA1', 'BA2']
                pass
            params["facets[respondent][]"] = ba_codes

        all_data = []
        offset = 0
        
        while True:
            current_params = params.copy()
            current_params["offset"] = offset
            try:
                response = requests.get(url, params=current_params, timeout=timeout)
                if not response.ok:
                    logger.error(f"EIA API Error: {response.text}")
                response.raise_for_status()
                data = response.json()
                
                if "response" not in data or "data" not in data["response"]:
                     logger.error(f"Unexpected response format: {data.keys()}")
                     break
                
                batch = data["response"]["data"]
                if not batch:
                    break
                    
                all_data.extend(batch)
                
                total = int(data["response"].get("total", 0))
                if len(all_data) >= total or len(batch) < 5000:
                    break
                
                offset += 5000
                
            except requests.RequestException as e:
                logger.error(f"Error fetching data from EIA: {e}")
                raise

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        
        # Parse timestamp
        # 'period' is usually in ISO format or similar for hourly 'YYYY-MM-DDTHH'
        df["period"] = pd.to_datetime(df["period"])
        
        # Rename columns to standard names
        df = df.rename(columns={
            "period": "timestamp",
            "value": "generation_mw",
            "respondent": "ba_code",
            "respondent-name": "ba_name"
        })
        
        # Select relevant columns
        cols_to_keep = ["timestamp", "ba_code", "ba_name", "generation_mw", "value-units"]
        # Filter existing columns
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        
        return df[cols_to_keep]
