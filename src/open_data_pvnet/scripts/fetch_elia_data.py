import logging
from typing import Optional, List

import pandas as pd
import requests
import xarray as xr

logger = logging.getLogger(__name__)


class EliaData:
    """
    Class to handle interactions with the Elia (Belgium TSO) Open Data API.

    Elia provides public solar generation data via the Opendatasoft platform.
    No API key is required.

    Reference: https://opendata.elia.be/explore/dataset/ods087/
    """

    def __init__(self) -> None:
        self.base_url = (
            "https://opendata.elia.be/api/explore/v2.1/catalog/datasets"
        )
        self.default_dataset = "ods087"

    def get_data(
        self,
        start_date: str,
        end_date: str,
        dataset: str = "ods087",
        limit: int = 100,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch solar generation data from the Elia Open Data API.

        Automatically paginates through all available results for the
        requested date range.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            dataset: Elia dataset identifier (default: ods087 for solar PV)
            limit: Number of records per API page (max 100)

        Returns:
            pd.DataFrame with solar generation records, or None if error/empty
        """
        url = f"{self.base_url}/{dataset}/records"

        where_clause = (
            f"datetime >= '{start_date}T00:00:00Z' "
            f"AND datetime <= '{end_date}T23:59:59Z'"
        )

        params = {
            "where": where_clause,
            "order_by": "datetime ASC",
            "limit": limit,
            "offset": 0,
        }

        all_data: List[dict] = []
        current_offset = 0

        try:
            while True:
                # Create a fresh copy to avoid mutating the original params
                request_params = params.copy()
                request_params["offset"] = current_offset

                logger.info(
                    f"Fetching data from {url}, offset={current_offset}..."
                )
                response = requests.get(url, params=request_params)
                response.raise_for_status()

                payload = response.json()
                results = payload.get("results", [])

                if not results:
                    logger.info("No more data returned from API.")
                    break

                all_data.extend(results)

                if len(results) < limit:
                    break

                current_offset += limit

            if not all_data:
                logger.warning("No data retrieved.")
                return None

            return pd.DataFrame(all_data)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if "response" in locals() and response is not None:
                logger.error(f"Response: {response.text}")
            return None

    def get_dataset(
        self,
        start_date: str,
        end_date: str,
        dataset: str = "ods087",
    ) -> Optional[xr.Dataset]:
        """
        Fetch data and convert to xarray Dataset compatible with ocf-data-sampler.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            dataset: Elia dataset identifier

        Returns:
            xr.Dataset with datetime_utc index, or None if no data
        """
        df = self.get_data(
            start_date=start_date,
            end_date=end_date,
            dataset=dataset,
        )

        if df is None or df.empty:
            return None

        # Convert datetime column to proper UTC datetime
        if "datetime" in df.columns:
            df["datetime_utc"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.drop(columns=["datetime"])

        # Select numeric columns for the dataset
        value_cols = [
            c
            for c in df.columns
            if c not in ("datetime_utc", "resolutioncode", "mostrecent")
        ]

        # Ensure numeric conversion for value columns
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop duplicates and set index
        df = df.drop_duplicates(subset=["datetime_utc"])
        df = df.set_index("datetime_utc")

        ds = xr.Dataset.from_dataframe(df)

        return ds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    elia = EliaData()
    print(
        "EliaData initialized. "
        "Call get_data(start_date, end_date) to fetch Belgium solar data."
    )