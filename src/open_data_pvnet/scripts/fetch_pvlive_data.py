from pvlive_api import PVLive
import logging
import s3fs

logger = logging.getLogger(__name__)


class PVLiveData:
    def __init__(self, s3_bucket, config):
        self.pvl = PVLive()
        self.s3_bucket = s3_bucket
        self.config = config

    def get_latest_data(self, year, month, day):
        """
        Fetch the latest PVLive dataset for the given date, 
        ensure it has installedcapacity_mwp, and write to S3 as Zarr.
        """
        # --- 1) Fetch your dataset however your code does it ---
        # For example, maybe:
        # ds = self.pvl.at_time( ... )  
        # or some other API call that returns an xarray.Dataset
        ds = ...  

        # --- 2) Ensure installedcapacity_mwp is present (fallback to attr) ---
        if "installedcapacity_mwp" not in ds:
            ds = ds.assign(
                installedcapacity_mwp=ds.attrs.get("capacity_mwp", 0)
            )

        # --- 3) Write directly to Zarr on S3 instead of NetCDF/CSV ---
        store = s3fs.S3Map(
            f"{self.s3_bucket}/pvlive/{year}/{month:02d}/{day:02d}.zarr",
            s3=s3fs.S3FileSystem(anon=False),
            check=False,
        )
        ds.to_zarr(store, consolidated=True)

        return ds  # or return path / success flag

    def get_data_between(self, start, end, entity_type="gsp", entity_id=0, extra_fields=""):
        """
        Get the data between two dates
        """
        try:
            df = self.pvl.between(
                start=start,
                end=end,
                entity_type=entity_type,
                entity_id=entity_id,
                extra_fields=extra_fields,
                dataframe=True,
            )
            return df
        except Exception as e:
            logger.error("Error in get_data_between: %s", e)
            return None

    def get_data_at_time(self, dt, entity_type="gsp", entity_id=0, extra_fields="", period=30):
        """
        Get data at a specific time
        """
        try:
            df = self.pvl.at_time(
                dt,
                entity_type=entity_type,
                entity_id=entity_id,
                extra_fields=extra_fields,
                period=period,
                dataframe=True,
            )
            return df
        except Exception as e:
            logger.error("Error in get_data_at_time: %s", e)
            return None
