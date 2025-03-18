import logging
from typing import Optional
from open_data_pvnet.nwp.met_office import process_met_office_data
from open_data_pvnet.nwp.gfs import process_gfs_data
from open_data_pvnet.nwp.dwd import process_dwd_data

logger = logging.getLogger(__name__)


def handle_archive(
    provider: str,
    year: int,
    month: int,
    day: int,
    hour: Optional[int] = None,
    region: str = "global",
    overwrite: bool = False,
    archive_type: str = "zarr.zip",
):
    """
    Handle the archive operation for different providers.

    Args:
        provider (str): The data provider (e.g., 'metoffice', 'gfs', 'dwd').
        year (int): Year of data.
        month (int): Month of data.
        day (int): Day of data.
        hour (Optional[int]): Hour of data (0-23), only used for Met Office data.
        region (str): Region for Met Office data ('global' or 'uk').
        overwrite (bool): Whether to overwrite existing files.
        archive_type (str): Type of archive to create ("zarr.zip" or "tar").
    """
    if provider == "metoffice":
        if region not in ["global", "uk"]:
            raise ValueError(
                f"Invalid region '{region}' for provider 'metoffice'. Must be 'global' or 'uk'."
            )

        hours = range(24) if hour is None else [hour]
        for hour in hours:
            logger.info(
                f"Processing Met Office {region} data for {year}-{month:02d}-{day:02d} at hour {hour:02d} with overwrite={overwrite}"
            )
            process_met_office_data(
                year=year,
                month=month,
                day=day,
                hour=hour,
                region=region,
                overwrite=overwrite,
                archive_type=archive_type,
            )

    elif provider == "gfs":
        logger.info(
            f"Processing GFS data for {year}-{month:02d}-{day:02d} at hour {hour:02d} with overwrite={overwrite}"
        )
        process_gfs_data(year, month, day, hour, overwrite=overwrite)
    elif provider == "dwd":
        logger.info(
            f"Processing DWD data for {year}-{month:02d}-{day:02d} at hour {hour:02d} with overwrite={overwrite}"
        )
        process_dwd_data(year, month, day, hour, overwrite=overwrite)
    else:
        raise NotImplementedError(f"Provider {provider} not yet implemented")
