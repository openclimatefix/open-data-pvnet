from pvlive_api import PVLive
import logging


logger = logging.getLogger(__name__)


class PVLiveData:
    def __init__(self):
        self.pvl = PVLive()

    def get_latest_data(self, period, entity_type="gsp", entity_id=0, extra_fields=""):
        """
        Get the latest data from PVlive
        """
        try:
            df = self.pvl.latest(
                entity_type=entity_type,
                entity_id=entity_id,
                extra_fields=extra_fields,
                period=period,
                dataframe=True,
            )
            return df
        except Exception as e:
            logger.error(e)
            return None

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
            logger.error(e)
            return None

    def get_data_at_time(self, dt):
        """
        Get data at a specific time
        """
        try:
            df = self.pvl.at_time(
                dt, entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=True
            )
            return df
        except Exception as e:
            logger.error(e)
            return None
