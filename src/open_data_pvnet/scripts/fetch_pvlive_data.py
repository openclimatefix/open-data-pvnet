from pvlive_api import PVLive
import logging
from datetime import datetime
import pytz
import pandas as pd

logger = logging.getLogger(__name__)

class PVLiveData:
    def __init__(self):
        self.pvl = PVLive()

    def get_latest_data(self, period, entity_type="gsp", entity_id=0, extra_fields=""):
        """
        Get the latest data from PVlive
        """
        try:
            if entity_id == 0 and entity_type == "gsp":
                return self._get_latest_data_for_all_gsps(period, extra_fields)
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

    def get_data_between(self, start, end, period, entity_type="gsp", entity_id=0, extra_fields=""):
        """
        Get the data between two dates
        """
        try:
            if entity_id == 0 and entity_type == "gsp":
                return self._get_data_between_for_all_gsps(start, end, period, extra_fields)
            df = self.pvl.between(
                start=start,
                end=end,
                period=period,
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

    def _get_latest_data_for_all_gsps(self, period, extra_fields):
        data = None
        for gsp_id in self.pvl.gsp_ids:
            data_ = self.pvl.latest(
                entity_type="gsp",
                entity_id=gsp_id,
                extra_fields=extra_fields,
                period=period,
                dataframe=True,
            )
            if data is None:
                data = data_
            else:
                data = pd.concat((data, data_), ignore_index=True)
        return data

    def _get_data_between_for_all_gsps(self, start, end, period, extra_fields):
        data = None
        for gsp_id in self.pvl.gsp_ids:
            data_ = self.pvl.between(
                period=period,
                start=start,
                end=end,
                entity_type="gsp",
                entity_id=gsp_id,
                extra_fields=extra_fields,
                dataframe=True,
            )
            if data is None:
                data = data_
            else:
                data = pd.concat((data, data_), ignore_index=True)
        return data

