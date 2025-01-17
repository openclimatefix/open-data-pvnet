import pytest
from unittest.mock import MagicMock
from datetime import datetime
import pytz
from open_data_pvnet.scripts.fetch_pvlive_data import PVLiveData


@pytest.fixture
def pvlive_mock():
    """
    Fixture to create a PVLiveData instance with mocked PVLive methods.
    """
    pvlive = PVLiveData()
    pvlive.pvl.latest = MagicMock()
    pvlive.pvl.between = MagicMock()
    pvlive.pvl.at_time = MagicMock()
    return pvlive


def test_get_latest_data(pvlive_mock):
    """
    Test the get_latest_data method.
    """
    mock_data = {"column1": [1, 2], "column2": [3, 4]}
    pvlive_mock.pvl.latest.return_value = mock_data

    result = pvlive_mock.get_latest_data(period=30)
    pvlive_mock.pvl.latest.assert_called_once_with(
        entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=True
    )
    assert result == mock_data


def test_get_data_between(pvlive_mock):
    """
    Test the get_data_between method.
    """
    mock_data = {"column1": [5, 6], "column2": [7, 8]}
    pvlive_mock.pvl.between.return_value = mock_data

    start = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)
    end = datetime(2021, 1, 2, 12, 0, tzinfo=pytz.utc)

    result = pvlive_mock.get_data_between(start, end)
    pvlive_mock.pvl.between.assert_called_once_with(
        start=start, end=end, entity_type="gsp", entity_id=0, extra_fields="", dataframe=True
    )
    assert result == mock_data


def test_get_data_at_time(pvlive_mock):
    """
    Test the get_data_at_time method.
    """
    mock_data = {"column1": [9, 10], "column2": [11, 12]}
    pvlive_mock.pvl.at_time.return_value = mock_data

    dt = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)

    result = pvlive_mock.get_data_at_time(dt)
    pvlive_mock.pvl.at_time.assert_called_once_with(
        dt, entity_type="pes", entity_id=0, extra_fields="", period=30, dataframe=True
    )
    assert result == mock_data
