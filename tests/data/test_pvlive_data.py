import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pytz
from open_data_pvnet.scripts.fetch_pvlive_data import PVLiveData


@pytest.fixture(autouse=True)
def mock_pvlive():
    """
    Fixture to mock PVLive class globally for all tests in this module.
    This prevents any actual API calls during test execution.
    """
    with patch('pvlive_api.pvlive.PVLive', autospec=True) as mock:
        mock_instance = mock.return_value
        mock_instance.latest = MagicMock()
        mock_instance.between = MagicMock()
        mock_instance.at_time = MagicMock()
        mock_instance._get_gsp_list = MagicMock(return_value=[])
        yield mock


@pytest.fixture
def pvlive_data(mock_pvlive):
    """
    Fixture to create a PVLiveData instance with mocked PVLive methods.
    """
    return PVLiveData()


def test_get_latest_data(pvlive_data, mock_pvlive):
    """
    Test the get_latest_data method.
    """
    mock_data = {"column1": [1, 2], "column2": [3, 4]}
    mock_pvlive.return_value.latest.return_value = mock_data

    result = pvlive_data.get_latest_data(period=30)
    mock_pvlive.return_value.latest.assert_called_once_with(
        entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=True
    )
    assert result == mock_data


def test_get_data_between(pvlive_data, mock_pvlive):
    """
    Test the get_data_between method.
    """
    mock_data = {"column1": [5, 6], "column2": [7, 8]}
    mock_pvlive.return_value.between.return_value = mock_data

    start = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)
    end = datetime(2021, 1, 2, 12, 0, tzinfo=pytz.utc)

    result = pvlive_data.get_data_between(start, end)
    mock_pvlive.return_value.between.assert_called_once_with(
        start=start, end=end, entity_type="gsp", entity_id=0, extra_fields="", dataframe=True
    )
    assert result == mock_data


def test_get_data_at_time(pvlive_data, mock_pvlive):
    """
    Test the get_data_at_time method.
    """
    mock_data = {"column1": [9, 10], "column2": [11, 12]}
    mock_pvlive.return_value.at_time.return_value = mock_data

    dt = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)

    result = pvlive_data.get_data_at_time(dt)
    mock_pvlive.return_value.at_time.assert_called_once_with(
        dt, entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=True
    )
    assert result == mock_data
