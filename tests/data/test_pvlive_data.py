import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pytz
import requests
import pandas as pd
from pvlive_api.pvlive import PVLiveException
from open_data_pvnet.scripts.fetch_pvlive_data import PVLiveData


class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.exceptions.HTTPError()


@pytest.fixture
def mock_pvlive():
    """Mock the PVLive class"""
    with patch('pvlive_api.pvlive.PVLive', autospec=True) as mock_pvlive_class:
        # Create a mock instance
        mock_instance = mock_pvlive_class.return_value
        
        # Create a mock DataFrame for gsp_list
        mock_gsp_df = pd.DataFrame({
            'gsp_id': [0, 1],
            'gsp_name': ['National', 'Region 1']
        })
        mock_instance.gsp_list = mock_gsp_df
        
        # Mock the API methods
        mock_instance.latest = MagicMock()
        mock_instance.between = MagicMock()
        mock_instance.at_time = MagicMock()
        
        # Mock _get_gsp_list to return the mock DataFrame
        mock_instance._get_gsp_list = MagicMock(return_value=mock_gsp_df)
        
        yield mock_instance


@pytest.fixture
def pvlive_data(mock_pvlive):
    """
    Fixture to create a PVLiveData instance with mocked PVLive.
    """
    with patch('open_data_pvnet.scripts.fetch_pvlive_data.PVLive', return_value=mock_pvlive):
        instance = PVLiveData()
        return instance


def test_get_latest_data(pvlive_data, mock_pvlive):
    """
    Test the get_latest_data method.
    """
    mock_data = pd.DataFrame({
        'datetime_gmt': [datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)],
        'generation_mw': [100.0]
    })
    mock_pvlive.latest.return_value = mock_data

    result = pvlive_data.get_latest_data(period=30)
    mock_pvlive.latest.assert_called_once_with(
        entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=True
    )
    assert result.equals(mock_data)


def test_get_latest_data_error(pvlive_data, mock_pvlive):
    """
    Test error handling in get_latest_data method.
    """
    mock_pvlive.latest.side_effect = Exception("API Error")
    result = pvlive_data.get_latest_data(period=30)
    assert result is None


def test_get_data_between(pvlive_data, mock_pvlive):
    """
    Test the get_data_between method.
    """
    mock_data = pd.DataFrame({
        'datetime_gmt': [
            datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc),
            datetime(2021, 1, 1, 12, 30, tzinfo=pytz.utc)
        ],
        'generation_mw': [100.0, 150.0]
    })
    mock_pvlive.between.return_value = mock_data

    start = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)
    end = datetime(2021, 1, 2, 12, 0, tzinfo=pytz.utc)

    result = pvlive_data.get_data_between(start, end)
    mock_pvlive.between.assert_called_once_with(
        start=start, end=end, entity_type="gsp", entity_id=0, extra_fields="", dataframe=True
    )
    assert result.equals(mock_data)


def test_get_data_between_error(pvlive_data, mock_pvlive):
    """
    Test error handling in get_data_between method.
    """
    start = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)
    end = datetime(2021, 1, 2, 12, 0, tzinfo=pytz.utc)
    mock_pvlive.between.side_effect = Exception("API Error")
    result = pvlive_data.get_data_between(start, end)
    assert result is None


def test_get_data_at_time(pvlive_data, mock_pvlive):
    """
    Test the get_data_at_time method.
    """
    mock_data = pd.DataFrame({
        'datetime_gmt': [datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)],
        'generation_mw': [100.0]
    })
    mock_pvlive.at_time.return_value = mock_data

    dt = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)

    result = pvlive_data.get_data_at_time(dt)
    mock_pvlive.at_time.assert_called_once_with(
        dt, entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=True
    )
    assert result.equals(mock_data)


def test_get_data_at_time_error(pvlive_data, mock_pvlive):
    """
    Test error handling in get_data_at_time method.
    """
    dt = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)
    mock_pvlive.at_time.side_effect = Exception("API Error")
    result = pvlive_data.get_data_at_time(dt)
    assert result is None
