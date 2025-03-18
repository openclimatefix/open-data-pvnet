import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pytz
import requests
import pandas as pd
import json
from pvlive_api.pvlive import PVLiveException
from open_data_pvnet.scripts.fetch_pvlive_data import PVLiveData


class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.exceptions.HTTPError()


@pytest.fixture(autouse=True)
def mock_requests():
    """
    Mock requests.get to prevent actual API calls.
    """
    with patch('requests.get') as mock_get:
        # Mock the GSP list response with pes_id
        mock_get.return_value = MockResponse({
            "data": [
                {"gsp_id": 0, "gsp_name": "National", "pes_id": 0},
                {"gsp_id": 1, "gsp_name": "Region 1", "pes_id": 1}
            ],
            "meta": ["gsp_id", "gsp_name", "pes_id"]
        })
        yield mock_get


@pytest.fixture
def pvlive_data():
    """
    Fixture to create a PVLiveData instance with mocked PVLive methods.
    """
    with patch('pvlive_api.PVLive') as mock_pvlive_class:
        instance = PVLiveData()
        return instance


def test_get_latest_data(pvlive_data):
    """
    Test the get_latest_data method.
    """
    mock_data = pd.DataFrame({"column1": [1, 2], "column2": [3, 4]})
    with patch.object(pvlive_data.pvl, 'latest', return_value=mock_data) as mock_latest:
        result = pvlive_data.get_latest_data(period=30)
        mock_latest.assert_called_once_with(
            entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=True
        )
        assert result.equals(mock_data)


def test_get_latest_data_error(pvlive_data):
    """
    Test error handling in get_latest_data method.
    """
    with patch.object(pvlive_data.pvl, 'latest', side_effect=Exception("API Error")):
        result = pvlive_data.get_latest_data(period=30)
        assert result is None


def test_get_data_between(pvlive_data):
    """
    Test the get_data_between method.
    """
    mock_data = pd.DataFrame({"column1": [5, 6], "column2": [7, 8]})
    start = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)
    end = datetime(2021, 1, 2, 12, 0, tzinfo=pytz.utc)

    with patch.object(pvlive_data.pvl, 'between', return_value=mock_data) as mock_between:
        result = pvlive_data.get_data_between(start, end)
        mock_between.assert_called_once_with(
            start=start, end=end, entity_type="gsp", entity_id=0, extra_fields="", dataframe=True
        )
        assert result.equals(mock_data)


def test_get_data_between_error(pvlive_data):
    """
    Test error handling in get_data_between method.
    """
    start = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)
    end = datetime(2021, 1, 2, 12, 0, tzinfo=pytz.utc)
    with patch.object(pvlive_data.pvl, 'between', side_effect=Exception("API Error")):
        result = pvlive_data.get_data_between(start, end)
        assert result is None


def test_get_data_at_time(pvlive_data):
    """
    Test the get_data_at_time method.
    """
    mock_data = pd.DataFrame({"column1": [9, 10], "column2": [11, 12]})
    dt = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)

    with patch.object(pvlive_data.pvl, 'at_time', return_value=mock_data) as mock_at_time:
        result = pvlive_data.get_data_at_time(dt)
        mock_at_time.assert_called_once_with(
            dt, entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=True
        )
        assert result.equals(mock_data)


def test_get_data_at_time_error(pvlive_data):
    """
    Test error handling in get_data_at_time method.
    """
    dt = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)
    with patch.object(pvlive_data.pvl, 'at_time', side_effect=Exception("API Error")):
        result = pvlive_data.get_data_at_time(dt)
        assert result is None 