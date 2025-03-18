import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import pytz
import requests
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


@pytest.fixture(autouse=True)
def mock_requests():
    """
    Mock requests.get to prevent actual API calls.
    """
    with patch('requests.get') as mock_get:
        # Mock the GSP list response
        mock_get.return_value = MockResponse({
            "data": [
                {"gsp_id": 0, "gsp_name": "National"},
                {"gsp_id": 1, "gsp_name": "Region 1"}
            ]
        })
        yield mock_get


@pytest.fixture
def pvlive_data():
    """
    Fixture to create a PVLiveData instance with mocked PVLive methods.
    """
    with patch('pvlive_api.pvlive.PVLive._fetch_url', autospec=True) as mock_fetch:
        mock_fetch.return_value = {"data": [{"gsp_id": 0, "gsp_name": "National"}]}
        instance = PVLiveData()
        # Mock the methods after initialization
        instance.pvl.latest = MagicMock()
        instance.pvl.between = MagicMock()
        instance.pvl.at_time = MagicMock()
        return instance


def test_get_latest_data(pvlive_data):
    """
    Test the get_latest_data method.
    """
    mock_data = {"column1": [1, 2], "column2": [3, 4]}
    pvlive_data.pvl.latest.return_value = mock_data

    result = pvlive_data.get_latest_data(period=30)
    pvlive_data.pvl.latest.assert_called_once_with(
        entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=True
    )
    assert result == mock_data


def test_get_data_between(pvlive_data):
    """
    Test the get_data_between method.
    """
    mock_data = {"column1": [5, 6], "column2": [7, 8]}
    pvlive_data.pvl.between.return_value = mock_data

    start = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)
    end = datetime(2021, 1, 2, 12, 0, tzinfo=pytz.utc)

    result = pvlive_data.get_data_between(start, end)
    pvlive_data.pvl.between.assert_called_once_with(
        start=start, end=end, entity_type="gsp", entity_id=0, extra_fields="", dataframe=True
    )
    assert result == mock_data


def test_get_data_at_time(pvlive_data):
    """
    Test the get_data_at_time method.
    """
    mock_data = {"column1": [9, 10], "column2": [11, 12]}
    pvlive_data.pvl.at_time.return_value = mock_data

    dt = datetime(2021, 1, 1, 12, 0, tzinfo=pytz.utc)

    result = pvlive_data.get_data_at_time(dt)
    pvlive_data.pvl.at_time.assert_called_once_with(
        dt, entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=True
    )
    assert result == mock_data
