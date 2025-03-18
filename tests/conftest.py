import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime
import pytz
import requests
from pvlive_api.pvlive import PVLiveException


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


@pytest.fixture(scope="session", autouse=True)
def mock_requests():
    """Mock requests.get to prevent actual API calls."""
    with patch('requests.get') as mock_get:
        mock_get.return_value = MockResponse({
            "data": [
                {"gsp_id": 0, "gsp_name": "National", "pes_id": 0},
                {"gsp_id": 1, "gsp_name": "Region 1", "pes_id": 1}
            ]
        })
        yield mock_get


@pytest.fixture(scope="session", autouse=True)
def mock_pvlive():
    """Mock PVLive API methods to prevent actual API calls."""
    # Create mock data as a DataFrame with all required columns
    mock_data = pd.DataFrame({
        "generation_mw": [100, 200],
        "datetime_gmt": [
            datetime(2021, 1, 1, 12, 0, tzinfo=pytz.UTC),
            datetime(2021, 1, 1, 12, 30, tzinfo=pytz.UTC)
        ],
        "pes_id": [0, 0],  # Added pes_id column
        "gsp_id": [0, 0],  # Added gsp_id column
        "gsp_name": ["National", "National"]  # Added gsp_name column
    })

    with patch('pvlive_api.pvlive.PVLive._fetch_url', autospec=True) as mock_fetch:
        mock_fetch.return_value = {
            "data": [{"gsp_id": 0, "gsp_name": "National", "pes_id": 0}],
            "meta": ["gsp_id", "gsp_name", "pes_id"]
        }
        
        # Mock the main PVLive methods
        with patch('pvlive_api.pvlive.PVLive.latest') as mock_latest, \
             patch('pvlive_api.pvlive.PVLive.between') as mock_between, \
             patch('pvlive_api.pvlive.PVLive.at_time') as mock_at_time:
            
            mock_latest.return_value = mock_data
            mock_between.return_value = mock_data
            mock_at_time.return_value = mock_data
            
            yield


def pytest_ignore_collect(path, config):
    """Ignore test files in the data directory."""
    if str(path).startswith(os.path.join('tests', 'data')):
        return True 