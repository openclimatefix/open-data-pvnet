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
    """Mock PVLive class and its methods to prevent actual API calls."""
    mock_data = pd.DataFrame({
        "generation_mw": [100, 200],
        "datetime_gmt": [
            datetime(2021, 1, 1, 12, 0, tzinfo=pytz.UTC),
            datetime(2021, 1, 1, 12, 30, tzinfo=pytz.UTC)
        ],
        "pes_id": [0, 0],
        "gsp_id": [0, 0],
        "gsp_name": ["National", "National"]
    })

    class MockPVLive:
        def __init__(self, *args, **kwargs):
            pass

        def latest(self, *args, **kwargs):
            return mock_data

        def between(self, *args, **kwargs):
            return mock_data

        def at_time(self, *args, **kwargs):
            return mock_data

    with patch('pvlive_api.pvlive.PVLive', MockPVLive):
        yield MockPVLive()


def pytest_ignore_collect(path, config):
    """Ignore test files in the data directory."""
    if str(path).startswith(os.path.join('tests', 'data')):
        return True 