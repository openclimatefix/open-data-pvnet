import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime
import pytz

def pytest_ignore_collect(path, config):
    """Ignore test files in the data directory."""
    if str(path).startswith(os.path.join('tests', 'data')):
        return True

@pytest.fixture(scope="session", autouse=True)
def mock_pvlive():
    """Mock PVLive API at the session level to prevent any real API calls."""
    mock_data = pd.DataFrame({
        "generation_mw": [100, 200],
        "datetime_gmt": [
            datetime(2021, 1, 1, 12, 0, tzinfo=pytz.UTC),
            datetime(2021, 1, 1, 12, 30, tzinfo=pytz.UTC)
        ]
    })
    
    # Create a mock class instead of instance to handle initialization
    class MockPVLive:
        def __init__(self, retries=3, proxies=None, timeout=10, ssl_verify=True):
            self.retries = retries
            self.proxies = proxies
            self.timeout = timeout
            self.ssl_verify = ssl_verify
            self.gsp_list = [
                {"gsp_id": 0, "gsp_name": "National", "pes_id": 0},
                {"gsp_id": 1, "gsp_name": "Region 1", "pes_id": 1}
            ]
            
        def _get_gsp_list(self):
            return self.gsp_list
            
        def _fetch_url(self, *args, **kwargs):
            return {"data": self.gsp_list, "meta": ["gsp_id", "gsp_name", "pes_id"]}
            
        def latest(self, entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=False):
            return mock_data if dataframe else mock_data.to_dict("records")
            
        def between(self, start, end, entity_type="gsp", entity_id=0, extra_fields="", dataframe=False):
            return mock_data if dataframe else mock_data.to_dict("records")
            
        def at_time(self, dt, entity_type="gsp", entity_id=0, extra_fields="", period=30, dataframe=False):
            return mock_data if dataframe else mock_data.to_dict("records")
    
    with patch('pvlive_api.PVLive', MockPVLive), \
         patch('pvlive_api.pvlive.PVLive', MockPVLive):  # Also patch the module-level import
        yield MockPVLive() 