import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

def pytest_ignore_collect(path, config):
    """Ignore test files in the data directory."""
    if str(path).startswith(os.path.join('tests', 'data')):
        return True

@pytest.fixture(scope="session", autouse=True)
def mock_pvlive():
    """Mock PVLive API at the session level to prevent any real API calls."""
    mock_data = pd.DataFrame({"column1": [1, 2], "column2": [3, 4]})
    mock_pvlive = MagicMock()
    mock_pvlive.latest.return_value = mock_data
    mock_pvlive.between.return_value = mock_data
    mock_pvlive.at_time.return_value = mock_data
    
    with patch('pvlive_api.PVLive', return_value=mock_pvlive):
        yield mock_pvlive 