import pytest
import pandas as pd
import numpy as np
from open_data_pvnet.scripts.fetch_eia_data import EIAData
from unittest.mock import MagicMock, patch
import os
import shutil

# Mock EIAData since we tested it separately, we just want to test collector logic
from open_data_pvnet.scripts.collect_eia_data import main as collect_main
import xarray as xr

@pytest.fixture
def mock_args():
    return ["--start", "2023-01-01", "--end", "2023-01-02", "--bas", "CISO", "--output", "tmp_test_output.zarr"]

def test_collect_data(mock_args):
    mock_df = pd.DataFrame({
        "timestamp": ["2023-01-01T00", "2023-01-01T01"],
        "ba_code": ["CISO", "CISO"],
        "value": [100, 200],
        "ba_name": ["CAISO", "CAISO"],
        "value-units": ["MWh", "MWh"]
    })
    
    with patch("open_data_pvnet.scripts.collect_eia_data.EIAData") as MockEIA:
        instance = MockEIA.return_value
        instance.api_key = "test_key"
        instance.get_hourly_solar_data.return_value = mock_df
        
        with patch("sys.argv", ["script_name"] + mock_args):
            collect_main()
            
    assert os.path.exists("tmp_test_output.zarr")
    ds = xr.open_zarr("tmp_test_output.zarr", consolidated=False)
    assert "timestamp" in ds.coords
    assert "ba_code" in ds.coords
    assert "latitude" in ds.data_vars or "latitude" in ds.coords
    assert ds["latitude"].values[0] == 37.0 # CISO lat
    shutil.rmtree("tmp_test_output.zarr")
