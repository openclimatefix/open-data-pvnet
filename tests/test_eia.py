import pytest
import pandas as pd
from unittest.mock import Mock, patch
from open_data_pvnet.scripts.fetch_eia_data import EIAData

@pytest.fixture
def mock_response():
    """Fixture to mock a successful API response."""
    mock = Mock()
    mock.json.return_value = {
        "response": {
            "data": [
                {"period": "2023-01-01T00", "value": 100, "fueltype": "SUN"},
                {"period": "2023-01-01T01", "value": 150, "fueltype": "SUN"},
            ]
        }
    }
    mock.raise_for_status.return_value = None
    return mock

def test_init_with_key():
    eia = EIAData(api_key="test_key")
    assert eia.api_key == "test_key"

def test_init_without_key(mocker):
    mocker.patch.dict("os.environ", {}, clear=True)
    eia = EIAData()
    assert eia.api_key is None

def test_get_data_success(mock_response):
    with patch("requests.get", return_value=mock_response) as mock_get:
        eia = EIAData(api_key="test_key")
        
        df = eia.get_data(
            route="test/route",
            start_date="2023-01-01",
            end_date="2023-01-02",
            frequency="hourly",
            data_cols=["value"],
            facets={"fueltype": "SUN"}
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "value" in df.columns
        
        # Verify API call parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["api_key"] == "test_key"
        assert kwargs["params"]["facets[fueltype][]"] == "SUN"
        assert kwargs["params"]["data[0]"] == "value"

def test_get_data_missing_key():
    eia = EIAData(api_key=None)
    with pytest.raises(ValueError, match="API Key is missing"):
        eia.get_data("route", "start", "end", frequency="hourly")

def test_get_data_api_error():
    mock_resp = Mock()
    import requests
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("API Error")
    
    with patch("requests.get", return_value=mock_resp):
        eia = EIAData(api_key="test_key")
        df = eia.get_data("route", "start", "end", frequency="hourly")
        assert df is None

def test_get_data_empty_response():
    mock_resp = Mock()
    mock_resp.json.return_value = {"response": {"data": []}}
    mock_resp.raise_for_status.return_value = None
    
    with patch("requests.get", return_value=mock_resp):
        eia = EIAData(api_key="test_key")
        df = eia.get_data("route", "start", "end", frequency="hourly")
        assert df is None

def test_get_data_with_region(mock_response):
    with patch("requests.get", return_value=mock_response) as mock_get:
        eia = EIAData(api_key="test_key")
        eia.get_data("route", "start", "end")
        
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["facets[respondent][0]"] == "US48"

def test_get_data_without_region(mock_response):
    with patch("requests.get", return_value=mock_response) as mock_get:
        eia = EIAData(api_key="test_key")
        eia.get_data("route", "start", "end", region=None)
        
        args, kwargs = mock_get.call_args
        assert not any("facets[respondent]" in k for k in kwargs["params"].keys())

def test_get_dataset_success(mock_response):
    import xarray as xr
    with patch("requests.get", return_value=mock_response) as mock_get:
        eia = EIAData(api_key="test_key")
        
        ds = eia.get_dataset(
            route="test/route",
            start_date="2023-01-01",
            end_date="2023-01-02"
        )
        
        assert isinstance(ds, xr.Dataset)
        assert "datetime_gmt" in ds.coords or "datetime_gmt" in ds.indexes
        assert "value" in ds.data_vars
        assert len(ds.datetime_gmt) == 2

def test_get_data_pagination():
    page1 = {
        "response": {
            "data": [
                {"period": "2023-01-01T00", "value": 100},
                {"period": "2023-01-01T01", "value": 150},
            ]
        }
    }
    page2 = {
        "response": {
            "data": [
                {"period": "2023-01-01T02", "value": 200},
            ]
        }
    }
    
    mock_resp1 = Mock()
    mock_resp1.json.return_value = page1
    mock_resp1.raise_for_status.return_value = None
    
    mock_resp2 = Mock()
    mock_resp2.json.return_value = page2
    mock_resp2.raise_for_status.return_value = None
    
    with patch("requests.get", side_effect=[mock_resp1, mock_resp2]) as mock_get:
        eia = EIAData(api_key="test_key")
        
        df = eia.get_data("route", "start", "end", length=2)
        
        assert len(df) == 3
        assert mock_get.call_count == 2
        
        call_args_list = mock_get.call_args_list
        assert call_args_list[0][1]["params"]["offset"] == 0
        assert call_args_list[1][1]["params"]["offset"] == 2
