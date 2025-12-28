import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from open_data_pvnet.scripts.fetch_eia_data import EIAData

@pytest.fixture
def mock_response_data():
    return {
        "response": {
            "total": 2,
            "data": [
                {
                    "period": "2023-01-01T00",
                    "respondent": "CISO",
                    "respondent-name": "California Independent System Operator",
                    "fueltypeid": "SUN",
                    "type-name": "Solar",
                    "value": 1000,
                    "value-units": "megawatthours"
                },
                {
                    "period": "2023-01-01T01",
                    "respondent": "CISO",
                    "respondent-name": "California Independent System Operator",
                    "fueltypeid": "SUN",
                    "type-name": "Solar",
                    "value": 1200,
                    "value-units": "megawatthours"
                }
            ]
        }
    }

def test_init_no_api_key():
    with patch.dict("os.environ", {}, clear=True):
        eia = EIAData()
        assert eia.api_key is None

def test_init_with_env_var():
    with patch.dict("os.environ", {"EIA_API_KEY": "test_key"}, clear=True):
        eia = EIAData()
        assert eia.api_key == "test_key"

def test_get_hourly_solar_data_success(mock_response_data):
    eia = EIAData(api_key="test_key")
    
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        df = eia.get_hourly_solar_data(
            start_date="2023-01-01T00",
            end_date="2023-01-01T01",
            ba_codes=["CISO"]
        )

        assert not df.empty
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "generation_mw" in df.columns
        assert df.iloc[0]["generation_mw"] == 1000
        assert df.iloc[1]["generation_mw"] == 1200
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        
        # Verify call args
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["api_key"] == "test_key"
        assert kwargs["params"]["facets[fueltypeid][]"] == "SUN"
        assert kwargs["params"]["facets[respondent][]"] == ["CISO"]

def test_get_hourly_solar_data_pagination():
    eia = EIAData(api_key="test_key")
    
    # Create a scenario where total is 6000 and we get 5000 in first batch
    first_batch = [{"period": "2023-01-01T00", "value": i, "respondent": "CISO"} for i in range(5000)]
    second_batch = [{"period": "2023-01-02T00", "value": i, "respondent": "CISO"} for i in range(1000)]
    
    response1 = {"response": {"total": 6000, "data": first_batch}}
    response2 = {"response": {"total": 6000, "data": second_batch}}
    
    with patch("requests.get") as mock_get:
        mock_response1 = MagicMock()
        mock_response1.json.return_value = response1
        
        mock_response2 = MagicMock()
        mock_response2.json.return_value = response2
        
        # Side effect to return different responses
        mock_get.side_effect = [mock_response1, mock_response2]

        df = eia.get_hourly_solar_data("2023-01-01", "2023-01-02")
        
        assert len(df) == 6000
        assert mock_get.call_count == 2
        
        # Check offsets
        call_args_list = mock_get.call_args_list
        assert call_args_list[0][1]["params"]["offset"] == 0
        assert call_args_list[1][1]["params"]["offset"] == 5000

def test_missing_api_key_error():
    eia = EIAData(api_key=None)
    with pytest.raises(ValueError, match="API Key is required"):
        eia.get_hourly_solar_data("2023-01-01", "2023-01-02")
