import pytest
import pandas as pd
from unittest.mock import Mock, patch

from open_data_pvnet.scripts.fetch_elia_data import EliaData


@pytest.fixture
def mock_response():
    """Fixture to mock a successful Elia API response."""
    mock = Mock()
    mock.json.return_value = {
        "results": [
            {
                "datetime": "2024-06-15T12:00:00+00:00",
                "measured": 2500.0,
                "mostrecentforecast": 2450.0,
                "monitoredcapacity": 7500.0,
                "resolutioncode": "PT15M",
            },
            {
                "datetime": "2024-06-15T12:15:00+00:00",
                "measured": 2520.0,
                "mostrecentforecast": 2460.0,
                "monitoredcapacity": 7500.0,
                "resolutioncode": "PT15M",
            },
        ]
    }
    mock.raise_for_status.return_value = None
    return mock


def test_init():
    """EliaData should initialize without any API key."""
    elia = EliaData()
    assert elia.base_url == (
        "https://opendata.elia.be/api/explore/v2.1/catalog/datasets"
    )
    assert elia.default_dataset == "ods087"


def test_get_data_success(mock_response):
    """Should return a DataFrame with solar generation data."""
    with patch("requests.get", return_value=mock_response) as mock_get:
        elia = EliaData()

        df = elia.get_data(
            start_date="2024-06-15",
            end_date="2024-06-15",
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "measured" in df.columns
        assert "datetime" in df.columns

        # Verify API call was made
        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        assert "ods087" in kwargs["params"]["where"] or "ods087" in _[0]


def test_get_data_custom_dataset(mock_response):
    """Should use the specified dataset ID in the API URL."""
    with patch("requests.get", return_value=mock_response) as mock_get:
        elia = EliaData()
        elia.get_data("2024-01-01", "2024-01-01", dataset="ods088")

        args, _ = mock_get.call_args
        assert "ods088" in args[0]


def test_get_data_empty_response():
    """Should return None when no data is available."""
    mock_resp = Mock()
    mock_resp.json.return_value = {"results": []}
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        elia = EliaData()
        df = elia.get_data("2024-06-15", "2024-06-15")
        assert df is None


def test_get_data_api_error():
    """Should return None on API errors."""
    mock_resp = Mock()
    import requests as req_lib

    mock_resp.raise_for_status.side_effect = req_lib.exceptions.HTTPError(
        "API Error"
    )

    with patch("requests.get", return_value=mock_resp):
        elia = EliaData()
        df = elia.get_data("2024-06-15", "2024-06-15")
        assert df is None


def test_get_data_pagination():
    """Should auto-paginate through all available data."""
    page1 = {
        "results": [
            {"datetime": "2024-06-15T12:00:00+00:00", "measured": 2500.0},
            {"datetime": "2024-06-15T12:15:00+00:00", "measured": 2520.0},
        ]
    }
    page2 = {
        "results": [
            {"datetime": "2024-06-15T12:30:00+00:00", "measured": 2510.0},
        ]
    }

    mock_resp1 = Mock()
    mock_resp1.json.return_value = page1
    mock_resp1.raise_for_status.return_value = None

    mock_resp2 = Mock()
    mock_resp2.json.return_value = page2
    mock_resp2.raise_for_status.return_value = None

    with patch(
        "requests.get", side_effect=[mock_resp1, mock_resp2]
    ) as mock_get:
        elia = EliaData()

        df = elia.get_data("2024-06-15", "2024-06-15", limit=2)

        assert len(df) == 3
        assert mock_get.call_count == 2

        call_args_list = mock_get.call_args_list
        assert call_args_list[0][1]["params"]["offset"] == 0
        assert call_args_list[1][1]["params"]["offset"] == 2


def test_get_dataset_success(mock_response):
    """Should return an xarray Dataset with datetime_utc index."""
    import xarray as xr

    with patch("requests.get", return_value=mock_response):
        elia = EliaData()

        ds = elia.get_dataset(
            start_date="2024-06-15",
            end_date="2024-06-15",
        )

        assert isinstance(ds, xr.Dataset)
        assert "datetime_utc" in ds.coords or "datetime_utc" in ds.indexes
        assert "measured" in ds.data_vars
        assert len(ds.datetime_utc) == 2


def test_get_dataset_empty():
    """Should return None when no data is available."""
    mock_resp = Mock()
    mock_resp.json.return_value = {"results": []}
    mock_resp.raise_for_status.return_value = None

    with patch("requests.get", return_value=mock_resp):
        elia = EliaData()
        ds = elia.get_dataset("2024-06-15", "2024-06-15")
        assert ds is None


def test_get_data_date_filtering(mock_response):
    """Should pass correct date range in the API where clause."""
    with patch("requests.get", return_value=mock_response) as mock_get:
        elia = EliaData()
        elia.get_data("2024-06-15", "2024-06-16")

        _, kwargs = mock_get.call_args
        where = kwargs["params"]["where"]
        assert "2024-06-15" in where
        assert "2024-06-16" in where