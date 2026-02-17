import pytest
import pandas as pd
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch
from open_data_pvnet.scripts.preprocess_eia_data import EIAPreprocessor, US_RTO_LOCATIONS


@pytest.fixture
def mock_eia_response():
    """Mock EIA API response data."""
    return pd.DataFrame({
        "period": [
            "2023-01-01T00", "2023-01-01T01", "2023-01-01T02",
            "2023-01-01T03", "2023-01-01T04", "2023-01-01T05"
        ],
        "value": [0, 50, 150, 300, 250, 100],
        "fueltype": ["SUN"] * 6,
        "respondent": ["CAISO"] * 6,
    })


@pytest.fixture
def preprocessor():
    """Create EIAPreprocessor instance."""
    return EIAPreprocessor(api_key="test_key")


def test_preprocessor_init():
    """Test initialization."""
    preprocessor = EIAPreprocessor(api_key="test_key")
    assert preprocessor.eia_data.api_key == "test_key"
    assert preprocessor.location_metadata == US_RTO_LOCATIONS


def test_transform_to_schema(preprocessor, mock_eia_response):
    """Test schema transformation."""
    ds = preprocessor.transform_to_schema(mock_eia_response, "CAISO")
    
    assert "time_utc" in ds.dims
    assert "location_id" in ds.dims
    assert "generation_mw" in ds.data_vars
    assert "longitude" in ds.coords
    assert "latitude" in ds.coords
    assert "location_id" in ds.coords
    
    assert ds.coords["location_id"].values[0] == US_RTO_LOCATIONS["CAISO"]["location_id"]
    assert ds.coords["longitude"].values[0] == US_RTO_LOCATIONS["CAISO"]["longitude"]
    assert ds.coords["latitude"].values[0] == US_RTO_LOCATIONS["CAISO"]["latitude"]
    
    assert len(ds.time_utc) == 6
    assert ds["generation_mw"].dtype == np.float32


def test_transform_to_schema_with_datetime_gmt(preprocessor):
    """Test with datetime_gmt column instead of period."""
    df = pd.DataFrame({
        "datetime_gmt": pd.to_datetime(["2023-01-01T00", "2023-01-01T01"], utc=True),
        "value": [100, 150],
    })
    
    ds = preprocessor.transform_to_schema(df, "ERCOT")
    
    assert "time_utc" in ds.dims
    assert len(ds.time_utc) == 2


def test_transform_to_schema_unknown_region(preprocessor, mock_eia_response):
    """Test unknown region raises error."""
    with pytest.raises(ValueError, match="Unknown region"):
        preprocessor.transform_to_schema(mock_eia_response, "UNKNOWN_REGION")


def test_transform_to_schema_missing_time_column(preprocessor):
    """Test missing time column raises error."""
    df = pd.DataFrame({"value": [100, 150]})
    
    with pytest.raises(ValueError, match="No time column found"):
        preprocessor.transform_to_schema(df, "CAISO")


def test_transform_to_schema_missing_value_column(preprocessor):
    """Test missing value column raises error."""
    df = pd.DataFrame({"period": ["2023-01-01T00", "2023-01-01T01"]})
    
    with pytest.raises(ValueError, match="No 'value' column found"):
        preprocessor.transform_to_schema(df, "CAISO")


def test_estimate_capacity(preprocessor, mock_eia_response):
    """Test capacity estimation."""
    ds = preprocessor.transform_to_schema(mock_eia_response, "CAISO")
    ds_with_capacity = preprocessor.estimate_capacity(ds, percentile=99.0)
    
    assert "capacity_mwp" in ds_with_capacity.data_vars
    
    expected_capacity = np.percentile(mock_eia_response["value"].values, 99.0)
    actual_capacity = ds_with_capacity["capacity_mwp"].isel(location_id=0, time_utc=0).values
    assert np.isclose(actual_capacity, expected_capacity, rtol=0.01)
    assert ds_with_capacity["capacity_mwp"].dtype == np.float32


def test_estimate_capacity_with_max_percentile(preprocessor, mock_eia_response):
    """Test capacity with 100th percentile."""
    ds = preprocessor.transform_to_schema(mock_eia_response, "CAISO")
    ds_with_capacity = preprocessor.estimate_capacity(ds, percentile=100.0)
    
    expected_capacity = mock_eia_response["value"].max()
    actual_capacity = ds_with_capacity["capacity_mwp"].isel(location_id=0, time_utc=0).values
    assert np.isclose(actual_capacity, expected_capacity, rtol=0.01)


def test_validate_data_success(preprocessor, mock_eia_response):
    """Test validation passes."""
    ds = preprocessor.transform_to_schema(mock_eia_response, "CAISO")
    ds = preprocessor.estimate_capacity(ds)
    
    assert preprocessor.validate_data(ds) is True


def test_validate_data_missing_dimension(preprocessor, mock_eia_response):
    """Test validation fails with missing dimension."""
    ds = preprocessor.transform_to_schema(mock_eia_response, "CAISO")
    ds = preprocessor.estimate_capacity(ds)
    
    ds_invalid = ds.isel(location_id=0, drop=True)
    assert preprocessor.validate_data(ds_invalid) is False


def test_validate_data_missing_variable(preprocessor, mock_eia_response):
    """Test validation fails with missing variable."""
    ds = preprocessor.transform_to_schema(mock_eia_response, "CAISO")
    assert preprocessor.validate_data(ds) is False


def test_validate_data_missing_coordinate(preprocessor, mock_eia_response):
    """Test validation fails with missing coordinate."""
    ds = preprocessor.transform_to_schema(mock_eia_response, "CAISO")
    ds = preprocessor.estimate_capacity(ds)
    
    ds_invalid = ds.drop_vars("longitude")
    assert preprocessor.validate_data(ds_invalid) is False


def test_save_to_zarr(preprocessor, mock_eia_response, tmp_path):
    """Test saving to Zarr."""
    ds = preprocessor.transform_to_schema(mock_eia_response, "CAISO")
    ds = preprocessor.estimate_capacity(ds)
    
    output_path = tmp_path / "test_output.zarr"
    preprocessor.save_to_zarr(ds, str(output_path))
    
    assert output_path.exists()
    
    ds_loaded = xr.open_zarr(output_path)
    assert "generation_mw" in ds_loaded.data_vars
    assert "capacity_mwp" in ds_loaded.data_vars
    assert len(ds_loaded.time_utc) == 6


def test_fetch_and_preprocess_single_region(preprocessor, mock_eia_response, tmp_path):
    """Test full pipeline for single region."""
    with patch.object(preprocessor.eia_data, 'get_data', return_value=mock_eia_response):
        output_path = tmp_path / "output.zarr"
        
        ds = preprocessor.fetch_and_preprocess(
            start_date="2023-01-01",
            end_date="2023-01-02",
            regions=["CAISO"],
            output_path=str(output_path),
        )
        
        assert "generation_mw" in ds.data_vars
        assert "capacity_mwp" in ds.data_vars
        assert "time_utc" in ds.dims
        assert "location_id" in ds.dims
        assert output_path.exists()


def test_fetch_and_preprocess_multiple_regions(preprocessor, mock_eia_response, tmp_path):
    """Test pipeline for multiple regions."""
    mock_caiso = mock_eia_response.copy()
    mock_ercot = mock_eia_response.copy()
    mock_ercot["value"] = mock_ercot["value"] * 1.5
    
    with patch.object(preprocessor.eia_data, 'get_data', side_effect=[mock_caiso, mock_ercot]):
        ds = preprocessor.fetch_and_preprocess(
            start_date="2023-01-01",
            end_date="2023-01-02",
            regions=["CAISO", "ERCOT"],
        )
        
        assert len(ds.location_id) == 2
        assert US_RTO_LOCATIONS["CAISO"]["location_id"] in ds.location_id.values
        assert US_RTO_LOCATIONS["ERCOT"]["location_id"] in ds.location_id.values


def test_fetch_and_preprocess_us48_default(preprocessor, mock_eia_response):
    """Test US48 is default region."""
    with patch.object(preprocessor.eia_data, 'get_data', return_value=mock_eia_response) as mock_get:
        ds = preprocessor.fetch_and_preprocess(
            start_date="2023-01-01",
            end_date="2023-01-02",
            regions=None,
        )
        
        call_args = mock_get.call_args
        assert call_args[1]["region"] == "US48"
        assert ds.location_id.values[0] == US_RTO_LOCATIONS["US48"]["location_id"]


def test_fetch_and_preprocess_no_data(preprocessor):
    """Test handling when no data retrieved."""
    with patch.object(preprocessor.eia_data, 'get_data', return_value=None):
        with pytest.raises(ValueError, match="No data retrieved"):
            preprocessor.fetch_and_preprocess(
                start_date="2023-01-01",
                end_date="2023-01-02",
                regions=["CAISO"],
            )


def test_fetch_and_preprocess_empty_dataframe(preprocessor):
    """Test handling empty DataFrame."""
    empty_df = pd.DataFrame()
    
    with patch.object(preprocessor.eia_data, 'get_data', return_value=empty_df):
        with pytest.raises(ValueError, match="No data retrieved"):
            preprocessor.fetch_and_preprocess(
                start_date="2023-01-01",
                end_date="2023-01-02",
                regions=["CAISO"],
            )


def test_fetch_and_preprocess_validation_failure(preprocessor, mock_eia_response):
    """Test validation failure raises error."""
    with patch.object(preprocessor.eia_data, 'get_data', return_value=mock_eia_response):
        with patch.object(preprocessor, 'validate_data', return_value=False):
            with pytest.raises(ValueError, match="Data validation failed"):
                preprocessor.fetch_and_preprocess(
                    start_date="2023-01-01",
                    end_date="2023-01-02",
                    regions=["CAISO"],
                )


def test_us_rto_locations_structure():
    """Test location metadata structure."""
    for region, info in US_RTO_LOCATIONS.items():
        assert "location_id" in info
        assert "latitude" in info
        assert "longitude" in info
        assert "name" in info
        
        assert isinstance(info["location_id"], int)
        assert isinstance(info["latitude"], (int, float))
        assert isinstance(info["longitude"], (int, float))
        assert isinstance(info["name"], str)
        
        assert -90 <= info["latitude"] <= 90
        assert -180 <= info["longitude"] <= 180


def test_us_rto_locations_unique_ids():
    """Test location IDs are unique."""
    location_ids = [info["location_id"] for info in US_RTO_LOCATIONS.values()]
    assert len(location_ids) == len(set(location_ids)), "Location IDs must be unique"
