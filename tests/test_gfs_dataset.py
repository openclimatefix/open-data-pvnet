import pytest
import xarray as xr
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from open_data_pvnet.nwp.gfs_dataset import (
    preprocess_gfs_data,
    open_gfs,
    handle_nan_values,
    GFSDataSampler,
)


def test_preprocess_gfs_data():
    # Create test data with correct longitude range (including 360)
    test_data = xr.Dataset(
        {
            "variable": (["time", "latitude", "longitude"], np.random.rand(24, 10, 37)),
        },
        coords={
            "init_time_utc": pd.date_range("2024-01-01", periods=24, freq="h"),
            "latitude": np.linspace(50, 60, 10),
            "longitude": np.linspace(340, 360, 37),  # Include values up to 360
        },
    )

    result = preprocess_gfs_data(test_data)
    assert isinstance(result, xr.Dataset)


@patch("fsspec.get_mapper")
@patch("xarray.open_dataset")
def test_open_gfs(mock_open_dataset, mock_get_mapper):
    # Create a proper mock dataset with all required dimensions
    mock_dataset = xr.Dataset(
        {
            "variable": (
                ["init_time_utc", "step", "channel", "latitude", "longitude"],
                np.random.rand(24, 10, 5, 21, 37),
            ),
        },
        coords={
            "init_time_utc": pd.date_range("2024-01-01", periods=24, freq="h"),
            "step": range(10),
            "channel": range(5),
            "latitude": np.linspace(50, 60, 21),
            "longitude": np.linspace(340, 360, 37),
        },
    )
    mock_open_dataset.return_value = mock_dataset

    result = open_gfs("fake_path")
    assert isinstance(result, xr.Dataset)

    # Verify all required dimensions are present
    expected_dims = ["init_time_utc", "step", "channel", "latitude", "longitude"]
    for dim in expected_dims:
        assert dim in result.dims


def test_handle_nan_values_invalid_method():
    data = np.array([[1.0, np.nan], [np.nan, 4.0]])
    da = xr.DataArray(data, dims=["latitude", "longitude"])

    with pytest.raises(ValueError, match="Invalid method for handling NaNs"):
        handle_nan_values(da, method="invalid")


@pytest.fixture
def mock_gfs_dataset():
    # Create sample data with the correct channels
    channels = ["dlwrf", "dswrf", "hcc"]
    data = np.random.rand(24, 10, len(channels), 21, 22)  # Example dimensions

    return xr.DataArray(
        data,
        dims=["init_time_utc", "step", "channel", "latitude", "longitude"],
        coords={
            "init_time_utc": pd.date_range("2024-01-01", periods=24, freq="h"),
            "step": np.array([np.timedelta64(i, "h") for i in range(0, 30, 3)]),
            "channel": channels,
            "latitude": np.linspace(45, 65, 21),
            "longitude": np.linspace(350, 10, 22),
        },
    )


@pytest.fixture
def mock_config(tmp_path):
    config_path = tmp_path / "test_config.yaml"
    config_content = """
input_data:
  nwp:
    gfs:
      time_resolution_minutes: 180
      interval_start_minutes: 0
      interval_end_minutes: 1080
      dropout_timedeltas_minutes: []  # Changed from null to empty list
      accum_channels: []
      max_staleness_minutes: 1080
      zarr_path: "s3://ocf-open-data-pvnet/data/gfs.zarr"
      provider: "gfs"
      image_size_pixels_height: 1
      image_size_pixels_width: 1
      channels:
        - "dlwrf"
        - "dswrf"
        - "hcc"
      normalisation_constants:  # Added required field
        dlwrf: {"mean": 0.0, "std": 1.0}
        dswrf: {"mean": 0.0, "std": 1.0}
        hcc: {"mean": 0.0, "std": 1.0}
      area:  # Replace lat_range and lon_range with area
        lat_min: 46
        lat_max: 64
        lon_min: 351
        lon_max: 9
"""
    config_path.write_text(config_content)
    return str(config_path)


@pytest.mark.xfail(reason="Configuration validation needs to be updated to match schema")
def test_gfs_data_sampler_initialization(mock_gfs_dataset, mock_config):
    """Test successful initialization and basic sampling of GFSDataSampler."""
    sampler = GFSDataSampler(mock_gfs_dataset, mock_config)

    # Test that initialization worked and data was filtered correctly
    assert isinstance(sampler.data, xr.DataArray)

    # Check if coordinates are within specified ranges
    assert sampler.data.latitude.min() >= 46
    assert sampler.data.latitude.max() <= 64
    assert sampler.data.longitude.min() >= 351 or sampler.data.longitude.max() <= 9
    assert len(sampler.data.channel) == 3  # We specified 3 channels in the config

    # Test that the data has the expected dimensions
    expected_dims = ["init_time_utc", "step", "channel", "latitude", "longitude"]
    assert all(dim in sampler.data.dims for dim in expected_dims)

    # Test time range
    assert pd.Timestamp(sampler.data.init_time_utc.min()) >= pd.Timestamp("2024-01-01")
    assert pd.Timestamp(sampler.data.init_time_utc.max()) <= pd.Timestamp("2024-01-02")
