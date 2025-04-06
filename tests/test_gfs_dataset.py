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
    # Create a sample dataset
    lats = np.linspace(45, 65, 21)
    lons = np.concatenate([np.linspace(350, 360, 11), np.linspace(0, 10, 11)])
    data = np.random.rand(24, 10, 5, len(lats), len(lons))

    ds = xr.Dataset(
        data_vars={
            "temperature": (("init_time_utc", "step", "channel", "latitude", "longitude"), data)
        },
        coords={
            "init_time_utc": pd.date_range("2024-01-01", periods=24, freq="H"),
            "step": range(10),
            "channel": range(5),
            "latitude": lats,
            "longitude": lons,
        },
    )

    result = preprocess_gfs_data(ds)

    assert all(
        dim in result.dims for dim in ["init_time_utc", "step", "channel", "latitude", "longitude"]
    )
    assert result.latitude.min() >= 45
    assert result.latitude.max() <= 65


@patch("fsspec.get_mapper")
@patch("xarray.open_dataset")
def test_open_gfs(mock_open_dataset, mock_get_mapper):
    # Mock the dataset
    mock_ds = MagicMock()
    mock_open_dataset.return_value = mock_ds
    mock_ds.dims = ["init_time_utc", "step", "channel", "latitude", "longitude"]

    result = open_gfs("fake_path")

    mock_get_mapper.assert_called_once_with("fake_path", anon=True)
    mock_open_dataset.assert_called_once()
    assert result is not None


def test_handle_nan_values_invalid_method():
    data = np.array([[1.0, np.nan], [np.nan, 4.0]])
    da = xr.DataArray(data, dims=["latitude", "longitude"])

    with pytest.raises(ValueError, match="Invalid method for handling NaNs"):
        handle_nan_values(da, method="invalid")


def test_gfs_data_sampler():
    # Mock dataset and config for testing
    data = np.random.rand(24, 10, 5, 21, 22)  # Example dimensions
    da = xr.DataArray(
        data,
        dims=["init_time_utc", "step", "channel", "latitude", "longitude"],
        coords={
            "init_time_utc": pd.date_range("2024-01-01", periods=24, freq="H"),
            "step": range(10),
            "channel": range(5),
            "latitude": np.linspace(45, 65, 21),
            "longitude": np.linspace(350, 10, 22),
        },
    )

    with pytest.raises(FileNotFoundError):
        # Should raise error with invalid config file
        GFSDataSampler(da, "nonexistent_config.yaml")
