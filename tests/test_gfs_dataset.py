"""
Tests for GFS dataset functionality.

These tests cover only core guarantees and avoid integration complexity.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import MagicMock, patch

from open_data_pvnet.nwp.gfs_dataset import (
    open_gfs,
    handle_nan_values,
    GFSDataSampler,
)


@pytest.fixture
def sample_gfs_data():
    """Minimal GFS-like DataArray with a NaN value."""
    data = np.array([[[[[np.nan]]]]])
    return xr.DataArray(
        data,
        dims=["init_time_utc", "step", "channel", "latitude", "longitude"],
        coords={
            "init_time_utc": [pd.Timestamp("2023-01-01")],
            "step": [np.timedelta64(0, "h")],
            "channel": ["t2m"],
            "latitude": [50.0],
            "longitude": [0.0],
        },
    )


@pytest.fixture
def sample_gfs_data_with_init_time():
    """DataArray using `init_time` to test renaming."""
    data = np.zeros((1, 1, 1, 1, 1))
    return xr.DataArray(
        data,
        dims=["init_time", "step", "channel", "latitude", "longitude"],
        coords={
            "init_time": [pd.Timestamp("2023-01-01")],
            "step": [np.timedelta64(0, "h")],
            "channel": ["t2m"],
            "latitude": [50.0],
            "longitude": [0.0],
        },
    )


@pytest.fixture
def mock_config():
    """Minimal config object for sampler initialization."""
    config = MagicMock()
    config.input_data.nwp.gfs.interval_start_minutes = 0
    config.input_data.nwp.gfs.interval_end_minutes = 60
    config.input_data.nwp.gfs.time_resolution_minutes = 60
    config.input_data.nwp.gfs.provider = "gfs"
    return config


class TestHandleNanValues:
    def test_fill_nan(self, sample_gfs_data):
        """NaN values should be filled correctly."""
        result = handle_nan_values(sample_gfs_data, method="fill", fill_value=0.0)
        assert not np.isnan(result.values).any()
        assert result.values[0, 0, 0, 0, 0] == 0.0


class TestOpenGfs:
    def test_renames_init_time(self, sample_gfs_data_with_init_time):
        """`init_time` should be renamed to `init_time_utc`."""
        mock_dataset = MagicMock()
        mock_dataset.to_array.return_value = sample_gfs_data_with_init_time

        with patch("open_data_pvnet.nwp.gfs_dataset.fsspec.get_mapper"), \
             patch("open_data_pvnet.nwp.gfs_dataset.xr.open_dataset", return_value=mock_dataset):

            result = open_gfs("s3://dummy/gfs.zarr")

            assert "init_time_utc" in result.dims
            assert "init_time" not in result.dims


class TestGFSDataSampler:
    def test_sampler_initialization(self, sample_gfs_data, mock_config):
        """Sampler should initialize with valid times."""
        valid_times = pd.DataFrame(
            {"t0": [pd.Timestamp("2023-01-01")]}
        )

        with patch(
            "open_data_pvnet.nwp.gfs_dataset.load_yaml_configuration",
            return_value=mock_config,
        ), patch(
            "open_data_pvnet.nwp.gfs_dataset.find_valid_time_periods",
            return_value=valid_times,
        ):
            sampler = GFSDataSampler(
                dataset=sample_gfs_data,
                config_filename="dummy.yaml",
            )

            assert len(sampler) == 1


class TestNormalizationFallback:
    def test_no_crash_without_nwp_stats(self, sample_gfs_data, mock_config):
        """Sampler should not crash if NWP stats are unavailable."""
        valid_times = pd.DataFrame(
            {"t0": [pd.Timestamp("2023-01-01")]}
        )

        with patch(
            "open_data_pvnet.nwp.gfs_dataset.load_yaml_configuration",
            return_value=mock_config,
        ), patch(
            "open_data_pvnet.nwp.gfs_dataset.find_valid_time_periods",
            return_value=valid_times,
        ), patch(
            "open_data_pvnet.nwp.gfs_dataset.HAS_NWP_STATS",
            False,
        ):
            sampler = GFSDataSampler(
                dataset=sample_gfs_data,
                config_filename="dummy.yaml",
            )

            sample = sampler[0]
            assert sample is not None
