"""Tests for GFS dataset module (Issue #120 - Code Coverage).

Covers:
- open_gfs(): Opening and preparing GFS zarr datasets
- handle_nan_values(): NaN handling strategies (fill/drop)
- GFSDataSampler: PyTorch Dataset for GFS data sampling and normalization
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from unittest.mock import MagicMock, patch

from open_data_pvnet.nwp.gfs_dataset import (
    GFSDataSampler,
    handle_nan_values,
    open_gfs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataarray():
    """Create a minimal xarray DataArray mimicking GFS structure."""
    init_times = pd.date_range("2023-01-01", periods=3, freq="6h")
    steps = [np.timedelta64(h, "h") for h in [0, 3, 6, 9, 12, 18]]
    channels = ["t", "dswrf", "tcc"]
    lats = np.array([40.0, 41.0, 42.0])
    lons = np.array([-100.0, -99.0, -98.0])

    data = np.random.rand(
        len(init_times), len(steps), len(channels), len(lats), len(lons)
    )
    da = xr.DataArray(
        data,
        dims=["init_time_utc", "step", "channel", "latitude", "longitude"],
        coords={
            "init_time_utc": init_times,
            "step": steps,
            "channel": channels,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return da


@pytest.fixture
def sample_dataarray_with_nans(sample_dataarray):
    """DataArray with NaN values injected."""
    da = sample_dataarray.copy(deep=True)
    da.values[0, 0, 0, :, :] = np.nan
    da.values[1, 1, 1, 0, :] = np.nan
    return da


@pytest.fixture
def mock_config():
    """Mock configuration object matching gfs_data_config.yaml structure."""
    config = MagicMock()
    nwp_gfs = config.input_data.nwp.gfs
    nwp_gfs.interval_start_minutes = 0
    nwp_gfs.interval_end_minutes = 1080
    nwp_gfs.time_resolution_minutes = 180
    nwp_gfs.provider = "gfs"
    nwp_gfs.channels = ["t", "dswrf", "tcc"]
    return config


@pytest.fixture
def valid_t0_df():
    """DataFrame of valid initialization times."""
    return pd.DataFrame(
        {"start_dt": pd.date_range("2023-01-01", periods=3, freq="6h")}
    )


# ---------------------------------------------------------------------------
# Tests: open_gfs()
# ---------------------------------------------------------------------------


class TestOpenGfs:
    """Tests for the open_gfs function."""

    def test_open_gfs_success(self):
        """open_gfs should open zarr, convert to DataArray, rename dims."""
        init_times = pd.date_range("2023-01-01", periods=2, freq="6h")
        steps = [np.timedelta64(0, "h"), np.timedelta64(3, "h")]
        channels = ["t", "dswrf"]
        lats = [40.0, 41.0]
        lons = [-100.0, -99.0]

        # Build a Dataset that .to_array() can convert
        ds = xr.Dataset(
            {
                "t": (["init_time", "step", "latitude", "longitude"],
                      np.random.rand(2, 2, 2, 2)),
                "dswrf": (["init_time", "step", "latitude", "longitude"],
                          np.random.rand(2, 2, 2, 2)),
            },
            coords={
                "init_time": init_times,
                "step": steps,
                "latitude": lats,
                "longitude": lons,
            },
        )

        with patch("open_data_pvnet.nwp.gfs_dataset.fsspec.get_mapper") as mock_mapper, \
             patch("open_data_pvnet.nwp.gfs_dataset.xr.open_dataset", return_value=ds):

            mock_mapper.return_value = MagicMock()
            result = open_gfs("s3://fake-bucket/gfs.zarr")

            # Should rename init_time -> init_time_utc
            assert "init_time_utc" in result.dims
            assert "init_time" not in result.dims

            # Should have all required dims
            expected_dims = ["init_time_utc", "step", "channel", "latitude", "longitude"]
            assert list(result.dims) == expected_dims

    def test_open_gfs_no_rename_needed(self):
        """open_gfs should skip renaming when init_time_utc already exists."""
        init_times = pd.date_range("2023-01-01", periods=2, freq="6h")
        steps = [np.timedelta64(0, "h"), np.timedelta64(3, "h")]

        ds = xr.Dataset(
            {
                "t": (["init_time_utc", "step", "latitude", "longitude"],
                      np.random.rand(2, 2, 2, 2)),
            },
            coords={
                "init_time_utc": init_times,
                "step": steps,
                "latitude": [40.0, 41.0],
                "longitude": [-100.0, -99.0],
            },
        )

        with patch("open_data_pvnet.nwp.gfs_dataset.fsspec.get_mapper") as mock_mapper, \
             patch("open_data_pvnet.nwp.gfs_dataset.xr.open_dataset", return_value=ds):

            mock_mapper.return_value = MagicMock()
            result = open_gfs("s3://fake-bucket/gfs.zarr")

            assert "init_time_utc" in result.dims

    def test_open_gfs_calls_fsspec(self):
        """open_gfs should call fsspec.get_mapper with correct args."""
        ds = xr.Dataset(
            {"t": (["init_time_utc", "step", "latitude", "longitude"],
                   np.random.rand(1, 1, 1, 1))},
            coords={
                "init_time_utc": pd.date_range("2023-01-01", periods=1),
                "step": [np.timedelta64(0, "h")],
                "latitude": [40.0],
                "longitude": [-100.0],
            },
        )

        with patch("open_data_pvnet.nwp.gfs_dataset.fsspec.get_mapper") as mock_mapper, \
             patch("open_data_pvnet.nwp.gfs_dataset.xr.open_dataset", return_value=ds):

            mock_mapper.return_value = MagicMock()
            open_gfs("s3://my-bucket/data.zarr")
            mock_mapper.assert_called_once_with("s3://my-bucket/data.zarr", anon=True)


# ---------------------------------------------------------------------------
# Tests: handle_nan_values()
# ---------------------------------------------------------------------------


class TestHandleNanValues:
    """Tests for the handle_nan_values function."""

    def test_fill_replaces_nans(self, sample_dataarray_with_nans):
        """method='fill' should replace all NaN values with fill_value."""
        result = handle_nan_values(sample_dataarray_with_nans, method="fill", fill_value=0.0)
        assert not np.isnan(result.values).any(), "NaN values remain after fill"

    def test_fill_uses_custom_value(self, sample_dataarray_with_nans):
        """method='fill' should use the provided fill_value."""
        result = handle_nan_values(sample_dataarray_with_nans, method="fill", fill_value=-999.0)
        # The NaN positions should now be -999.0
        assert (result.values[0, 0, 0, :, :] == -999.0).all()

    def test_fill_default_value_is_zero(self, sample_dataarray_with_nans):
        """Default fill_value should be 0.0."""
        result = handle_nan_values(sample_dataarray_with_nans, method="fill")
        assert (result.values[0, 0, 0, :, :] == 0.0).all()

    def test_drop_removes_nan_rows(self):
        """method='drop' should drop latitude/longitude slices that are all NaN."""
        da = xr.DataArray(
            np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]]),
            dims=["latitude", "longitude"],
            coords={"latitude": [40.0, 41.0, 42.0], "longitude": [-100.0, -99.0]},
        )
        result = handle_nan_values(da, method="drop")
        assert 41.0 not in result.latitude.values

    def test_invalid_method_raises_value_error(self, sample_dataarray):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            handle_nan_values(sample_dataarray, method="interpolate")


# ---------------------------------------------------------------------------
# Tests: GFSDataSampler
# ---------------------------------------------------------------------------


class TestGFSDataSampler:
    """Tests for the GFSDataSampler PyTorch Dataset."""

    def _create_sampler(self, sample_dataarray, mock_config, valid_t0_df,
                        start_time=None, end_time=None):
        """Helper to create a GFSDataSampler with mocked dependencies."""
        with patch(
            "open_data_pvnet.nwp.gfs_dataset.load_yaml_configuration",
            return_value=mock_config,
        ), patch(
            "open_data_pvnet.nwp.gfs_dataset.find_valid_time_periods",
            return_value=valid_t0_df,
        ):
            return GFSDataSampler(
                dataset=sample_dataarray,
                config_filename="fake_config.yaml",
                start_time=start_time,
                end_time=end_time,
            )

    def test_init_loads_config(self, sample_dataarray, mock_config, valid_t0_df):
        """__init__ should load config and find valid time periods."""
        sampler = self._create_sampler(sample_dataarray, mock_config, valid_t0_df)
        assert sampler.dataset is sample_dataarray
        assert sampler.config is mock_config

    def test_init_renames_start_dt_column(self, sample_dataarray, mock_config, valid_t0_df):
        """__init__ should rename 'start_dt' column to 't0'."""
        sampler = self._create_sampler(sample_dataarray, mock_config, valid_t0_df)
        assert "t0" in sampler.valid_t0_times.columns
        assert "start_dt" not in sampler.valid_t0_times.columns

    def test_init_filters_start_time(self, sample_dataarray, mock_config, valid_t0_df):
        """__init__ should filter times >= start_time."""
        sampler = self._create_sampler(
            sample_dataarray, mock_config, valid_t0_df,
            start_time="2023-01-01T06:00:00",
        )
        assert all(sampler.valid_t0_times["t0"] >= pd.Timestamp("2023-01-01T06:00:00"))

    def test_init_filters_end_time(self, sample_dataarray, mock_config, valid_t0_df):
        """__init__ should filter times <= end_time."""
        sampler = self._create_sampler(
            sample_dataarray, mock_config, valid_t0_df,
            end_time="2023-01-01T06:00:00",
        )
        assert all(sampler.valid_t0_times["t0"] <= pd.Timestamp("2023-01-01T06:00:00"))

    def test_init_filters_both_times(self, sample_dataarray, mock_config, valid_t0_df):
        """__init__ should filter between start_time and end_time."""
        sampler = self._create_sampler(
            sample_dataarray, mock_config, valid_t0_df,
            start_time="2023-01-01T00:00:00",
            end_time="2023-01-01T06:00:00",
        )
        t0s = sampler.valid_t0_times["t0"]
        assert all(t0s >= pd.Timestamp("2023-01-01T00:00:00"))
        assert all(t0s <= pd.Timestamp("2023-01-01T06:00:00"))

    def test_len_returns_valid_count(self, sample_dataarray, mock_config, valid_t0_df):
        """__len__ should return number of valid t0 times."""
        sampler = self._create_sampler(sample_dataarray, mock_config, valid_t0_df)
        assert len(sampler) == len(valid_t0_df)

    def test_getitem_returns_sample(self, sample_dataarray, mock_config, valid_t0_df):
        """__getitem__ should return a normalized xr.Dataset via _get_sample."""
        sampler = self._create_sampler(sample_dataarray, mock_config, valid_t0_df)

        # Mock _get_sample to avoid needing full normalization pipeline
        mock_sample = MagicMock()
        with patch.object(sampler, "_get_sample", return_value=mock_sample):
            result = sampler[0]
            assert result is mock_sample
            sampler._get_sample.assert_called_once()


class TestGFSDataSamplerGetSample:
    """Tests for _get_sample method."""

    def _create_sampler(self, sample_dataarray, mock_config, valid_t0_df):
        """Helper to create sampler."""
        with patch(
            "open_data_pvnet.nwp.gfs_dataset.load_yaml_configuration",
            return_value=mock_config,
        ), patch(
            "open_data_pvnet.nwp.gfs_dataset.find_valid_time_periods",
            return_value=valid_t0_df,
        ):
            return GFSDataSampler(
                dataset=sample_dataarray,
                config_filename="fake_config.yaml",
            )

    def test_get_sample_slices_data(self, sample_dataarray, mock_config, valid_t0_df):
        """_get_sample should slice data by init_time and valid steps."""
        sampler = self._create_sampler(sample_dataarray, mock_config, valid_t0_df)
        t0 = pd.Timestamp("2023-01-01")

        # Mock normalization to return input unchanged
        with patch.object(sampler, "_normalize_sample", side_effect=lambda x: x):
            result = sampler._get_sample(t0)
            # Result should be sliced for the init_time
            assert "init_time_utc" not in result.dims  # selected single time

    def test_get_sample_raises_on_no_valid_steps(self, mock_config, valid_t0_df):
        """_get_sample should raise ValueError if no valid steps found."""
        # Create a DataArray whose steps don't match the config intervals
        init_times = pd.date_range("2023-01-01", periods=1, freq="6h")
        # Only step = 999h which won't match config's 0-1080 min range
        steps = [np.timedelta64(999, "h")]
        da = xr.DataArray(
            np.random.rand(1, 1, 1, 1, 1),
            dims=["init_time_utc", "step", "channel", "latitude", "longitude"],
            coords={
                "init_time_utc": init_times,
                "step": steps,
                "channel": ["t"],
                "latitude": [40.0],
                "longitude": [-100.0],
            },
        )

        sampler = None
        with patch(
            "open_data_pvnet.nwp.gfs_dataset.load_yaml_configuration",
            return_value=mock_config,
        ), patch(
            "open_data_pvnet.nwp.gfs_dataset.find_valid_time_periods",
            return_value=valid_t0_df,
        ):
            sampler = GFSDataSampler(dataset=da, config_filename="fake.yaml")

        with pytest.raises(ValueError, match="No valid steps found"):
            sampler._get_sample(pd.Timestamp("2023-01-01"))


class TestGFSDataSamplerNormalize:
    """Tests for _normalize_sample method."""

    def _create_sampler(self, sample_dataarray, mock_config, valid_t0_df):
        """Helper to create sampler."""
        with patch(
            "open_data_pvnet.nwp.gfs_dataset.load_yaml_configuration",
            return_value=mock_config,
        ), patch(
            "open_data_pvnet.nwp.gfs_dataset.find_valid_time_periods",
            return_value=valid_t0_df,
        ):
            return GFSDataSampler(
                dataset=sample_dataarray,
                config_filename="fake_config.yaml",
            )

    def test_normalize_applies_formula(self, sample_dataarray, mock_config, valid_t0_df):
        """_normalize_sample should compute (data - mean) / std."""
        sampler = self._create_sampler(sample_dataarray, mock_config, valid_t0_df)

        channels = ["t", "dswrf", "tcc"]
        mock_means = xr.DataArray([10.0, 200.0, 50.0], dims=["channel"],
                                  coords={"channel": channels})
        mock_stds = xr.DataArray([5.0, 100.0, 25.0], dims=["channel"],
                                 coords={"channel": channels})

        # Create a simple dataset slice to normalize
        data = xr.DataArray(
            np.array([[[15.0]], [[300.0]], [[75.0]]]),
            dims=["channel", "latitude", "longitude"],
            coords={"channel": channels, "latitude": [40.0], "longitude": [-100.0]},
        )

        with patch.dict(
            "open_data_pvnet.nwp.gfs_dataset.NWP_MEANS", {"gfs": mock_means}
        ), patch.dict(
            "open_data_pvnet.nwp.gfs_dataset.NWP_STDS", {"gfs": mock_stds}
        ):
            result = sampler._normalize_sample(data)

            # (15 - 10) / 5 = 1.0
            assert np.isclose(result.sel(channel="t").values, 1.0).all()
            # (300 - 200) / 100 = 1.0
            assert np.isclose(result.sel(channel="dswrf").values, 1.0).all()
            # (75 - 50) / 25 = 1.0
            assert np.isclose(result.sel(channel="tcc").values, 1.0).all()

    def test_normalize_handles_missing_channels(self, sample_dataarray, mock_config, valid_t0_df):
        """_normalize_sample should handle channel mismatches gracefully."""
        sampler = self._create_sampler(sample_dataarray, mock_config, valid_t0_df)

        # Means/stds only have 2 of 3 channels
        mock_means = xr.DataArray([10.0, 200.0], dims=["channel"],
                                  coords={"channel": ["t", "dswrf"]})
        mock_stds = xr.DataArray([5.0, 100.0], dims=["channel"],
                                 coords={"channel": ["t", "dswrf"]})

        data = xr.DataArray(
            np.array([[[15.0]], [[300.0]], [[75.0]]]),
            dims=["channel", "latitude", "longitude"],
            coords={"channel": ["t", "dswrf", "tcc"],
                    "latitude": [40.0], "longitude": [-100.0]},
        )

        with patch.dict(
            "open_data_pvnet.nwp.gfs_dataset.NWP_MEANS", {"gfs": mock_means}
        ), patch.dict(
            "open_data_pvnet.nwp.gfs_dataset.NWP_STDS", {"gfs": mock_stds}
        ):
            result = sampler._normalize_sample(data)
            # Should only contain the intersection channels
            assert "tcc" not in result.channel.values
            assert "t" in result.channel.values
            assert "dswrf" in result.channel.values
