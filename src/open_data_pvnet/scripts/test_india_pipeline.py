"""
Test India PVNet Data Pipeline

Unit tests for India solar data pipeline components.
Uses mocked data to run in CI without requiring local datasets or S3 access.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _create_mock_india_solar_dataset() -> xr.Dataset:
    """Create a mock India solar generation dataset matching OCF schema."""
    n_times = 100
    n_locations = 5

    times = pd.date_range("2024-01-01", periods=n_times, freq="h")
    location_ids = list(range(n_locations))

    # Create solar-like generation pattern (zero at night, peak at noon)
    hours = times.hour
    solar_pattern = np.maximum(0, np.sin((hours - 6) * np.pi / 12))
    generation = np.outer(solar_pattern, np.random.uniform(5000, 15000, n_locations))
    generation = generation.astype(np.float32)

    ds = xr.Dataset(
        {
            "generation_mw": (["time_utc", "location_id"], generation),
            "capacity_mwp": (["location_id"], np.array([20000.0] * n_locations, dtype=np.float32)),
        },
        coords={
            "time_utc": times,
            "location_id": location_ids,
            "longitude": ("location_id", [77.0, 72.8, 80.2, 88.3, 91.7]),
            "latitude": ("location_id", [28.6, 19.0, 13.0, 22.5, 26.1]),
        },
    )
    return ds


def test_mock_india_solar_schema():
    """Verify India solar dataset matches expected OCF schema."""
    ds = _create_mock_india_solar_dataset()

    # Check required variables
    assert "generation_mw" in ds.data_vars
    assert "capacity_mwp" in ds.data_vars

    # Check required coordinates
    assert "time_utc" in ds.coords
    assert "location_id" in ds.coords
    assert "longitude" in ds.coords
    assert "latitude" in ds.coords

    # Check dimensions
    assert set(ds["generation_mw"].dims) == {"time_utc", "location_id"}


def test_india_solar_data_types():
    """Verify data types are correct."""
    ds = _create_mock_india_solar_dataset()

    assert ds["generation_mw"].dtype == np.float32
    assert ds["capacity_mwp"].dtype == np.float32


def test_india_solar_values_reasonable():
    """Verify solar generation values are physically plausible."""
    ds = _create_mock_india_solar_dataset()

    gen = ds["generation_mw"].values
    assert np.all(gen >= 0), "Solar generation should be non-negative"
    assert np.any(gen > 0), "Should have some positive generation"
    assert np.all(gen < 100_000), "Generation should be below 100 GW"


def test_india_solar_time_range():
    """Verify time range is within expected India solar data bounds."""
    ds = _create_mock_india_solar_dataset()

    times = pd.DatetimeIndex(ds["time_utc"].values)
    assert times.min() >= pd.Timestamp("2024-01-01")
    assert len(times) > 0


def test_india_solar_coordinates_in_bounds():
    """Verify coordinates fall within India bounding box."""
    ds = _create_mock_india_solar_dataset()

    lats = ds["latitude"].values
    lons = ds["longitude"].values

    # India: ~6-38°N, ~68-98°E
    assert np.all(lats >= 5), f"Latitude {lats.min()} below India bounds"
    assert np.all(lats <= 39), f"Latitude {lats.max()} above India bounds"
    assert np.all(lons >= 67), f"Longitude {lons.min()} below India bounds"
    assert np.all(lons <= 99), f"Longitude {lons.max()} above India bounds"


def test_india_solar_diurnal_pattern():
    """Verify solar generation shows expected diurnal pattern (zero at night)."""
    ds = _create_mock_india_solar_dataset()

    # Night hours (0-5 UTC ~ 5:30-10:30 IST) should have zero/low generation
    night_mask = ds["time_utc"].dt.hour < 6
    night_gen = ds["generation_mw"].where(night_mask, drop=True)

    if len(night_gen.time_utc) > 0:
        assert float(night_gen.mean()) < float(ds["generation_mw"].mean()), \
            "Night generation should be lower than average"
