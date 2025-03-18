import pytest
from unittest.mock import Mock
from pathlib import Path

from open_data_pvnet.nwp.dwd import (
    generate_variable_url,
    fetch_dwd_data,
    process_dwd_data,
)


@pytest.fixture
def mock_config():
    return {
        "input_data": {
            "nwp": {
                "dwd": {
                    "local_output_dir": "test_output",
                    "nwp_channels": ["T_2M", "CLCT"],
                    "nwp_accum_channels": ["ASWDIR_S"],
                }
            }
        }
    }


def test_generate_variable_url():
    """Test the URL generation for DWD data."""
    url = generate_variable_url("T_2M", 2023, 1, 1, 0)
    assert url == "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t_2m/icon-eu_europe_regular-lat-lon_single-level_2023010100_*"

    url = generate_variable_url("CLCT", 2023, 12, 31, 23)
    assert url == "https://opendata.dwd.de/weather/nwp/icon-eu/grib/23/clct/icon-eu_europe_regular-lat-lon_single-level_2023123123_*"


def test_fetch_dwd_data_success(mocker, mock_config, tmp_path):
    """Test successful fetching of DWD data."""
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.dwd.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.dwd.CONFIG_PATH", "test_config.yaml")
    mocker.patch("open_data_pvnet.nwp.dwd.load_config", return_value=mock_config)

    # Mock HTML content
    html_content = b"""
    <html><body>
    <a href="icon-eu_europe_regular-lat-lon_single-level_202301010000_000_T_2M.grib2.bz2">T_2M file</a>
    <a href="icon-eu_europe_regular-lat-lon_single-level_202301010000_000_CLCT.grib2.bz2">CLCT file</a>
    <a href="icon-eu_europe_regular-lat-lon_single-level_202301010000_000_ASWDIR_S.grib2.bz2">ASWDIR_S file</a>
    </body></html>
    """

    # Mock requests
    mock_head = mocker.patch("requests.head")
    mock_head.return_value.status_code = 200

    mock_get = mocker.patch("requests.get")
    mock_get.return_value.content = html_content
    mock_get.return_value.text = html_content.decode('utf-8')
    mock_get.return_value.raise_for_status = Mock()
    mock_get.return_value.iter_content = lambda chunk_size: [b"mock grib data"]

    # Mock file operations
    mocker.patch("pathlib.Path.mkdir")
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("bz2.open")
    mocker.patch("os.remove")

    # Call function
    total_files = fetch_dwd_data(2023, 1, 1, 0)

    # Assertions
    assert total_files == 3
    assert mock_get.call_count == 6  # Three directory listings and three file downloads


def test_fetch_dwd_data_no_files(mocker, mock_config, tmp_path):
    """Test fetching DWD data when no files are available."""
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.dwd.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.dwd.CONFIG_PATH", "test_config.yaml")
    mocker.patch("open_data_pvnet.nwp.dwd.load_config", return_value=mock_config)

    # Mock empty HTML response
    mock_head = mocker.patch("requests.head")
    mock_head.return_value.status_code = 404

    # Call function
    total_files = fetch_dwd_data(2023, 1, 1, 0)

    # Assertions
    assert total_files == 0


def test_process_dwd_data_success(mocker, mock_config, tmp_path):
    """Test successful processing of DWD data."""
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.dwd.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.dwd.CONFIG_PATH", "test_config.yaml")
    mocker.patch("open_data_pvnet.nwp.dwd.load_config", return_value=mock_config)
    mock_fetch = mocker.patch("open_data_pvnet.nwp.dwd.fetch_dwd_data", return_value=3)

    # Mock xarray operations
    mock_ds = mocker.MagicMock()
    mock_ds.data_vars = ["t2m"]
    mock_ds.rename.return_value = mock_ds

    mock_open_dataset = mocker.patch("xarray.open_dataset", return_value=mock_ds)
    mock_merge = mocker.patch("xarray.merge")
    mock_merged = mocker.MagicMock()
    mock_merged.to_zarr = mocker.MagicMock()
    mock_merge.return_value = mock_merged

    # Mock file operations
    mocker.patch("pathlib.Path.exists", return_value=False)
    mocker.patch("pathlib.Path.mkdir")
    mocker.patch("pathlib.Path.glob", return_value=[
        Path("T_2M_file.grib2"),
        Path("CLCT_file.grib2"),
        Path("ASWDIR_S_file.grib2")
    ])

    # Call function
    process_dwd_data(2023, 1, 1, 0)

    # Assertions
    mock_fetch.assert_called_once()
    mock_merged.to_zarr.assert_called_once()


def test_process_dwd_data_no_files(mocker, mock_config):
    """Test processing when no files are downloaded."""
    mocker.patch("open_data_pvnet.nwp.dwd.load_config", return_value=mock_config)
    mock_fetch = mocker.patch("open_data_pvnet.nwp.dwd.fetch_dwd_data", return_value=0)

    process_dwd_data(2023, 1, 1, 0)

    mock_fetch.assert_called_once_with(2023, 1, 1, 0)
    # Should exit early if no files are downloaded 