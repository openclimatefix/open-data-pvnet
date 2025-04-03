import pytest
from unittest.mock import Mock

from open_data_pvnet.nwp.met_office import (
    generate_prefix,
    fetch_met_office_data,
    process_met_office_data,
)


@pytest.fixture
def mock_config():
    return {
        "input_data": {
            "nwp": {
                "met_office": {
                    "s3_bucket": "test-bucket",
                    "local_output_dir": "test_output",
                    "nwp_channels": ["temperature", "pressure"],
                    "nwp_accum_channels": ["precipitation"],
                }
            }
        }
    }


def test_generate_prefix_uk():
    prefix = generate_prefix("uk", 2023, 12, 25, 0)
    assert prefix == "uk-deterministic-2km/20231225T0000Z/"


def test_generate_prefix_global():
    prefix = generate_prefix("global", 2023, 12, 25, 0)
    assert prefix == "global-deterministic-10km/20231225T0000Z/"


def test_process_met_office_data_success(mocker, mock_config, tmp_path):
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.met_office.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.met_office.CONFIG_PATHS", {"uk": "test_config.yaml"})
    mocker.patch("open_data_pvnet.nwp.met_office.load_config", return_value=mock_config)
    mock_fetch = mocker.patch(
        "open_data_pvnet.nwp.met_office.fetch_met_office_data", return_value=3
    )
    mock_convert = mocker.patch(
        "open_data_pvnet.nwp.met_office.convert_nc_to_zarr", return_value=(3, 1000)
    )
    mock_upload = mocker.patch("open_data_pvnet.nwp.met_office.upload_to_huggingface")
    mock_rmtree = mocker.patch("open_data_pvnet.nwp.met_office.shutil.rmtree")

    # Call function with default archive_type
    process_met_office_data(2023, 12, 25, 0, "uk", overwrite=False)

    # Assertions
    mock_fetch.assert_called_once_with(2023, 12, 25, 0, "uk")
    mock_convert.assert_called_once()
    mock_upload.assert_called_once_with(
        mocker.ANY,  # config_path
        mocker.ANY,  # folder_name
        2023,  # year
        12,  # month
        25,  # day
        False,  # overwrite
        "zarr.zip",  # default archive_type
    )
    assert mock_rmtree.call_count == 2


def test_process_met_office_data_with_tar(mocker, mock_config, tmp_path):
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.met_office.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.met_office.CONFIG_PATHS", {"uk": "test_config.yaml"})
    mocker.patch("open_data_pvnet.nwp.met_office.load_config", return_value=mock_config)
    mock_fetch = mocker.patch(
        "open_data_pvnet.nwp.met_office.fetch_met_office_data", return_value=3
    )
    mock_convert = mocker.patch(
        "open_data_pvnet.nwp.met_office.convert_nc_to_zarr", return_value=(3, 1000)
    )
    mock_upload = mocker.patch("open_data_pvnet.nwp.met_office.upload_to_huggingface")
    mock_rmtree = mocker.patch("open_data_pvnet.nwp.met_office.shutil.rmtree")

    # Call function with tar archive_type
    process_met_office_data(2023, 12, 25, 0, "uk", overwrite=True, archive_type="tar")

    # Assertions
    mock_fetch.assert_called_once_with(2023, 12, 25, 0, "uk")
    mock_convert.assert_called_once()
    mock_upload.assert_called_once_with(
        mocker.ANY,  # config_path
        mocker.ANY,  # folder_name
        2023,  # year
        12,  # month
        25,  # day
        True,  # overwrite
        "tar",  # specified archive_type
    )
    assert mock_rmtree.call_count == 2


def test_process_met_office_data_no_files(mocker, mock_config, tmp_path):
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.met_office.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.met_office.CONFIG_PATHS", {"uk": "test_config.yaml"})
    mocker.patch("open_data_pvnet.nwp.met_office.load_config", return_value=mock_config)
    mock_fetch = mocker.patch(
        "open_data_pvnet.nwp.met_office.fetch_met_office_data", return_value=0
    )
    mock_convert = mocker.patch("open_data_pvnet.nwp.met_office.convert_nc_to_zarr")
    mock_upload = mocker.patch("open_data_pvnet.nwp.met_office.upload_to_huggingface")

    # Call function
    process_met_office_data(2023, 12, 25, 0, "uk")

    # Assertions
    mock_fetch.assert_called_once_with(2023, 12, 25, 0, "uk")
    mock_convert.assert_not_called()
    mock_upload.assert_not_called()


def test_fetch_met_office_data_success(mocker, mock_config):
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.met_office.CONFIG_PATHS", {"uk": "test_config.yaml"})
    mocker.patch("open_data_pvnet.nwp.met_office.load_config", return_value=mock_config)
    mock_s3 = Mock()
    mocker.patch("open_data_pvnet.nwp.met_office.boto3.client", return_value=mock_s3)

    # Mock S3 response
    mock_s3.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "uk-deterministic-2km/20231225T0000Z/file-temperature.nc"},
            {"Key": "uk-deterministic-2km/20231225T0000Z/file-pressure.nc"},
            {"Key": "uk-deterministic-2km/20231225T0000Z/file-precipitation.nc"},
            {"Key": "uk-deterministic-2km/20231225T0000Z/file-ignored.nc"},
        ]
    }

    # Call function
    total_files = fetch_met_office_data(2023, 12, 25, 0, "uk")

    # Assertions
    assert total_files == 3
    mock_s3.list_objects_v2.assert_called_once()
    assert mock_s3.download_file.call_count == 3


def test_fetch_met_office_data_no_files(mocker, mock_config):
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.met_office.CONFIG_PATHS", {"uk": "test_config.yaml"})
    mocker.patch("open_data_pvnet.nwp.met_office.load_config", return_value=mock_config)
    mock_s3 = Mock()
    mocker.patch("open_data_pvnet.nwp.met_office.boto3.client", return_value=mock_s3)

    # Mock empty S3 response
    mock_s3.list_objects_v2.return_value = {}

    # Call function
    total_files = fetch_met_office_data(2023, 12, 25, 0, "uk")

    # Assertions
    assert total_files == 0
    mock_s3.list_objects_v2.assert_called_once()
    mock_s3.download_file.assert_not_called()


def test_fetch_met_office_data_invalid_region():
    with pytest.raises(ValueError, match="Invalid region 'invalid'. Must be 'uk' or 'global'."):
        fetch_met_office_data(2023, 12, 25, 0, "invalid")
