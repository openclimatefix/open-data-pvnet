import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import numpy as np
import xarray as xr
import yaml
import pandas as pd
from huggingface_hub.utils import EntryNotFoundError

from open_data_pvnet.utils.config_loader import load_config
from open_data_pvnet.utils.env_loader import load_environment_variables
from open_data_pvnet.utils.data_converters import convert_nc_to_zarr
from open_data_pvnet.utils.data_uploader import (
    _validate_config,
    _validate_token,
    _ensure_repository,
    create_tar_archive,
    create_zarr_zip,
    _upload_archive,
    upload_to_huggingface,
)
from open_data_pvnet.utils.data_downloader import load_zarr_data, restructure_dataset


# Fixtures
@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary input and output directories."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    return input_dir, output_dir


@pytest.fixture
def sample_nc_file(temp_dirs):
    """Create a sample NetCDF file for testing."""
    input_dir, _ = temp_dirs
    nc_file = input_dir / "test_data.nc"

    # Create a simple dataset
    data = np.random.rand(10, 10)
    ds = xr.Dataset(
        {
            "temperature": (["x", "y"], data),
            "pressure": (["x", "y"], data * 2),
        },
        coords={
            "x": np.arange(10),
            "y": np.arange(10),
        },
    )
    ds.to_netcdf(nc_file)
    return nc_file


@pytest.fixture
def mock_config():
    return {
        "general": {"destination_dataset_id": "test/dataset"},
        "input_data": {"nwp": {"met_office": {"local_output_dir": "/test/path"}}},
    }


@pytest.fixture
def mock_hf_api():
    api = Mock()
    api.whoami.return_value = {"name": "test_user"}
    return api


@pytest.fixture
def mock_tarfile():
    with patch("tarfile.open") as mock:
        yield mock


@pytest.fixture
def mock_zipstore():
    with patch("zarr.storage.ZipStore") as mock:
        yield mock


@pytest.fixture
def sample_zarr_dataset():
    """Create a sample dataset that mimics the Met Office data structure."""
    # Create sample data
    times = pd.date_range("2024-01-01", periods=24, freq="h")
    lats = np.linspace(49, 61, 970)
    lons = np.linspace(-10, 2, 1042)

    ds = xr.Dataset(
        {
            "air_temperature": (
                ["projection_y_coordinate", "projection_x_coordinate"],
                np.random.rand(970, 1042),
            ),
            "height": np.array([10]),
        },
        coords={
            "projection_y_coordinate": lats,
            "projection_x_coordinate": lons,
            "forecast_period": np.array([0, 1, 2, 3]),
            "forecast_reference_time": np.datetime64("2024-01-01"),
            "time": times,
        },
    )
    return ds


# -----------------------------
# Config Loader Tests
# -----------------------------

def test_load_config_valid_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_content = """
    key1: value1
    key2:
        nested_key: value2
    """
    config_file.write_text(config_content)

    config = load_config(str(config_file))

    assert isinstance(config, dict)
    assert config["key1"] == "value1"
    assert config["key2"]["nested_key"] == "value2"


def test_load_config_empty_path():
    with pytest.raises(ValueError, match="Missing config path"):
        load_config("")


def test_load_config_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_file.yaml")


def test_load_config_invalid_yaml(tmp_path):
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text("key1: value1\n  invalid_indent: value2")

    with pytest.raises(yaml.YAMLError):
        load_config(str(config_file))


# -----------------------------
# Environment Loader Tests
# -----------------------------

def test_load_environment_variables_success(monkeypatch, tmp_path):
    from open_data_pvnet.utils.env_loader import load_environment_variables

    # Create a temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_VAR=test_value\nANOTHER_VAR=123")

    # Ensure environment variables are absent before loading
    monkeypatch.delenv("TEST_VAR", raising=False)
    monkeypatch.delenv("ANOTHER_VAR", raising=False)

    # Point PROJECT_BASE to our temporary directory
    monkeypatch.setenv("PROJECT_BASE", str(tmp_path))

    # Execute loading
    load_environment_variables()

    # Verify that variables were set
    assert os.getenv("TEST_VAR") == "test_value"
    assert os.getenv("ANOTHER_VAR") == "123"


def test_load_environment_variables_file_not_found(monkeypatch, tmp_path):
    from open_data_pvnet.utils.env_loader import load_environment_variables

    # Ensure no .env present
    monkeypatch.setenv("PROJECT_BASE", str(tmp_path))
    monkeypatch.delenv("TEST_VAR", raising=False)

    with pytest.raises(FileNotFoundError) as exc_info:
        load_environment_variables()

    assert ".env file not found" in str(exc_info.value)


# -----------------------------
# Data Converters Tests
# -----------------------------

def test_successful_conversion(temp_dirs, sample_nc_file):
    input_dir, output_dir = temp_dirs

    num_files, total_size = convert_nc_to_zarr(input_dir, output_dir)

    assert num_files == 1
    assert total_size > 0
    assert (output_dir / "test_data.zarr").exists()

    original_ds = xr.open_dataset(sample_nc_file)
    converted_ds = xr.open_zarr(output_dir / "test_data.zarr")
    xr.testing.assert_equal(original_ds, converted_ds)

    original_ds.close()
    converted_ds.close()


def test_no_nc_files(temp_dirs):
    input_dir, output_dir = temp_dirs

    num_files, total_size = convert_nc_to_zarr(input_dir, output_dir)

    assert num_files == 0
    assert total_size == 0


def test_overwrite_existing(temp_dirs, sample_nc_file):
    input_dir, output_dir = temp_dirs

    convert_nc_to_zarr(input_dir, output_dir)
    num_files, _ = convert_nc_to_zarr(input_dir, output_dir, overwrite=False)
    assert num_files == 0

    num_files, _ = convert_nc_to_zarr(input_dir, output_dir, overwrite=True)
    assert num_files == 1


def test_invalid_input_dir():
    with pytest.raises(Exception):
        convert_nc_to_zarr(Path("nonexistent_dir"), Path("output_dir"))


# -----------------------------
# Data Uploader Tests
# -----------------------------

def test_validate_config_success(mock_config):
    repo_id, zarr_base_path = _validate_config(mock_config)
    assert repo_id == "test/dataset"
    assert zarr_base_path == Path("/test/path/zarr")


def test_validate_config_missing_dataset_id():
    config = {"general": {}}
    with pytest.raises(ValueError, match="No destination_dataset_exception"):
        _validate_config(config)

@patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "test_token"})
@patch("open_data_pvnet.utils.data_uploader.HfApi")
def test_validate_token_success(mock_hf_api_class, mock_hf_api):
    mock_hf_api_class.return_value = mock_hf_api
    api, token = _validate_token()
    assert token == "test_token"
    assert api == mock_hf_api

@patch.dict(os.environ, {}, clear=True)
def test_validate_token_missing_token():
    with pytest.raises(ValueError, match="Hugging Face token not found"):
        _validate_token()


def test_ensure_repository_exists(mock_hf_api):
    _ensure_repository(mock_hf_api, "test/dataset", "test_token")
    mock_hf_api.dataset_info.assert_called_once_with("test/dataset", token="test_token")


def test_ensure_repository_create_new(mock_hf_api):
    mock_hf_api.dataset_info.side_effect = Exception("Not found")
    _ensure_repository(mock_hf_api, "test/dataset", "test_token")
    mock_hf_api.create_repo.assert_called_once_with(
        repo_id="test/dataset", repo_type="dataset", token="test_token"
    )


def test_create_tar_archive(tmp_path, mock_tarfile):
    folder_path = tmp_path / "test_folder"
    folder_path.mkdir()
    archive_path = create_tar_archive(folder_path, "test.tar.gz")
    assert archive_path == folder_path.parent / "test.tar.gz"
    mock_tarfile.assert_called_once()

@patch("zarr.open")
def test_create_zarr_zip(mock_zarr_open, tmp_path):
    folder_path = tmp_path / "test_folder.zarr"
    folder_path.mkdir()

    mock_group = Mock()
    mock_zarr_open.return_value = mock_group

    with (
        patch("open_data_pvnet.utils.data_uploader.ZipStore") as mock_zip_store,
        patch("zarr.DirectoryStore") as mock_dir_store,
        patch("zarr.copy_store") as mock_copy_store,
    ):
        mock_zip_store.return_value.__enter__.return_value = Mock()

        archive_path = create_zarr_zip(folder_path, "test.zarr.zip")

        assert archive_path == folder_path.parent / "test.zarr.zip"
        mock_zarr_open.assert_called_once_with(str(folder_path))
        mock_dir_store.assert_called_once_with(str(folder_path))
        mock_zip_store.assert_called_once_with(str(archive_path), mode="w")
        mock_copy_store.assert_called_once()


def test_create_zarr_zip_invalid_zarr(tmp_path):
    folder_path = tmp_path / "test_folder.zarr"
    folder_path.mkdir()

    with patch("zarr.open", side_effect=Exception("Invalid Zarr")):
        with pytest.raises(RuntimeError) as exc_info:
            create_zarr_zip(folder_path, "test.zarr.zip")

        assert "Failed to create Zarr zip archive" in str(exc_info.value)
        assert "Invalid Zarr" in str(exc_info.value)


def test_create_tar_archive_existing_no_overwrite(tmp_path):
    folder_path = tmp_path / "test_folder"
    folder_path.mkdir()
    archive_path = tmp_path / "test.tar.gz"
    archive_path.touch()
    result = create_tar_archive(folder_path, "test.tar.gz", overwrite=False)
    assert result == archive_path


def test_create_zarr_zip_existing_no_overwrite(tmp_path):
    folder_path = tmp_path / "test_folder.zarr"
    folder_path.mkdir()
    archive_path = tmp_path / "test.zarr.zip"
    archive_path.touch()

    result = create_zarr_zip(folder_path, "test.zarr.zip", overwrite=False)
    assert result == archive_path


def test_upload_archive_success(mock_hf_api):
    archive_path = Path("test.zarr.zip")
    _upload_archive(mock_hf_api, archive_path, "test/dataset", "test_token", False, 2024, 1, 1)
    mock_hf_api.upload_file.assert_called_once()


def test_upload_archive_with_overwrite(mock_hf_api):
    archive_path = Path("test.zarr.zip")
    _upload_archive(mock_hf_api, archive_path, "test/dataset", "test_token", True, 2024, 1, 1)
    mock_hf_api.delete_file.assert_called_once()
    mock_hf_api.upload_file.assert_called_once()

@patch("open_data_pvnet.utils.data_uploader.load_config")
def test_upload_to_huggingface_success(mock_load_config, mock_config, tmp_path):
    mock_load_config.return_value = mock_config

    with (
        patch("open_data_pvnet.utils.data_uploader._validate_token") as mock_validate_token,
        patch("open_data_pvnet.utils.data_uploader._ensure_repository") as mock_ensure_repo,
        patch("open_data_pvnet.utils.data_uploader.create_zarr_zip") as mock_create_archive,
        patch("open_data_pvnet.utils.data_uploader._upload_archive") as mock_upload,
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.unlink") as mock_unlink,
    ):
        mock_validate_token.return_value = (Mock(), "test_token")
        mock_create_archive.return_value = tmp_path / "test.zarr.zip"
        mock_exists.return_value = True

        upload_to_huggingface(
            Path("config.yaml"), "test_folder", 2024, 1, 1, overwrite=False, archive_type="zarr.zip"
        )

        mock_load_config.assert_called_once()
        mock_validate_token.assert_called_once()
        mock_ensure_repo.assert_called_once()
        mock_create_archive.assert_called_once()
        mock_upload.assert_called_once()
        mock_unlink.assert_called_once()


def test_upload_to_huggingface_missing_folder(mock_config):
    with (
        patch("open_data_pvnet.utils.data_uploader.load_config") as mock_load_config,
        patch("open_data_pvnet.utils.data_uploader._validate_token") as mock_validate_token,
        patch("pathlib.Path.exists") as mock_exists,
    ):
        mock_load_config.return_value = mock_config
        mock_validate_token.return_value = (Mock(), "test_token")
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            upload_to_huggingface(
                Path("config.yaml"),
                "nonexistent_folder",
                2024,
                1,
                1,
                overwrite=False,
                archive_type="zarr.zip",
            )


def test_restructure_dataset(sample_zarr_dataset):
    restructured_ds = restructure_dataset(sample_zarr_dataset)

    assert "step" in restructured_ds.dims
    assert "initialization_time" in restructured_ds.coords
    assert "height" not in restructured_ds.coords
    assert "bnds" not in restructured_ds.dims
    assert "projection_x_coordinate" in restructured_ds.dims
    assert "projection_y_coordinate" in restructured_
