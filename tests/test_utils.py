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

    # Create a dataset with similar structure to Met Office data
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


def test_load_config_valid_yaml(tmp_path):
    # Create a temporary YAML file
    config_file = tmp_path / "config.yaml"
    config_content = """
    key1: value1
    key2:
        nested_key: value2
    """
    config_file.write_text(config_content)

    # Test loading the config
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
    # Create a temporary file with invalid YAML
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text("key1: value1\n  invalid_indent: value2")

    with pytest.raises(yaml.YAMLError):
        load_config(str(config_file))


def test_load_environment_variables_success(tmp_path, monkeypatch):
    """Test successful loading of environment variables."""
    # Create a temporary .env file
    fake_env_content = "TEST_VAR=test_value\nANOTHER_VAR=123"
    env_file = tmp_path / ".env"
    env_file.write_text(fake_env_content)

    # Patch PROJECT_BASE to point to our temporary directory
    monkeypatch.setattr("open_data_pvnet.utils.env_loader.PROJECT_BASE", tmp_path)

    # Should not raise any exception
    load_environment_variables()


def test_load_environment_variables_file_not_found(tmp_path, monkeypatch):
    """Test that FileNotFoundError is raised when .env file doesn't exist."""
    # Patch PROJECT_BASE to point to our temporary directory
    monkeypatch.setattr("open_data_pvnet.utils.env_loader.PROJECT_BASE", tmp_path)

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_info:
        load_environment_variables()

    assert ".env file not found" in str(exc_info.value)


def test_successful_conversion(temp_dirs, sample_nc_file):
    """Test successful conversion of NC file to Zarr."""
    input_dir, output_dir = temp_dirs

    # Run conversion
    num_files, total_size = convert_nc_to_zarr(input_dir, output_dir)

    # Assertions
    assert num_files == 1
    assert total_size > 0
    assert (output_dir / "test_data.zarr").exists()

    # Verify data integrity
    original_ds = xr.open_dataset(sample_nc_file)
    converted_ds = xr.open_zarr(output_dir / "test_data.zarr")
    xr.testing.assert_equal(original_ds, converted_ds)

    original_ds.close()
    converted_ds.close()


def test_no_nc_files(temp_dirs):
    """Test behavior when no NC files are present."""
    input_dir, output_dir = temp_dirs

    num_files, total_size = convert_nc_to_zarr(input_dir, output_dir)

    assert num_files == 0
    assert total_size == 0


def test_overwrite_existing(temp_dirs, sample_nc_file):
    """Test overwrite behavior."""
    input_dir, output_dir = temp_dirs

    # First conversion
    convert_nc_to_zarr(input_dir, output_dir)

    # Second conversion without overwrite
    num_files, _ = convert_nc_to_zarr(input_dir, output_dir, overwrite=False)
    assert num_files == 0  # No files should be converted

    # Second conversion with overwrite
    num_files, _ = convert_nc_to_zarr(input_dir, output_dir, overwrite=True)
    assert num_files == 1  # File should be converted again


def test_invalid_input_dir():
    """Test behavior with invalid input directory."""
    with pytest.raises(Exception):
        convert_nc_to_zarr(Path("nonexistent_dir"), Path("output_dir"))


# Tests for _validate_config
def test_validate_config_success(mock_config):
    repo_id, zarr_base_path = _validate_config(mock_config)
    assert repo_id == "test/dataset"
    assert zarr_base_path == Path("/test/path/zarr")


def test_validate_config_missing_dataset_id():
    config = {"general": {}}
    with pytest.raises(ValueError, match="No destination_dataset_id found"):
        _validate_config(config)


# Tests for _validate_token
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


# Tests for _ensure_repository
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
    """Test creating a Zarr zip archive."""
    # Setup
    folder_path = tmp_path / "test_folder.zarr"
    folder_path.mkdir()

    # Create a mock source store and group
    mock_group = Mock()
    mock_zarr_open.return_value = mock_group

    with (
        patch("open_data_pvnet.utils.data_uploader.ZipStore") as mock_zip_store,
        patch("zarr.DirectoryStore") as mock_dir_store,
        patch("zarr.copy_store") as mock_copy_store,
    ):
        # Setup mock context manager for ZipStore
        mock_zip_store.return_value.__enter__.return_value = Mock()

        # Run the function
        archive_path = create_zarr_zip(folder_path, "test.zarr.zip")

        # Assertions
        assert archive_path == folder_path.parent / "test.zarr.zip"
        mock_zarr_open.assert_called_once_with(str(folder_path))
        mock_dir_store.assert_called_once_with(str(folder_path))
        mock_zip_store.assert_called_once_with(str(archive_path), mode="w")
        mock_copy_store.assert_called_once()


def test_create_zarr_zip_invalid_zarr(tmp_path):
    """Test creating a Zarr zip archive with invalid Zarr directory."""
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
    """Test that existing archives are not overwritten when overwrite=False."""
    folder_path = tmp_path / "test_folder.zarr"
    folder_path.mkdir()
    archive_path = tmp_path / "test.zarr.zip"
    archive_path.touch()  # Create empty file

    result = create_zarr_zip(folder_path, "test.zarr.zip", overwrite=False)
    assert result == archive_path


# Tests for _upload_archive
def test_upload_archive_success(mock_hf_api):
    archive_path = Path("test.zarr.zip")
    _upload_archive(mock_hf_api, archive_path, "test/dataset", "test_token", False, 2024, 1, 1)
    mock_hf_api.upload_file.assert_called_once()


def test_upload_archive_with_overwrite(mock_hf_api):
    archive_path = Path("test.zarr.zip")
    _upload_archive(mock_hf_api, archive_path, "test/dataset", "test_token", True, 2024, 1, 1)
    mock_hf_api.delete_file.assert_called_once()
    mock_hf_api.upload_file.assert_called_once()


# Tests for upload_to_huggingface
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
    """Test the dataset restructuring functionality."""
    # Restructure the dataset
    restructured_ds = restructure_dataset(sample_zarr_dataset)

    # Check that the dimensions were properly renamed
    assert "step" in restructured_ds.dims
    assert "initialization_time" in restructured_ds.coords

    # Check that unnecessary coordinates were removed
    assert "height" not in restructured_ds.coords
    assert "bnds" not in restructured_ds.dims

    # Check that spatial dimensions were preserved
    assert "projection_x_coordinate" in restructured_ds.dims
    assert "projection_y_coordinate" in restructured_ds.dims


@patch("fsspec.get_mapper")
def test_load_zarr_data_remote(mock_get_mapper, sample_zarr_dataset):
    """Test remote loading of Zarr data."""
    # Mock the mapper to return our sample dataset
    mock_mapper = Mock()
    mock_mapper._store_version = 2  # Set zarr version
    mock_get_mapper.return_value = mock_mapper

    # Create a mock zarr root with group_keys method
    mock_root = Mock()
    mock_root.group_keys.return_value = ["group1.zarr"]

    with (
        patch("xarray.open_zarr") as mock_open_zarr,
        patch("zarr.open", return_value=mock_root) as mock_zarr_open,
    ):
        mock_open_zarr.return_value = sample_zarr_dataset

        # Test remote loading
        ds = load_zarr_data("data/2024/01/01/2024-01-01-00.zarr.zip", remote=True)

        # Check that fsspec was called with the correct URL
        mock_get_mapper.assert_called_once()
        assert "zip::simplecache::" in mock_get_mapper.call_args[0][0]

        # Verify zarr.open was called
        mock_zarr_open.assert_called_once()

        # Verify the dataset was restructured
        assert "step" in ds.dims
        assert "initialization_time" in ds.coords
        assert "height" not in ds.coords


@patch("zarr.storage.ZipStore")
def test_load_zarr_data_local(mock_zipstore, sample_zarr_dataset, tmp_path):
    """Test local loading of Zarr data."""
    # Create a mock zarr store
    mock_store = Mock()
    mock_store._store_version = 2  # Set zarr version
    mock_zipstore.return_value.__enter__.return_value = mock_store

    with (
        patch("xarray.open_zarr") as mock_open_zarr,
        patch("open_data_pvnet.utils.data_downloader.get_zarr_groups") as mock_get_groups,
        patch("open_data_pvnet.utils.data_downloader.open_zarr_group") as mock_open_group,
    ):
        mock_open_zarr.return_value = sample_zarr_dataset
        mock_get_groups.return_value = ["group1.zarr"]  # Mock a group
        mock_open_group.return_value = sample_zarr_dataset

        # Create a temporary zarr file
        test_file = tmp_path / "test.zarr.zip"
        test_file.touch()

        # Test local loading
        ds = load_zarr_data(test_file)

        # Verify the dataset was restructured
        assert "step" in ds.dims
        assert "initialization_time" in ds.coords
        assert "height" not in ds.coords


def test_load_zarr_data_nonexistent_file():
    """Test loading from a nonexistent file."""
    with (
        patch("pathlib.Path.exists", return_value=False),
        pytest.raises((FileNotFoundError, EntryNotFoundError)),  # Accept either error
    ):
        load_zarr_data("nonexistent/file.zarr.zip", remote=False)


def test_load_zarr_data_invalid_url():
    """Test loading from an invalid remote URL."""
    with pytest.raises(Exception):
        load_zarr_data("invalid/path.zarr.zip", remote=True)
