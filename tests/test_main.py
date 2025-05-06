import pytest
from unittest.mock import patch
from open_data_pvnet.main import (
    configure_parser,
    load_env_and_setup_logger,
    main,
)

# Helper function for argument parsing tests
def parse_args(args_list):
    parser = configure_parser()
    return parser.parse_args(args_list)


@pytest.mark.parametrize(
    "args, expected",
    [
        (["--list", "providers"], {"list": "providers"}),
        (
            [
                "metoffice",
                "archive",
                "--year", "2024",
                "--month", "3",
                "--day", "1",
                "--hour", "12",
                "--region", "global",
                "--archive-type", "zarr.zip",
            ],
            {
                "command": "metoffice",
                "operation": "archive",
                "year": 2024,
                "month": 3,
                "day": 1,
                "hour": 12,
                "region": "global",
                "archive_type": "zarr.zip",
                "overwrite": False,
            },
        ),
        (
            [
                "gfs",
                "archive",
                "--year", "2024",
                "--month", "3",
                "--day", "1",
                "--archive-type", "tar",
            ],
            {
                "command": "gfs",
                "operation": "archive",
                "year": 2024,
                "month": 3,
                "day": 1,
                "archive_type": "tar",
                "overwrite": False,
            },
        ),
    ],
)
def test_configure_parser(args, expected):
    parsed_args = parse_args(args)
    for key, value in expected.items():
        assert getattr(parsed_args, key) == value


@patch("open_data_pvnet.main.load_environment_variables")
@patch("open_data_pvnet.main.logging.basicConfig")
def test_load_env_and_setup_logger_success(mock_logging, mock_load_env):
    """Test successful environment setup and logging initialization."""
    load_env_and_setup_logger()
    mock_load_env.assert_called_once()
    mock_logging.assert_called_once()


@patch("open_data_pvnet.main.load_environment_variables")
def test_load_env_and_setup_logger_failure(mock_load_env):
    """Test failure when environment variables cannot be loaded."""
    mock_load_env.side_effect = FileNotFoundError("Config file not found")
    with pytest.raises(FileNotFoundError):
        load_env_and_setup_logger()


@pytest.mark.parametrize(
    "test_args, expected_kwargs",
    [
        (
            [
                "metoffice",
                "load",
                "--year", "2024",
                "--month", "3",
                "--day", "1",
                "--hour", "12",
                "--region", "uk",
                "--chunks", "time:24,latitude:100",
            ],
            {
                "provider": "metoffice",
                "year": 2024,
                "month": 3,
                "day": 1,
                "hour": 12,
                "region": "uk",
                "overwrite": False,
                "chunks": "time:24,latitude:100",
                "remote": False,
            },
        ),
        (
            [
                "metoffice",
                "load",
                "--year", "2024",
                "--month", "3",
                "--day", "1",
                "--hour", "12",
                "--region", "uk",
                "--chunks", "time:24,latitude:100",
                "--remote",
            ],
            {
                "provider": "metoffice",
                "year": 2024,
                "month": 3,
                "day": 1,
                "hour": 12,
                "region": "uk",
                "overwrite": False,
                "chunks": "time:24,latitude:100",
                "remote": True,
            },
        ),
    ],
)
@patch("open_data_pvnet.main.handle_load")
@patch("open_data_pvnet.main.load_env_and_setup_logger")
def test_main_metoffice_load(mock_load_env, mock_handle_load, test_args, expected_kwargs):
    """Test 'metoffice load' command execution."""
    with patch("sys.argv", ["script"] + test_args):
        main()
        mock_handle_load.assert_called_once_with(**expected_kwargs)


@patch("open_data_pvnet.main.print")
@patch("open_data_pvnet.main.load_env_and_setup_logger")
def test_main_list_providers(mock_load_env, mock_print):
    """Test listing available providers."""
    test_args = ["--list", "providers"]
    with patch("sys.argv", ["script"] + test_args):
        main()
        assert mock_print.call_count >= 3  # Expect at least three providers listed


def test_invalid_arguments():
    """Test handling of invalid arguments."""
    parser = configure_parser()
    with pytest.raises(SystemExit):  # argparse raises SystemExit on invalid input
        parser.parse_args(["invalid_command"])


@patch("open_data_pvnet.main.handle_load")
@patch("open_data_pvnet.main.load_env_and_setup_logger")
def test_main_with_missing_required_args(mock_load_env, mock_handle_load):
    """Test missing required arguments handling."""
    test_args = ["metoffice", "load", "--year", "2024"]  # Missing required args
    with patch("sys.argv", ["script"] + test_args):
        with pytest.raises(SystemExit):  # argparse exits on missing args
            main()
