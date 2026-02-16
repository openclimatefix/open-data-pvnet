"""Unit tests for GFS NWP data processing module."""

import pytest
from unittest.mock import patch


class TestProcessGfsData:
    """Tests for the process_gfs_data function."""

    def test_unsupported_region_raises(self):
        """Unsupported regions should raise ValueError."""
        from open_data_pvnet.nwp.gfs import process_gfs_data

        with pytest.raises(ValueError, match="Unsupported region"):
            process_gfs_data(year=2024, month=1, region="brazil")

    def test_unsupported_region_message(self):
        """Error message should include the bad region name."""
        from open_data_pvnet.nwp.gfs import process_gfs_data

        with pytest.raises(ValueError, match="brazil"):
            process_gfs_data(year=2024, month=1, region="brazil")

    @patch("open_data_pvnet.scripts.download_gfs_india.process_month")
    def test_india_region_calls_process_month(self, mock_process_month):
        """India region should call process_month with correct args."""
        from open_data_pvnet.nwp.gfs import process_gfs_data

        mock_process_month.return_value = "/tmp/gfs_india/2024-01.zarr"

        result = process_gfs_data(year=2024, month=1, region="india")

        mock_process_month.assert_called_once_with(
            year=2024,
            month=1,
            output_dir="data/gfs_india",
            max_days=None,
        )
        assert result == "/tmp/gfs_india/2024-01.zarr"

    @patch("open_data_pvnet.scripts.download_gfs_india.process_month")
    def test_uk_region_accepted(self, mock_process_month):
        """UK region should be accepted without error."""
        from open_data_pvnet.nwp.gfs import process_gfs_data

        mock_process_month.return_value = "/tmp/gfs_uk/2024-06.zarr"

        result = process_gfs_data(year=2024, month=6, region="uk")

        mock_process_month.assert_called_once_with(
            year=2024,
            month=6,
            output_dir="data/gfs_uk",
            max_days=None,
        )
        assert result == "/tmp/gfs_uk/2024-06.zarr"

    @patch("open_data_pvnet.scripts.download_gfs_india.process_month")
    def test_custom_output_dir(self, mock_process_month):
        """Custom output_dir should be passed through."""
        from open_data_pvnet.nwp.gfs import process_gfs_data

        mock_process_month.return_value = "/custom/path/2024-03.zarr"

        process_gfs_data(
            year=2024, month=3, region="india", output_dir="/custom/path"
        )

        mock_process_month.assert_called_once_with(
            year=2024,
            month=3,
            output_dir="/custom/path",
            max_days=None,
        )

    @patch("open_data_pvnet.scripts.download_gfs_india.process_month")
    def test_max_days_passed(self, mock_process_month):
        """max_days should be forwarded to process_month."""
        from open_data_pvnet.nwp.gfs import process_gfs_data

        mock_process_month.return_value = "/tmp/out.zarr"

        process_gfs_data(year=2024, month=1, region="india", max_days=5)

        mock_process_month.assert_called_once_with(
            year=2024,
            month=1,
            output_dir="data/gfs_india",
            max_days=5,
        )

    @patch("open_data_pvnet.scripts.download_gfs_india.process_month")
    def test_none_result_raises_runtime_error(self, mock_process_month):
        """If process_month returns None, should raise RuntimeError."""
        from open_data_pvnet.nwp.gfs import process_gfs_data

        mock_process_month.return_value = None

        with pytest.raises(RuntimeError, match="No GFS data processed"):
            process_gfs_data(year=2024, month=1, region="india")

    @patch("open_data_pvnet.scripts.download_gfs_india.process_month")
    def test_default_output_dir_india(self, mock_process_month):
        """Default output dir for india should be data/gfs_india."""
        from open_data_pvnet.nwp.gfs import process_gfs_data

        mock_process_month.return_value = "/tmp/out.zarr"

        process_gfs_data(year=2024, month=1)  # default region="india"

        call_kwargs = mock_process_month.call_args[1]
        assert call_kwargs["output_dir"] == "data/gfs_india"
