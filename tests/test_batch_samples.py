import os
import shutil
import tempfile
import pytest
import torch
from src.open_data_pvnet.scripts.batch_samples import run_batch_samples


# Mark this as an integration test so that it can be run conditionally.
@pytest.mark.integration
def test_run_batch_samples_real_data():
    """
    Integration test that runs the batch sampling process on a subset of real data.
    Adjust dataset_path and config_path to point to a small, controlled subset of your real data.
    """
    # These paths should point to a small subset of the real dataset and its config.
    dataset_path = os.environ.get("TEST_ZARR_PATH", "s3://ocf-open-data-pvnet/data/gfs.zarr")
    config_path = (
        "src/open_data_pvnet/configs/gfs_data_config.yaml"  # Use a test config if available.
    )
    start_time = "2023-01-01T00:00:00"
    end_time = "2023-01-02T00:00:00"

    temp_dir = tempfile.mkdtemp()
    try:
        run_batch_samples(dataset_path, config_path, start_time, end_time, temp_dir, num_batches=2)
        files = os.listdir(temp_dir)
        assert len(files) == 2, f"Expected 2 batch files, got {len(files)}"
        batch_file = os.path.join(temp_dir, sorted(files)[0])
        saved_batch = torch.load(batch_file)
        assert isinstance(saved_batch, list), "Saved batch should be a list of samples."
    finally:
        shutil.rmtree(temp_dir)
