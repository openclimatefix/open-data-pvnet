import os
import logging
import tarfile
from pathlib import Path
from huggingface_hub import HfApi
from open_data_pvnet.utils.config_loader import load_config
import zarr
from zarr.storage import ZipStore

logger = logging.getLogger(__name__)


def _validate_config(config):
    """Validate configuration and return required values."""
    repo_id = config.get("general", {}).get("destination_dataset_id")
    if not repo_id:
        raise ValueError("No destination_dataset_id found in the configuration file.")

    local_output_dir = config["input_data"]["nwp"]["met_office"]["local_output_dir"]
    zarr_base_path = Path(local_output_dir) / "zarr"
    return repo_id, zarr_base_path


def _validate_token():
    """Validate Hugging Face token and return API instance."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Ensure HUGGINGFACE_TOKEN is set in the environment."
        )

    hf_api = HfApi()
    try:
        user_info = hf_api.whoami(token=hf_token)
        logger.info(f"Authenticated with Hugging Face as user: {user_info['name']}")
    except Exception as e:
        raise ValueError(
            f"Failed to authenticate with Hugging Face. Check your token. Details: {e}"
        )
    return hf_api, hf_token


def _ensure_repository(hf_api, repo_id, hf_token):
    """Ensure repository exists, create if it doesn't."""
    try:
        hf_api.dataset_info(repo_id, token=hf_token)
        logger.info(f"Found existing repository: {repo_id}")
    except Exception:
        logger.info(f"Creating new dataset repository: {repo_id}")
        hf_api.create_repo(repo_id=repo_id, repo_type="dataset", token=hf_token)


def create_tar_archive(folder_path: Path, archive_name: str, overwrite: bool = False) -> Path:
    """
    Create a .tar.gz archive of the given folder, overwriting if specified.

    Args:
        folder_path (Path): The folder to archive.
        archive_name (str): Name of the archive file.
        overwrite (bool): Whether to overwrite the existing archive.

    Returns:
        Path: The path to the created archive.
    """
    archive_path = folder_path.parent / archive_name

    if archive_path.exists() and not overwrite:
        logger.info(f"Archive already exists: {archive_path}. Skipping creation.")
        return archive_path

    if archive_path.exists() and overwrite:
        logger.info(f"Overwriting existing archive: {archive_path}")
        archive_path.unlink()  # Delete the existing archive

    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(folder_path, arcname=folder_path.name)
        logger.info(f"Created archive: {archive_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create archive: {e}")

    return archive_path


def _upload_archive(
    hf_api,
    archive_path: Path,
    repo_id: str,
    hf_token: str,
    overwrite: bool,
    year: int,
    month: int,
    day: int,
):
    """
    Upload an archive file to the Hugging Face repository in the data/year/month/day structure.

    Args:
        hf_api (HfApi): The Hugging Face API instance.
        archive_path (Path): Path to the archive file.
        repo_id (str): Repository ID.
        hf_token (str): Hugging Face authentication token.
        overwrite (bool): Whether to overwrite existing files.
        year (int): Year for folder structure.
        month (int): Month for folder structure.
        day (int): Day for folder structure.
    """
    # Create the path structure: data/year/month/day/archive_name
    target_path = f"data/{year:04d}/{month:02d}/{day:02d}/{archive_path.name}"
    logger.info(f"Uploading archive {archive_path} to {repo_id}:{target_path}")

    try:
        if overwrite:
            try:
                # Delete the file if it exists and overwrite is True
                hf_api.delete_file(
                    path_in_repo=target_path, repo_id=repo_id, repo_type="dataset", token=hf_token
                )
                logger.info(f"Deleted existing file {target_path} from repository")
            except Exception as e:
                logger.debug(
                    f"File {target_path} not found in repository or couldn't be deleted: {e}"
                )

        # Upload the new file
        hf_api.upload_file(
            path_or_fileobj=str(archive_path),
            path_in_repo=target_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )
        logger.info(f"Upload completed for {archive_path} to {repo_id}:{target_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to upload archive: {e}")


def create_zarr_zip(folder_path: Path, archive_name: str, overwrite: bool = False) -> Path:
    """
    Create a zip archive of a Zarr directory using zarr.zip functionality.

    Args:
        folder_path (Path): The Zarr folder to archive.
        archive_name (str): Name of the archive file (should end with .zip).
        overwrite (bool): Whether to overwrite the existing archive.

    Returns:
        Path: The path to the created archive.

    Raises:
        RuntimeError: If archive creation fails.
        ValueError: If the input folder is not a valid Zarr directory.
    """
    if not archive_name.endswith(".zip"):
        archive_name = f"{archive_name}.zip"

    archive_path = folder_path.parent / archive_name

    # Check if archive already exists
    if archive_path.exists() and not overwrite:
        logger.info(f"Archive already exists: {archive_path}. Skipping creation.")
        return archive_path

    if archive_path.exists() and overwrite:
        logger.info(f"Overwriting existing archive: {archive_path}")
        archive_path.unlink()  # Delete the existing archive

    try:
        # Try to open the Zarr directory to verify it's valid
        try:
            zarr.open(str(folder_path))
        except Exception as e:
            raise ValueError(f"Not a valid Zarr directory: {folder_path}. Error: {e}")

        # Create zip archive
        logger.info(f"Creating Zarr zip archive: {archive_path}")

        # Open original Zarr directory
        source_store = zarr.DirectoryStore(str(folder_path))

        # Create new zip store and copy data
        with ZipStore(str(archive_path), mode="w") as zip_store:
            zarr.copy_store(source_store, zip_store)

        logger.info(f"Created Zarr zip archive: {archive_path}")
        return archive_path

    except Exception as e:
        if archive_path.exists():
            archive_path.unlink()  # Clean up partial archive on failure
        raise RuntimeError(f"Failed to create Zarr zip archive: {e}")


def upload_to_huggingface(
    config_path: Path,
    folder_name: str,
    year: int,
    month: int,
    day: int,
    overwrite: bool = False,
    archive_type: str = "zarr.zip",
):
    """
    Upload a specific folder from the local Zarr directory to a Hugging Face dataset repository.

    Args:
        config_path (Path): Path to the configuration YAML file.
        folder_name (str): Name of the folder to upload (e.g., '2022-12-01-00').
        year (int): Year for folder structure.
        month (int): Month for folder structure.
        day (int): Day for folder structure.
        overwrite (bool): Whether to overwrite existing files in the repository.
        archive_type (str): Type of archive to create ("zarr.zip" or "tar").

    Raises:
        Exception: If the upload fails due to authentication, network, or other issues.
    """
    try:
        # Load and validate configuration
        config = load_config(config_path)
        repo_id, zarr_base_path = _validate_config(config)

        # Validate token and ensure repository
        hf_api, hf_token = _validate_token()
        _ensure_repository(hf_api, repo_id, hf_token)

        # Local paths
        folder_path = zarr_base_path / folder_name
        if not folder_path.exists():
            raise FileNotFoundError(f"Local folder does not exist: {folder_path}")

        # Create archive based on type
        if archive_type == "zarr.zip":
            archive_name = f"{folder_name}.zarr.zip"
            archive_path = create_zarr_zip(folder_path, archive_name, overwrite=overwrite)
        else:  # tar
            archive_name = f"{folder_name}.tar.gz"
            archive_path = create_tar_archive(folder_path, archive_name, overwrite=overwrite)

        # Upload archive with year/month/day structure
        _upload_archive(hf_api, archive_path, repo_id, hf_token, overwrite, year, month, day)

        logger.info(f"Upload to Hugging Face completed: {repo_id}")

        # Remove the archive after successful upload
        logger.info(f"Removing local archive: {archive_path}")
        archive_path.unlink()

    except Exception as e:
        logger.error(f"Error uploading to Hugging Face: {e}")
        raise


def upload_monthly_zarr(
    config_path: Path,
    year: int,
    month: int,
    overwrite: bool = False,
) -> None:
    """
    Upload a monthly consolidated zarr.zip file to the Hugging Face dataset repository.
    """
    repo_id = "openclimatefix/met-office-uk-deterministic-solar"

    try:
        # Validate token and get API instance (same as daily uploads)
        hf_api, hf_token = _validate_token()
        _ensure_repository(hf_api, repo_id, hf_token)

        # Construct paths
        month_str = f"{month:02d}"
        monthly_file = f"{year}-{month_str}.zarr.zip"
        local_path = Path("data") / str(year) / month_str / "monthly" / monthly_file

        if not local_path.exists():
            raise FileNotFoundError(f"Monthly consolidated file not found: {local_path}")

        # Create the path structure: data/year/month/monthly/file.zarr.zip
        target_path = f"data/{year}/{month_str}/monthly/{monthly_file}"
        logger.info(f"Uploading monthly archive {local_path} to {repo_id}:{target_path}")

        try:
            if overwrite:
                try:
                    # Delete the file if it exists and overwrite is True
                    hf_api.delete_file(
                        path_in_repo=target_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=hf_token,
                    )
                    logger.info(f"Deleted existing file {target_path} from repository")
                except Exception as e:
                    logger.debug(
                        f"File {target_path} not found in repository or couldn't be deleted: {e}"
                    )

            # Upload the new file
            hf_api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=target_path,
                repo_id=repo_id,
                repo_type="dataset",
                token=hf_token,
            )
            logger.info(f"Upload completed for {local_path} to {repo_id}:{target_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to upload monthly archive: {e}")

    except Exception as e:
        logger.error(f"Error uploading monthly archive to Hugging Face: {e}")
        raise
