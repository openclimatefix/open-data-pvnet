import zarr
from pathlib import Path
import xarray as xr
import logging
from typing import Union, Optional, List
from huggingface_hub import hf_hub_download
import fsspec
import shutil

logger = logging.getLogger(__name__)

DATASET_REPO = "openclimatefix/met-office-uk-deterministic-solar"


def download_from_hf(repo_path: str, local_path: Path) -> None:
    """Download a file from HuggingFace."""
    logger.info(f"Downloading {repo_path} from Hugging Face...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=DATASET_REPO,
        filename=repo_path,
        local_dir=".",
        repo_type="dataset",
    )
    logger.info(f"Successfully downloaded to {local_path}")


def get_zarr_groups(store: zarr.storage.ZipStore) -> List[str]:
    """Get all top-level Zarr groups from the store."""
    return [k.split("/")[0] for k in store.keys() if k.endswith(".zarr/.zgroup")]


def open_zarr_group(
    store: zarr.storage.ZipStore, group: str, chunks: Optional[dict], consolidated: bool
) -> xr.Dataset:
    """Open a single Zarr group as an xarray Dataset."""
    return xr.open_zarr(store, group=group, chunks=chunks, consolidated=consolidated)


def merge_datasets(datasets: List[xr.Dataset]) -> xr.Dataset:
    """Merge multiple datasets into one and log the result."""
    ds = xr.merge(datasets, compat="override")
    logger.info("Dataset info:")
    logger.info(f"Variables: {list(ds.variables)}")
    logger.info(f"Dimensions: {dict(ds.sizes)}")
    logger.info(f"Coordinates: {list(ds.coords)}")
    return ds


def restructure_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Restructure the dataset to have proper forecast dimensions and remove unnecessary coordinates.

    Args:
        ds (xr.Dataset): Original dataset with flat structure

    Returns:
        xr.Dataset: Restructured dataset with proper dimensions
    """
    # Convert forecast_period and forecast_reference_time to dimensions if they aren't already
    if "forecast_period" in ds.coords and "forecast_period" not in ds.dims:
        ds = ds.expand_dims("forecast_period")

    # Remove unnecessary coordinates
    if "height" in ds.coords:
        ds = ds.drop_vars("height")

    # Remove bounds if they exist and aren't needed
    if "bnds" in ds.dims:
        for var in list(ds.variables):
            if "bnds" in ds[var].dims:
                ds = ds.drop_vars(var)

    # Rename dimensions to be more intuitive
    dim_mapping = {"forecast_period": "step", "forecast_reference_time": "initialization_time"}
    ds = ds.rename(dim_mapping)

    logger.info("Restructured dataset dimensions:")
    logger.info(f"Original dims: {list(ds.dims)}")
    logger.info(f"Original coords: {list(ds.coords)}")

    return ds


def get_hf_url(archive_path: Union[str, Path]) -> str:
    """Construct the HuggingFace URL for a given archive path."""
    return f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{archive_path}"


def _load_remote_zarr(
    url: str, chunks: Optional[dict], consolidated: bool, restructure: bool
) -> xr.Dataset:
    """Load a zarr dataset remotely using fsspec."""
    logger.info(f"Loading dataset remotely from: {url}")
    mapper = fsspec.get_mapper(f"zip::simplecache::{url}")

    # Get all groups from the store
    root = zarr.open(mapper, mode="r")
    zarr_groups = [k for k in root.group_keys() if k.endswith(".zarr")]

    # Open each group as a dataset
    datasets = []
    for group in zarr_groups:
        try:
            group_ds = xr.open_zarr(mapper, group=group, consolidated=consolidated, chunks=chunks)
            datasets.append(group_ds)
        except Exception as e:
            logger.warning(f"Could not open group {group}: {e}")
            continue

    if not datasets:
        raise ValueError("No valid datasets found in the Zarr store")

    ds = merge_datasets(datasets)
    if restructure:
        ds = restructure_dataset(ds)
    return ds


def _load_local_zarr(
    local_path: Path, chunks: Optional[dict], consolidated: bool, restructure: bool
) -> xr.Dataset:
    """Load a zarr dataset from a local file."""
    logger.info(f"Opening zarr store from {local_path}")
    logger.info(f"File size: {local_path.stat().st_size / (1024*1024):.2f} MB")

    store = None
    try:
        store = zarr.storage.ZipStore(str(local_path), mode="r")
        zarr_groups = get_zarr_groups(store)

        datasets = []
        for group in zarr_groups:
            try:
                group_ds = open_zarr_group(store, group, chunks, consolidated)
                datasets.append(group_ds)
            except Exception as e:
                logger.warning(f"Could not open group {group}: {e}")
                continue

        if not datasets:
            raise ValueError("No valid datasets found in the Zarr store")

        ds = merge_datasets(datasets)
        if restructure:
            ds = restructure_dataset(ds)
        return ds
    finally:
        if store is not None:
            store.close()


def load_zarr_data(
    archive_path: Union[str, Path],
    chunks: Optional[dict] = None,
    download: bool = True,
    consolidated: bool = False,
    restructure: bool = True,
    remote: bool = False,
) -> xr.Dataset:
    """
    Load a zarr dataset from a zip archive using xarray.

    Args:
        archive_path (Union[str, Path]): Path to the .zarr.zip file
        chunks (Optional[dict]): Dictionary specifying chunk sizes
        download (bool): Whether to download if not found locally
        consolidated (bool): Whether to use consolidated metadata
        restructure (bool): Whether to restructure the dataset dimensions
        remote (bool): Whether to load the data lazily from HuggingFace

    Returns:
        xr.Dataset: The loaded (and optionally restructured) dataset
    """
    try:
        archive_path = Path(archive_path)

        if remote:
            url = get_hf_url(archive_path)
            return _load_remote_zarr(url, chunks, consolidated, restructure)

        # Local loading logic
        local_path = archive_path
        if not local_path.exists() and download:
            download_from_hf(str(archive_path), local_path)

        return _load_local_zarr(local_path, chunks, consolidated, restructure)

    except Exception as e:
        logger.error(f"Error loading zarr dataset: {e}")
        raise


def load_zarr_data_for_day(  # noqa: C901
    base_path: Path,
    year: int,
    month: int,
    day: int,
    chunks: Optional[dict] = None,
    remote: bool = False,
    download: bool = True,
) -> xr.Dataset:
    """Load all hourly Zarr datasets for a given day."""
    datasets = []
    stores = []  # Keep track of stores to close them later

    try:
        for hour in range(24):
            # Construct paths to match HuggingFace structure
            repo_path = f"data/{year}/{month:02d}/{day:02d}/{year}-{month:02d}-{day:02d}-{hour:02d}.zarr.zip"
            local_path = (
                base_path
                / str(year)
                / f"{month:02d}"
                / f"{day:02d}"
                / f"{year}-{month:02d}-{day:02d}-{hour:02d}.zarr.zip"
            )

            # Log the exact paths we're using
            logger.info(f"Local path: {local_path}")
            logger.info(f"Repo path: {repo_path}")

            try:
                if not local_path.exists() and download:
                    download_from_hf(repo_path, local_path)

                logger.info(f"Opening zarr store from {local_path}")
                logger.info(f"File size: {local_path.stat().st_size / (1024*1024):.2f} MB")

                store = zarr.storage.ZipStore(str(local_path), mode="r")
                stores.append(store)  # Keep track of the store

                zarr_groups = get_zarr_groups(store)
                hour_datasets = []

                for group in zarr_groups:
                    try:
                        group_ds = open_zarr_group(store, group, chunks, False)
                        hour_datasets.append(group_ds)
                    except Exception as e:
                        logger.warning(f"Could not open group {group}: {e}")
                        continue

                if not hour_datasets:
                    raise ValueError("No valid datasets found in the Zarr store")

                dataset = merge_datasets(hour_datasets)
                dataset = restructure_dataset(dataset)

                datasets.append(dataset)
                logger.info(
                    f"Successfully loaded dataset for {year}-{month:02d}-{day:02d} hour {hour:02d}"
                )

            except Exception as e:
                logger.warning(f"Could not load dataset for hour {hour}: {e}")
                continue

        if not datasets:
            raise ValueError(f"No datasets could be loaded for {year}-{month:02d}-{day:02d}")

        # Merge all datasets along the time dimension
        merged_dataset = xr.concat(datasets, dim="time")
        logger.info(f"Successfully merged {len(datasets)} hourly datasets")

        return merged_dataset

    finally:
        # Close all stores in the finally block
        for store in stores:
            try:
                store.close()
            except Exception as e:
                logger.warning(f"Error closing store: {e}")


def save_consolidated_zarr(
    dataset: xr.Dataset, output_path: Union[str, Path], safe_chunks: bool = False
) -> None:
    """Save a consolidated dataset to zarr format.

    Args:
        dataset: The xarray Dataset to save
        output_path: Path where to save the zarr archive
        safe_chunks: If False, allow chunk sizes that might overlap dask chunks
    """
    logger.info(f"Saving consolidated dataset to {output_path}")
    temp_dir = None

    try:
        # Get the first variable name from the dataset
        first_var = list(dataset.data_vars)[0]

        # Check if the dataset has chunks before creating encoding
        encoding = None
        if dataset[first_var].chunks is not None:
            chunks = tuple(x[0] for x in dataset[first_var].chunks)
            encoding = {var: {"chunks": chunks} for var in dataset.data_vars}

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to temporary zarr directory first
        temp_dir = output_path.parent / f"{output_path.stem}_temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Save to temporary zarr directory
        logger.info("Writing to temporary directory...")
        dataset.to_zarr(
            temp_dir,
            mode="w",
            encoding=encoding,
            compute=True,
            consolidated=True,
            safe_chunks=safe_chunks,
        )

        # Create zip archive
        logger.info("Creating zip archive...")
        zip_store = zarr.storage.ZipStore(str(output_path), mode="w")
        temp_store = zarr.DirectoryStore(str(temp_dir))

        try:
            zarr.copy_store(temp_store, zip_store)
        finally:
            zip_store.close()

        logger.info(f"Successfully saved consolidated file to {output_path}")
        logger.info(f"Final file size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    finally:
        # Cleanup temporary directory
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir)


def merge_hours_to_day(
    base_path: Path,
    year: int,
    month: int,
    day: int,
    chunks: Optional[dict] = None,
) -> Path:
    """Merge 24 hourly files into a single daily zarr.zip file."""
    logger.info(f"\nMerging hours for {year}-{month:02d}-{day:02d}")

    # Define paths
    day_str = f"{day:02d}"
    month_str = f"{month:02d}"
    daily_dir = base_path / str(year) / month_str / day_str
    daily_output = daily_dir / "daily" / f"{year}-{month_str}-{day_str}.zarr.zip"

    if daily_output.exists():
        logger.info(f"Daily file already exists: {daily_output}")
        return daily_output

    logger.info(f"Loading hourly data from {daily_dir}")
    daily_dataset = load_zarr_data_for_day(
        base_path,
        year,
        month,
        day,
        chunks=chunks,
        remote=False,
        download=True,
    )

    logger.info(f"Creating daily directory: {daily_output.parent}")
    daily_output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving daily dataset to: {daily_output}")
    save_consolidated_zarr(daily_dataset, daily_output)
    logger.info(f"Successfully created daily file: {daily_output}")

    return daily_output


def test_consolidated_zarr(output_path: Path) -> None:
    """Test that the consolidated zarr.zip file was created correctly.

    Args:
        output_path (Path): Path to the consolidated zarr.zip file
    """
    logger.info(f"\nTesting consolidated file at {output_path}")
    test_ds = xr.open_zarr(output_path)
    logger.info("\nConsolidated dataset info:")
    logger.info(f"Dimensions: {dict(test_ds.dims)}")
    logger.info(f"Variables: {list(test_ds.variables)}")
    logger.info(f"Time range: {test_ds.time.values.min()} to {test_ds.time.values.max()}")
    logger.info(f"Number of time points: {len(test_ds.time)}")
    test_ds.close()


def process_month_by_days(
    base_path: Path,
    year: int,
    month: int,
    chunks: Optional[dict] = None,
) -> List[Path]:
    """Process all days in a month, creating daily consolidated files.

    Returns:
        List[Path]: List of paths to successfully created daily files
    """
    import calendar

    # Get number of days in the month
    _, num_days = calendar.monthrange(year, month)
    successful_files = []

    logger.info(f"\nProcessing all days in {year}-{month:02d}")

    for day in range(1, num_days + 1):
        try:
            logger.info(f"\nProcessing day {year}-{month:02d}-{day:02d}")
            daily_file = merge_hours_to_day(base_path, year, month, day, chunks)

            if daily_file.exists():
                successful_files.append(daily_file)
                logger.info(f"Successfully processed day {day}")
            else:
                logger.warning(f"Failed to create daily file for day {day}")

        except Exception as e:
            logger.error(f"Error processing day {day}: {e}")
            continue

    logger.info(f"\nProcessed {len(successful_files)} days successfully out of {num_days} days")
    return successful_files


def merge_days_to_month(
    base_path: Path,
    year: int,
    month: int,
    chunks: Optional[dict] = None,
    safe_chunks: bool = False,
) -> Path:
    """Merge all daily zarr.zip files in a month into a single monthly zarr.zip file."""
    logger.info(f"\nMerging daily files for {year}-{month:02d}")

    # Define paths
    month_str = f"{month:02d}"
    month_dir = base_path / str(year) / month_str
    monthly_output = month_dir / "monthly" / f"{year}-{month_str}.zarr.zip"

    if monthly_output.exists():
        logger.info(f"Monthly file already exists: {monthly_output}")
        return monthly_output

    # Collect all daily files
    daily_datasets = []
    daily_dir_pattern = month_dir / "*" / "daily" / f"{year}-{month_str}-*.zarr.zip"
    daily_files = sorted(Path(month_dir).glob(f"*/daily/{year}-{month_str}-*.zarr.zip"))

    if not daily_files:
        raise ValueError(f"No daily files found matching pattern: {daily_dir_pattern}")

    logger.info(f"Found {len(daily_files)} daily files")

    # Load and concatenate all daily datasets
    for daily_file in daily_files:
        try:
            logger.info(f"Loading {daily_file}")
            store = zarr.storage.ZipStore(str(daily_file), mode="r")
            ds = xr.open_zarr(store, consolidated=True)
            daily_datasets.append(ds)
            logger.info(f"Successfully loaded {daily_file}")
        except Exception as e:
            logger.error(f"Error loading {daily_file}: {e}")
            store.close()
            continue

    if not daily_datasets:
        raise ValueError("No datasets could be loaded")

    # Concatenate along time dimension
    logger.info("Concatenating datasets...")
    monthly_dataset = xr.concat(daily_datasets, dim="time")
    logger.info(
        f"Combined dataset spans: {monthly_dataset.time.values.min()} to {monthly_dataset.time.values.max()}"
    )

    # Define new chunks for the monthly dataset
    if chunks is None:
        n_times = len(monthly_dataset.time)
        chunks = {
            "time": min(24, n_times),  # One day at a time, or less for partial months
            "projection_y_coordinate": 243,
            "projection_x_coordinate": 261,
        }

    # Rechunk the dataset with explicit chunks
    logger.info(f"Rechunking dataset with chunks: {chunks}")
    monthly_dataset = monthly_dataset.chunk(chunks)

    # Create monthly directory
    logger.info(f"Creating monthly directory: {monthly_output.parent}")
    monthly_output.parent.mkdir(parents=True, exist_ok=True)

    # Save consolidated monthly file
    logger.info(f"Saving monthly dataset to: {monthly_output}")
    save_consolidated_zarr(monthly_dataset, monthly_output, safe_chunks=safe_chunks)
    logger.info(f"Successfully created monthly file: {monthly_output}")

    # Close all datasets
    for ds in daily_datasets:
        ds.close()

    return monthly_output
