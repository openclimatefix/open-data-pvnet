import argparse
import logging
from open_data_pvnet.utils.env_loader import load_environment_variables
from open_data_pvnet.utils.data_downloader import (
    load_zarr_data,
    load_zarr_data_for_day,
    merge_hours_to_day,
    process_month_by_days,
    merge_days_to_month,
    get_zarr_groups,
)
from pathlib import Path
import concurrent.futures
from typing import List, Tuple, Optional
from open_data_pvnet.utils.data_uploader import upload_monthly_zarr, upload_to_huggingface
from open_data_pvnet.scripts.archive import handle_archive
from open_data_pvnet.nwp.met_office import CONFIG_PATHS
from open_data_pvnet.utils.data_sampler import prepare_nwp_dataset_for_ocf

logger = logging.getLogger(__name__)

PROVIDERS = ["metoffice", "gfs", "dwd"]
DEFAULT_REGION = "global"  # Default region for Met Office datasets


def load_env_and_setup_logger():
    """Initialize environment variables and configure logging.

    This function performs two main tasks:
    1. Loads environment variables from configuration files
    2. Sets up basic logging configuration with INFO level

    Raises:
        FileNotFoundError: If the environment configuration file cannot be found
    """
    try:
        load_environment_variables()
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger.info("Environment variables loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Error loading environment variables: {e}")
        raise


def _add_common_arguments(parser, provider_name):
    """Add arguments common to both archive and load operations."""
    parser.add_argument("--year", type=int, required=True, help="Year of data")
    parser.add_argument("--month", type=int, required=True, help="Month of data")
    parser.add_argument(
        "--day",
        type=int,
        help="Day of data (optional - if not provided, processes entire month)",
        default=None,
    )

    # Add Met Office specific arguments
    if provider_name == "metoffice":
        parser.add_argument(
            "--hour",
            type=int,
            help="Hour of data (0-23). If not specified, process all hours of the day.",
            default=None,
        )
        parser.add_argument(
            "--region",
            choices=["global", "uk"],
            default="global",
            help="Specify the Met Office dataset region (default: global)",
        )

    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Overwrite existing files in output directories",
    )


def parse_chunks(chunks_str):
    """Parse chunks string into dictionary."""
    if not chunks_str:
        return None
    chunks = {}
    for chunk in chunks_str.split(","):
        dim, size = chunk.split(":")
        chunks[dim.strip()] = int(size)
    return chunks


def handle_load(provider: str, year: int, month: int, day: int, **kwargs):
    """Handle loading archived data."""
    chunks = parse_chunks(kwargs.get("chunks"))
    remote = kwargs.get("remote", False)
    hour = kwargs.get("hour")

    # Base path for the data
    base_path = Path("data") / str(year) / f"{month:02d}" / f"{day:02d}"

    try:
        if hour is not None:
            # Load specific hour
            archive_path = base_path / f"{year}-{month:02d}-{day:02d}-{hour:02d}.zarr.zip"
            dataset = load_zarr_data(
                archive_path,
                chunks=chunks,
                remote=remote,
                download=not remote,
            )
            logger.info(
                f"Successfully loaded dataset for {year}-{month:02d}-{day:02d} hour {hour:02d}"
            )
        else:
            # Load all hours for the day
            dataset = load_zarr_data_for_day(
                base_path,
                year,
                month,
                day,
                chunks=chunks,
                remote=remote,
                download=not remote,
            )
            logger.info(
                f"Successfully loaded all available datasets for {year}-{month:02d}-{day:02d}"
            )

        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def configure_parser():
    """Configure the main argument parser for the CLI tool."""
    parser = argparse.ArgumentParser(prog="open-data-pvnet", description="Open Data PVNet CLI")

    # Create a parent parser for the --list argument
    parser.add_argument(
        "--list",
        choices=["providers"],
        help="List available options (e.g., providers)",
        nargs="?",  # Make it optional
    )

    # Create subparsers for commands (providers)
    subparsers = parser.add_subparsers(dest="command", help="Data provider")

    # Add provider-specific parsers
    for provider in ["metoffice", "gfs"]:
        provider_parser = subparsers.add_parser(
            provider, help=f"Commands for {provider.capitalize()} data"
        )
        operation_subparsers = provider_parser.add_subparsers(
            dest="operation", help="Operation to perform"
        )

        # Archive operation parser
        archive_parser = operation_subparsers.add_parser(
            "archive", help="Archive data to Hugging Face"
        )
        _add_common_arguments(archive_parser, provider)
        archive_parser.add_argument(
            "--archive-type",
            choices=["zarr.zip", "tar"],
            default="zarr.zip",
            help="Type of archive to create (default: zarr.zip)",
        )
        archive_parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="Number of concurrent workers for parallel processing (default: 1)",
        )

        # Load operation parser
        load_parser = operation_subparsers.add_parser("load", help="Load archived data")
        _add_common_arguments(load_parser, provider)
        load_parser.add_argument(
            "--chunks",
            type=str,
            help="Chunking specification in format 'dim1:size1,dim2:size2' (e.g., 'time:24,latitude:100')",
        )
        load_parser.add_argument(
            "--remote",
            action="store_true",
            help="Load data lazily from HuggingFace without downloading",
        )

        # Consolidate operation parser
        consolidate_parser = operation_subparsers.add_parser("consolidate", help="Consolidate data")
        _add_common_arguments(consolidate_parser, provider)

        # Add only the new sample operation to existing metoffice_subparsers
        sample = operation_subparsers.add_parser("sample", help="Sample data for ML training")
        sample.add_argument("--year", type=int, required=True, help="Year of data")
        sample.add_argument("--month", type=int, required=True, help="Month of data")
        sample.add_argument("--day", type=int, required=True, help="Day of data")
        sample.add_argument(
            "--region", choices=["uk", "global"], default="global", help="Region to process"
        )
        sample.add_argument("--chunks", type=str, help="Chunk sizes (e.g., 'time:24,latitude:100')")
        sample.add_argument(
            "--output", type=str, required=True, help="Output path for sampled data"
        )
        sample.add_argument("--remote", action="store_true", help="Use remote data source")

    return parser


def chunk_hours(start: int = 0, end: int = 23, chunk_size: int = 6) -> List[Tuple[int, int]]:
    """Split hours into chunks."""
    chunks = []
    for i in range(start, end + 1, chunk_size):
        chunk_end = min(i + chunk_size - 1, end)
        chunks.append((i, chunk_end))
    return chunks


def archive_hours_chunk(
    provider: str,
    year: int,
    month: int,
    day: int,
    hour_range: Tuple[int, int],
    region: str,
    overwrite: bool,
    archive_type: str,
) -> None:
    """Archive a chunk of hours."""
    start_hour, end_hour = hour_range
    for hour in range(start_hour, end_hour + 1):
        archive_to_hf(
            provider=provider,
            year=year,
            month=month,
            day=day,
            hour=hour,
            region=region,
            overwrite=overwrite,
            archive_type=archive_type,
        )


def parallel_archive(
    provider: str,
    year: int,
    month: int,
    day: int,
    region: str,
    overwrite: bool,
    archive_type: str,
    max_workers: int = 4,
) -> None:
    """Archive data in parallel using multiple workers."""
    hour_chunks = chunk_hours()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                archive_hours_chunk,
                provider,
                year,
                month,
                day,
                chunk,
                region,
                overwrite,
                archive_type,
            )
            for chunk in hour_chunks
        ]
        concurrent.futures.wait(futures)

        # Check for exceptions
        for future in futures:
            if future.exception():
                raise future.exception()


def handle_monthly_consolidation(provider: str, year: int, month: int, **kwargs):
    """Handle consolidating data into zarr.zip files."""
    chunks = parse_chunks(kwargs.get("chunks"))
    base_path = Path("data")
    day = kwargs.get("day")

    try:
        if day is not None:
            # Consolidate a single day
            logger.info(f"Consolidating day {year}-{month:02d}-{day:02d}")
            daily_file = merge_hours_to_day(base_path, year, month, day, chunks)
            logger.info(f"Successfully consolidated day to {daily_file}")
            return

        # First ensure all days are processed
        logger.info(f"Processing all days in month {year}-{month:02d}")
        successful_files = process_month_by_days(base_path, year, month, chunks)

        if successful_files:
            logger.info("\nSuccessfully created daily files:")
            for file in successful_files:
                logger.info(f"- {file}")

            # Now create the monthly file
            logger.info("\nCreating monthly consolidated file")
            monthly_file = merge_days_to_month(base_path, year, month, chunks)
            logger.info(f"Successfully created monthly file: {monthly_file}")
        else:
            logger.warning("No daily files were created, cannot create monthly file")

    except Exception as e:
        logger.error(f"Error in consolidation: {e}")
        raise


def handle_upload(provider: str, year: int, month: int, day: int = None, **kwargs):
    """Handle uploading data to Hugging Face."""
    config_path = Path("config.yaml")
    overwrite = kwargs.get("overwrite", False)
    upload_type = kwargs.get("type", "hourly")  # New parameter to specify upload type

    try:
        if upload_type == "monthly":
            # Upload monthly consolidated file
            logger.info(f"Uploading monthly consolidated file for {year}-{month:02d}")
            upload_monthly_zarr(
                config_path=config_path, year=year, month=month, overwrite=overwrite
            )
        else:
            # Original hourly upload functionality
            logger.info(f"Uploading hourly data for {year}-{month:02d}-{day:02d}")
            upload_to_huggingface(
                config_path=config_path,
                folder_name=f"{year}-{month:02d}-{day:02d}",
                year=year,
                month=month,
                day=day,
                overwrite=overwrite,
            )

    except Exception as e:
        logger.error(f"Error in upload: {e}")
        raise


def archive_to_hf(provider: str, year: int, month: int, day: int = None, **kwargs):
    """Handle archiving data to Hugging Face."""
    overwrite = kwargs.get("overwrite", False)
    archive_type = kwargs.get("archive_type", "zarr.zip")

    try:
        if day is None:
            # Archive monthly consolidated file
            logger.info(f"Archiving monthly consolidated file for {year}-{month:02d}")
            region = kwargs.get("region", "global")

            if provider == "metoffice":
                if region not in CONFIG_PATHS:
                    raise ValueError(f"Invalid region '{region}'. Must be 'uk' or 'global'.")
                config_path = CONFIG_PATHS[region]
            else:
                raise NotImplementedError(
                    f"Monthly archive for provider {provider} not yet implemented"
                )

            upload_monthly_zarr(
                config_path=config_path,
                year=year,
                month=month,
                overwrite=overwrite,
            )
        else:
            # Use provider-specific archive processing for daily data
            handle_archive(
                provider=provider,
                year=year,
                month=month,
                day=day,
                hour=kwargs.get("hour"),
                region=kwargs.get("region", "global"),
                overwrite=overwrite,
                archive_type=archive_type,
            )

    except Exception as e:
        logger.error(f"Error in archive: {e}")
        raise


def handle_sample(
    provider: str,
    year: int,
    month: int,
    day: int,
    region: str = "global",
    chunks: Optional[str] = None,
    output: str = None,
    remote: bool = False,
) -> None:
    """
    Handle the sample operation.

    Args:
        provider: Data provider (e.g., 'metoffice').
        year: Year of data.
        month: Month of data.
        day: Day of data.
        region: Region to process ('uk' or 'global').
        chunks: Optional chunk sizes for dask.
        output: Output path for sampled data.
        remote: Whether to use remote data source.
    """
    logger.info(f"Loading data for {year}-{month:02d}-{day:02d}")

    # Parse chunks if provided
    chunk_dict = None
    if chunks:
        chunk_dict = dict(chunk.split(":") for chunk in chunks.split(","))

    # Load the dataset
    store, ds = load_zarr_data(year, month, day, chunks=chunk_dict, remote=remote)

    try:
        # Get Zarr groups and prepare dataset for OCF
        zarr_groups = get_zarr_groups(store)
        ds_ocf = prepare_nwp_dataset_for_ocf(ds, zarr_groups, store, chunk_dict)

        # Save the prepared dataset
        logger.info(f"Saving prepared dataset to {output}")
        ds_ocf.to_netcdf(output)

    finally:
        # Clean up
        store.close()

    logger.info("Sampling operation completed successfully")


def main():
    """Entry point for the Open Data PVNet CLI tool.

    Examples:
    ---------
    Met Office Data:
        # Archive all hours for a given day with default workers (1)
        open-data-pvnet metoffice archive --year 2023 --month 12 --day 1 --region uk -o

        # Archive all hours for a given day with parallel processing with 4 workers
        open-data-pvnet metoffice archive --year 2023 --month 12 --day 1 --region uk -o --workers 4

        # Archive global region data for a specific hour
        open-data-pvnet metoffice archive --year 2023 --month 12 --day 1 --hour 12 --region uk -o

        # Archive as tar instead of zarr.zip
        open-data-pvnet metoffice archive --year 2023 --month 12 --day 1 --hour 12 --region uk -o --archive-type tar

        # Archive monthly data, this requires a consolidation first
        open-data-pvnet metoffice archive --year 2023 --month 12

        # Consolidate monthly data
        open-data-pvnet metoffice consolidate --year 2023 --month 12

        # Consolidate specific day
        open-data-pvnet metoffice consolidate --year 2023 --month 12 --day 1

        # Sample data for ML training
        open-data-pvnet metoffice sample --year 2023 --month 12 --day 1 --region uk --chunks "time:24,latitude:100" --output /path/to/output

    GFS Data:
        Partially implemented

    DWD Data:
        Not implemented yet

    Loading Data:
        # Load local data with default chunking
        open-data-pvnet metoffice load --year 2023 --month 1 --day 16 --hour 0 --region uk

        # Load remotely without downloading
        open-data-pvnet metoffice load --year 2023 --month 1 --day 16 --hour 0 --region uk --remote

        # Load with custom chunking
        open-data-pvnet metoffice load --year 2023 --month 1 --day 16 --hour 0 --region uk --chunks "time:24,latitude:100,longitude:100"

    List Available Providers:
        open-data-pvnet --list providers
    """
    parser = configure_parser()
    args = parser.parse_args()

    # Handle the --list providers case first
    if args.list == "providers":
        print("Available providers:")
        for provider in PROVIDERS:
            if provider == "gfs":
                print(f"- {provider} (partially implemented)")
            elif provider == "dwd":
                print(f"- {provider} (not implemented)")
            else:
                print(f"- {provider}")
        return 0

    # For all other commands, we need a provider and operation
    if not args.command or not args.operation:
        parser.print_help()
        return 1

    # Load environment variables
    load_env_and_setup_logger()

    # Execute the requested operation
    if args.operation == "load":
        load_kwargs = {
            "provider": args.command,
            "year": args.year,
            "month": args.month,
            "day": args.day,
            "hour": args.hour,
            "region": args.region,
            "overwrite": args.overwrite,
            "chunks": args.chunks,
            "remote": args.remote,
        }
        handle_load(**load_kwargs)
    elif args.operation == "consolidate":
        consolidate_kwargs = {
            "provider": args.command,
            "year": args.year,
            "month": args.month,
            "day": args.day,
            "region": getattr(args, "region", None),
            "overwrite": args.overwrite,
        }
        handle_monthly_consolidation(**consolidate_kwargs)
    elif args.operation == "archive":
        archive_kwargs = {
            "provider": args.command,
            "year": args.year,
            "month": args.month,
            "day": args.day,
            "hour": getattr(args, "hour", None),
            "region": getattr(args, "region", None),
            "overwrite": args.overwrite,
            "archive_type": getattr(args, "archive_type", "zarr.zip"),
        }
        archive_to_hf(**archive_kwargs)
    elif args.operation == "sample":
        handle_sample(
            provider="metoffice",
            year=args.year,
            month=args.month,
            day=args.day,
            region=args.region,
            chunks=args.chunks,
            output=args.output,
            remote=args.remote,
        )

    return 0
