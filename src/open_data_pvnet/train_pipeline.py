import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import hydra
import pandas as pd
import typer
from omegaconf import OmegaConf

# Note: We need to add local pvnet to path before importing it
# This is done at module level to ensure it happens before the import below
_local_pvnet_path = Path.cwd().parent / "pvnet"
if _local_pvnet_path.exists():
    sys.path.insert(0, str(_local_pvnet_path))

from pvnet.training import train as pvnet_train

app = typer.Typer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print path information after all imports
if _local_pvnet_path.exists():
    print(f"Adding local pvnet to path: {_local_pvnet_path}")
else:
    print(f"Local pvnet not found at {_local_pvnet_path}")

if "WANDB_RUN_ID" not in os.environ:
    import datetime

    os.environ["WANDB_RUN_ID"] = datetime.datetime.now().strftime("%y%m%d%H%M%S")


def split_dates(start_date: str, end_date: str, val_split: float, test_split: float):
    """
    Splits a date range into train, validation, and test periods.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    total_days = (end - start).days
    if total_days <= 0:
        raise ValueError("End date must be after start date.")

    test_days = int(total_days * test_split)
    val_days = int(total_days * val_split)

    # Calculate split points
    # Train: start -> val_start
    # Val: val_start -> test_start
    # Test: test_start -> end

    val_start = end - timedelta(days=test_days + val_days)
    test_start = end - timedelta(days=test_days)

    # Define periods as [start, end] inclusive strings
    train_period = [
        start.strftime("%Y-%m-%d"),
        (val_start - timedelta(days=1)).strftime("%Y-%m-%d"),
    ]
    # Ensure validation period is valid
    val_end = test_start - timedelta(days=1)
    if val_end < val_start:
        # If split resulted in 0 days, push boundaries or handle gracefully.
        # For this fix, we will just force non-overlapping or warning.
        # But better: Use the calculated days directly.
        pass

    val_period = [
        val_start.strftime("%Y-%m-%d"),
        (test_start - timedelta(days=1)).strftime("%Y-%m-%d"),
    ]
    test_period = [test_start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]

    # Check for invalid periods
    if pd.to_datetime(train_period[1]) < pd.to_datetime(train_period[0]):
        train_period[1] = train_period[0]

    if pd.to_datetime(val_period[1]) < pd.to_datetime(val_period[0]):
        # Fallback: if period is empty, use the start date as both start and end (1 day overlap) or adjust splits.
        # For now, let's just log a warning and clamp.
        val_period[1] = val_period[0]

    if pd.to_datetime(test_period[1]) < pd.to_datetime(test_period[0]):
        test_period[1] = test_period[0]

    return train_period, val_period, test_period


@app.command()
def main(
    start_date: str = typer.Option(
        ..., help="Start date in YYYY-MM-DD format (e.g. 2023-01-01)"
    ),
    end_date: str = typer.Option(
        ..., help="End date in YYYY-MM-DD format (e.g. 2023-12-31)"
    ),
    val_split: float = typer.Option(0.1, help="Fraction of data to use for validation"),
    test_split: float = typer.Option(0.1, help="Fraction of data to use for testing"),
    config_dir: str = typer.Option(
        os.path.join("src", "open_data_pvnet", "configs", "PVNet_configs"),
        help="Path to PVNet configs directory relative to current working directory",
    ),
    config_name: str = typer.Option(
        "config", help="Name of the config file (without .yaml)"
    ),
    data_configuration: str = typer.Option(
        None, help="Path to ocf-data-sampler configuration.yaml"
    ),
    num_workers: int = typer.Option(4, help="Number of workers for dataloader"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
):
    """
    Train/Test/Split Pipeline for PVNet.

    This script automates the splitting of time-series data into training, validation,
    and testing sets, and then initiates the PVNet training loop.
    """

    logger.info(f"Starting pipeline from {start_date} to {end_date}")

    # 1. Calculate Date Splits
    train_period, val_period, test_period = split_dates(
        start_date, end_date, val_split, test_split
    )

    logger.info(f"Train Period: {train_period}")
    logger.info(f"Val Period:   {val_period}")
    logger.info(f"Test Period:  {test_period}")

    # 2. Setup Hydra Config
    # We need to make sure the config path is absolute or correctly relative
    abs_config_dir = os.path.abspath(config_dir)
    if not os.path.exists(abs_config_dir):
        logger.error(f"Config directory not found: {abs_config_dir}")
        raise FileNotFoundError(f"Config directory not found: {abs_config_dir}")

    # Initialize Hydra
    # Note: hydra.initialize_config_dir requires an absolute path
    with hydra.initialize_config_dir(version_base="1.2", config_dir=abs_config_dir):
        # Compose the configuration
        # We process overrides here
        overrides = [
            f"datamodule.train_period=[{train_period[0]},{train_period[1]}]",
            f"datamodule.val_period=[{val_period[0]},{val_period[1]}]",
            f"datamodule.num_workers={num_workers}",
            f"datamodule.batch_size={batch_size}",
            "trainer.max_epochs=1",
            "+trainer.limit_train_batches=5",
            "+trainer.limit_val_batches=2",
            "+trainer.limit_test_batches=2",
            "+trainer.enable_progress_bar=false",
            "logger.wandb.offline=true",
            "logger.wandb.log_model=false",
            f"work_dir={os.getcwd()}",
        ]

        if data_configuration:
            overrides.append(f"datamodule.configuration={data_configuration}")

        cfg = hydra.compose(config_name=config_name, overrides=overrides)

        # Resolve config to ensure specific variables are interpolated
        OmegaConf.resolve(cfg)

        logger.info("Configuration loaded and resolved.")

        if not cfg.datamodule.configuration:
            logger.warning(
                "No data configuration specified! Use --data-configuration to point to an ocf-data-sampler config."
            )

        # 3. Run Training
        logger.info("Initializing PVNet training...")
        pvnet_train(cfg)
        logger.info("Training complete.")


if __name__ == "__main__":
    app()