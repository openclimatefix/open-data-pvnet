"""
PVNet Evaluation Pipeline

A production-ready evaluation script for PVNet-based GSP forecasting models.
This script loads a trained Lightning checkpoint, runs inference on the test
set, computes metrics (MAE, RMSE, Pinball Loss, CRPS), and generates diagnostic
plots.

Usage:
    python -m open_data_pvnet.evaluate_pipeline \
        --checkpoint path/to/checkpoint.ckpt \
        --data-config path/to/data_configuration.yaml \
        --output-dir ./eval_output

Features:
    - Works on CPU by default
    - Supports probabilistic evaluation (quantile forecasts)
    - Generates reproducible metrics and visualizations
    - Compatible with W&B offline logging
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import typer
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Typer CLI app
app = typer.Typer(
    name="evaluate", help="Evaluate a trained PVNet model on held-out test data."
)


def load_model_from_checkpoint(
    checkpoint_path: Path, device: str = "cpu"
) -> torch.nn.Module:
    """
    Load a PVNet model from a Lightning checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded model in eval mode
    """
    from pvnet.models.multimodal.multimodal import Model

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    model = Model.load_from_checkpoint(
        checkpoint_path, map_location=device, strict=False, weights_only=False
    )

    model.eval()
    model.to(device)

    logger.info(f"Model loaded successfully. Device: {device}")

    return model


def create_test_dataloader(
    data_config_path: Path,
    test_period: Tuple[str, str],
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Create a DataLoader for the test period.

    Args:
        data_config_path: Path to ocf-data-sampler configuration
        test_period: Tuple of (start_date, end_date) strings
        batch_size: Batch size for evaluation
        num_workers: Number of dataloader workers (0 for CPU)

    Returns:
        PyTorch DataLoader for test data
    """
    from pvnet.data import DataModule

    logger.info(f"Creating DataModule with test period: {test_period}")

    datamodule = DataModule(
        configuration=str(data_config_path),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=None,
    )

    # Set test period
    datamodule.val_period = list(test_period)

    # Setup for validation (we use val as test to avoid needing separate config)
    datamodule.setup(stage="fit")

    return datamodule.val_dataloader()


def extract_predictions_and_targets(
    batch: Dict, model: torch.nn.Module, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run model inference and extract predictions and targets from a batch.

    Args:
        batch: Data batch from DataLoader
        model: Trained PVNet model
        device: Device to run inference on

    Returns:
        Tuple of (predictions, targets) tensors with aligned shapes
    """
    # Move batch to device
    batch_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        elif isinstance(value, dict):
            batch_device[key] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
        else:
            batch_device[key] = value

    # Run inference
    with torch.no_grad():
        predictions = model(batch_device)

    # Extract targets - common key names in PVNet
    target_keys = ["gsp", "gsp_future", "gsp_yield", "gsp_targets"]
    targets = None

    for key in target_keys:
        if key in batch_device:
            targets = batch_device[key]
            break

    if targets is None:
        # Try to find any key containing 'target' or 'future'
        for key in batch_device.keys():
            if "target" in key.lower() or "future" in key.lower():
                targets = batch_device[key]
                break

    if targets is None:
        raise KeyError(
            f"Could not find targets in batch. Keys: {list(batch_device.keys())}"
        )

    # Align shapes: predictions (batch, horizon, quantiles), targets (batch, horizon)
    # The model may output fewer horizons than the data provides
    pred_horizon = predictions.shape[1]

    if targets.dim() == 3:
        target_horizon = targets.shape[1]
    else:
        target_horizon = targets.shape[1] if targets.dim() == 2 else targets.shape[0]

    # Slice to minimum horizon
    min_horizon = min(pred_horizon, target_horizon)
    predictions = predictions[:, :min_horizon, :]

    if targets.dim() == 3:
        targets = targets[:, :min_horizon, :]
    elif targets.dim() == 2:
        targets = targets[:, :min_horizon]

    # Convert targets to same dtype as predictions for metric computation
    targets = targets.float()

    return predictions, targets


def run_evaluation(
    model: torch.nn.Module,
    dataloader,
    quantiles: List[float],
    device: str = "cpu",
    limit_batches: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, List[float]], torch.Tensor, torch.Tensor]:
    """
    Run evaluation loop and compute metrics.

    Args:
        model: Trained model in eval mode
        dataloader: Test dataloader
        quantiles: List of quantile values
        device: Device for inference
        limit_batches: Optional limit on number of batches (for quick testing)

    Returns:
        Tuple of (overall_metrics, horizon_metrics, all_predictions, all_targets)
    """
    from open_data_pvnet.evaluation.metrics import MetricsAccumulator

    logger.info("Starting evaluation loop...")

    accumulator = MetricsAccumulator(quantiles=quantiles)

    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(dataloader):
        if limit_batches and batch_idx >= limit_batches:
            logger.info(
                f"Stopping at batch {batch_idx} (limit_batches={limit_batches})"
            )
            break

        try:
            predictions, targets = extract_predictions_and_targets(batch, model, device)

            # Accumulate metrics
            accumulator.update(predictions, targets)

            # Store for visualization
            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed batch {batch_idx + 1}")

        except Exception as e:
            logger.warning(f"Error processing batch {batch_idx}: {e}")
            continue

    # Compute final metrics
    overall_metrics = accumulator.compute()
    horizon_metrics = accumulator.compute_per_horizon()

    # Concatenate all predictions and targets
    if all_preds:
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
    else:
        raise RuntimeError("No batches were successfully processed")

    logger.info(f"Evaluation complete. Processed {accumulator._n_samples} samples.")

    return overall_metrics, horizon_metrics, all_preds_tensor, all_targets_tensor


def save_results(
    metrics: Dict[str, float],
    horizon_metrics: Dict[str, List[float]],
    output_dir: Path,
    config_snapshot: Optional[Dict] = None,
) -> None:
    """
    Save evaluation results to files.

    Args:
        metrics: Overall metrics dictionary
        horizon_metrics: Per-horizon metrics
        output_dir: Directory to save results
        config_snapshot: Optional config to save for reproducibility
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save overall metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to: {metrics_path}")

    # Save horizon metrics
    horizon_df = pd.DataFrame(horizon_metrics)
    horizon_path = output_dir / "horizon_metrics.csv"
    horizon_df.to_csv(horizon_path, index=False)
    logger.info(f"Saved horizon metrics to: {horizon_path}")

    # Save config snapshot
    if config_snapshot:
        config_path = output_dir / "config_snapshot.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_snapshot, f, default_flow_style=False)
        logger.info(f"Saved config snapshot to: {config_path}")


def generate_plots(
    metrics: Dict[str, float],
    horizon_metrics: Dict[str, List[float]],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float],
    output_dir: Path,
) -> None:
    """
    Generate and save diagnostic plots.

    Args:
        metrics: Overall metrics (includes coverage)
        horizon_metrics: Per-horizon metrics
        predictions: All predictions tensor
        targets: All targets tensor
        quantiles: List of quantile values
        output_dir: Directory to save plots
    """
    from open_data_pvnet.evaluation.visualization import generate_all_plots

    plots_dir = output_dir / "plots"

    # Extract median for scatter plot
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    median_preds = predictions[..., median_idx].numpy()

    # Ensure targets is 2D
    if targets.dim() == 3:
        targets = targets.squeeze(-1)
    targets_np = targets.numpy()

    saved_plots = generate_all_plots(
        metrics=metrics,
        horizon_metrics=horizon_metrics,
        predictions=median_preds,
        targets=targets_np,
        quantiles=quantiles,
        output_dir=plots_dir,
    )

    logger.info(f"Generated {len(saved_plots)} plots in: {plots_dir}")


def log_to_wandb(
    metrics: Dict[str, float],
    horizon_metrics: Dict[str, List[float]],
    output_dir: Path,
    run_name: Optional[str] = None,
) -> None:
    """
    Log results to Weights & Biases (offline mode compatible).

    Args:
        metrics: Overall metrics
        horizon_metrics: Per-horizon metrics
        output_dir: Directory containing artifacts
        run_name: Optional W&B run name
    """
    try:
        import wandb

        # Check if W&B is available and configured
        if os.environ.get("WANDB_MODE") != "disabled":
            wandb.init(
                project="pvnet-evaluation",
                name=run_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={"output_dir": str(output_dir)},
                mode=os.environ.get("WANDB_MODE", "offline"),
            )

            # Log metrics
            wandb.log(metrics)

            # Log plots as images
            plots_dir = output_dir / "plots"
            if plots_dir.exists():
                for plot_file in plots_dir.glob("*.png"):
                    wandb.log({plot_file.stem: wandb.Image(str(plot_file))})

            wandb.finish()
            logger.info("Logged results to W&B")

    except ImportError:
        logger.warning("wandb not installed, skipping W&B logging")
    except Exception as e:
        logger.warning(f"Failed to log to W&B: {e}")


@app.command()
def evaluate(
    checkpoint: str = typer.Option(
        ..., "--checkpoint", "-c", help="Path to Lightning checkpoint (.ckpt file)"
    ),
    data_config: str = typer.Option(
        ..., "--data-config", "-d", help="Path to ocf-data-sampler configuration YAML"
    ),
    output_dir: str = typer.Option(
        "./eval_output",
        "--output-dir",
        "-o",
        help="Directory to save evaluation results",
    ),
    test_start: str = typer.Option(
        "2023-10-01", "--test-start", help="Test period start date (YYYY-MM-DD)"
    ),
    test_end: str = typer.Option(
        "2023-12-31", "--test-end", help="Test period end date (YYYY-MM-DD)"
    ),
    batch_size: int = typer.Option(
        32, "--batch-size", "-b", help="Batch size for evaluation"
    ),
    limit_batches: Optional[int] = typer.Option(
        None, "--limit-batches", help="Limit number of batches for quick testing"
    ),
    device: str = typer.Option(
        "cpu", "--device", help="Device to run evaluation on ('cpu' or 'cuda')"
    ),
    quantiles: str = typer.Option(
        "0.02,0.1,0.25,0.5,0.75,0.9,0.98",
        "--quantiles",
        help="Comma-separated list of quantile values",
    ),
    wandb_log: bool = typer.Option(
        False, "--wandb/--no-wandb", help="Log results to Weights & Biases"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
) -> None:
    """
    Evaluate a trained PVNet model on held-out test data.

    This command loads a checkpoint, runs inference on the test set,
    computes comprehensive metrics, and generates diagnostic plots.
    """
    import lightning as L

    # Set seed for reproducibility
    L.seed_everything(seed, workers=True)

    # Parse paths
    checkpoint_path = Path(checkpoint)
    data_config_path = Path(data_config)
    output_path = Path(output_dir)

    # Parse quantiles
    quantile_list = [float(q.strip()) for q in quantiles.split(",")]

    # Validate inputs
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise typer.Exit(1)

    if not data_config_path.exists():
        logger.error(f"Data config not found: {data_config_path}")
        raise typer.Exit(1)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_path / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PVNet Evaluation Pipeline")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Data config: {data_config_path}")
    logger.info(f"Test period: {test_start} to {test_end}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Device: {device}")
    logger.info(f"Quantiles: {quantile_list}")
    logger.info("=" * 60)

    # Step 1: Load model
    model = load_model_from_checkpoint(checkpoint_path, device)

    # Step 2: Create test dataloader
    test_dataloader = create_test_dataloader(
        data_config_path,
        test_period=(test_start, test_end),
        batch_size=batch_size,
        num_workers=0,  # CPU-safe
    )

    # Step 3: Run evaluation
    overall_metrics, horizon_metrics, all_preds, all_targets = run_evaluation(
        model=model,
        dataloader=test_dataloader,
        quantiles=quantile_list,
        device=device,
        limit_batches=limit_batches,
    )

    # Step 4: Save results
    config_snapshot = {
        "checkpoint": str(checkpoint_path),
        "data_config": str(data_config_path),
        "test_period": [test_start, test_end],
        "batch_size": batch_size,
        "quantiles": quantile_list,
        "seed": seed,
    }
    save_results(overall_metrics, horizon_metrics, output_path, config_snapshot)

    # Step 5: Generate plots
    generate_plots(
        metrics=overall_metrics,
        horizon_metrics=horizon_metrics,
        predictions=all_preds,
        targets=all_targets,
        quantiles=quantile_list,
        output_dir=output_path,
    )

    # Step 6: Log to W&B (optional)
    if wandb_log:
        log_to_wandb(overall_metrics, horizon_metrics, output_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Samples evaluated: {overall_metrics.get('n_samples', 'N/A')}")
    logger.info(f"MAE: {overall_metrics.get('mae', 'N/A'):.4f}")
    logger.info(f"RMSE: {overall_metrics.get('rmse', 'N/A'):.4f}")
    logger.info(f"CRPS: {overall_metrics.get('crps', 'N/A'):.4f}")
    logger.info(
        f"Pinball (overall): {overall_metrics.get('pinball_overall', 'N/A'):.4f}"
    )
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    app()
