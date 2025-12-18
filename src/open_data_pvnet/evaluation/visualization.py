"""
Visualization Module for PVNet Evaluation

This module provides plotting functions for diagnostic visualizations
of probabilistic forecast performance.

Plots included:
- MAE vs Forecast Horizon
- Predicted vs True Scatter Plot
- Reliability Diagram
- Coverage Plot
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

# Use non-interactive backend for headless environments
matplotlib.use("Agg")

# Set default style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_mae_vs_horizon(
    horizon_metrics: Dict[str, List[float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "MAE vs Forecast Horizon",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Plot MAE as a function of forecast horizon.

    This plot shows how forecast error grows with lead time. Steeper slopes
    indicate faster skill degradation. Flat or decreasing trends may indicate
    issues with the data or model.

    Args:
        horizon_metrics: Dictionary with 'horizon_idx' or 'horizon_minutes' and 'mae' keys
        output_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size (width, height) in inches

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine x-axis values
    if "horizon_minutes" in horizon_metrics:
        x = horizon_metrics["horizon_minutes"]
        xlabel = "Forecast Horizon (minutes)"
    else:
        x = horizon_metrics["horizon_idx"]
        xlabel = "Forecast Horizon (step index)"

    mae = horizon_metrics["mae"]

    # Plot MAE
    ax.plot(x, mae, "o-", color="#2196F3", linewidth=2, markersize=6, label="MAE")

    # Add RMSE if available
    if "rmse" in horizon_metrics:
        rmse = horizon_metrics["rmse"]
        ax.plot(
            x, rmse, "s--", color="#FF5722", linewidth=2, markersize=6, label="RMSE"
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Set reasonable y-axis limits
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_scatter(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Predicted vs Actual",
    figsize: tuple = (8, 8),
    max_points: int = 5000,
) -> plt.Figure:
    """
    Create a scatter plot of predicted vs actual values.

    Points on the diagonal indicate perfect predictions. Systematic offset
    from the diagonal indicates bias. Wide spread indicates high variance.

    Args:
        predictions: 1D array of predicted values (median/point forecast)
        targets: 1D array of actual values
        output_path: Optional path to save the figure
        title: Plot title
        figsize: Figure size
        max_points: Maximum points to plot (for performance)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # Subsample if too many points
    if len(pred_flat) > max_points:
        idx = np.random.choice(len(pred_flat), max_points, replace=False)
        pred_flat = pred_flat[idx]
        target_flat = target_flat[idx]

    # Create scatter plot with alpha
    ax.scatter(
        target_flat, pred_flat, alpha=0.3, s=10, color="#2196F3", edgecolors="none"
    )

    # Add perfect prediction line
    min_val = min(target_flat.min(), pred_flat.min())
    max_val = max(target_flat.max(), pred_flat.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect forecast",
    )

    # Add trend line
    z = np.polyfit(target_flat, pred_flat, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min_val, max_val, 100)
    ax.plot(
        x_line,
        p(x_line),
        "g-",
        linewidth=1.5,
        alpha=0.8,
        label=f"Trend (y={z[0]:.2f}x+{z[1]:.2f})",
    )

    ax.set_xlabel("Actual Value", fontsize=12)
    ax.set_ylabel("Predicted Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_reliability_diagram(
    coverage: Dict[str, float],
    quantiles: List[float],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Reliability Diagram",
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    Create a reliability diagram showing calibration of quantile forecasts.

    A well-calibrated model will have points on the diagonal. Points above
    the diagonal indicate over-prediction (actuals fall below predictions
    more often than expected). Points below indicate under-prediction.

    Args:
        coverage: Dictionary mapping 'coverage_X.XX' to observed coverage
        quantiles: List of quantile values
        output_path: Optional path to save
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract coverage values
    observed = []
    expected = []

    for q in quantiles:
        key = f"coverage_{q:.2f}"
        if key in coverage:
            expected.append(q)
            observed.append(coverage[key])

    # Plot points
    ax.scatter(
        expected,
        observed,
        s=100,
        c="#2196F3",
        zorder=5,
        edgecolors="white",
        linewidth=2,
    )

    # Connect with line
    ax.plot(expected, observed, "-", color="#2196F3", linewidth=2, alpha=0.7)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration")

    # Fill regions
    ax.fill_between(
        [0, 1], [0, 1], [0, 0], alpha=0.1, color="red", label="Under-confident"
    )
    ax.fill_between(
        [0, 1], [0, 1], [1, 1], alpha=0.1, color="blue", label="Over-confident"
    )

    ax.set_xlabel("Nominal Quantile", fontsize=12)
    ax.set_ylabel("Observed Coverage", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def plot_coverage(
    coverage: Dict[str, float],
    quantiles: List[float],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Quantile Coverage",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create a bar chart comparing expected vs observed coverage per quantile.

    This provides a quick visual check of calibration. Large gaps between
    expected and observed coverage indicate miscalibration.

    Args:
        coverage: Dictionary mapping 'coverage_X.XX' to observed coverage
        quantiles: List of quantile values
        output_path: Optional path to save
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract values
    expected = []
    observed = []
    labels = []

    for q in quantiles:
        key = f"coverage_{q:.2f}"
        if key in coverage:
            expected.append(q)
            observed.append(coverage[key])
            labels.append(f"q{q:.2f}")

    x = np.arange(len(labels))
    width = 0.35

    # Create bars
    _bars1 = ax.bar(
        x - width / 2,
        expected,
        width,
        label="Expected",
        color="#BBDEFB",
        edgecolor="#2196F3",
    )
    bars2 = ax.bar(
        x + width / 2,
        observed,
        width,
        label="Observed",
        color="#2196F3",
        edgecolor="#1565C0",
    )

    ax.set_xlabel("Quantile", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    return fig


def generate_all_plots(
    metrics: Dict[str, float],
    horizon_metrics: Dict[str, List[float]],
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: List[float],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """
    Generate all diagnostic plots and save to output directory.

    Args:
        metrics: Overall metrics dictionary (includes coverage values)
        horizon_metrics: Per-horizon metrics
        predictions: Model predictions (median), shape (n_samples, horizon)
        targets: Ground truth, shape (n_samples, horizon)
        quantiles: List of quantile values
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot name to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_plots = {}

    # 1. MAE vs Horizon
    path = output_dir / "mae_vs_horizon.png"
    plot_mae_vs_horizon(horizon_metrics, output_path=path)
    saved_plots["mae_vs_horizon"] = path
    plt.close()

    # 2. Scatter plot
    path = output_dir / "scatter.png"
    plot_scatter(predictions, targets, output_path=path)
    saved_plots["scatter"] = path
    plt.close()

    # 3. Reliability diagram
    path = output_dir / "reliability_diagram.png"
    coverage = {k: v for k, v in metrics.items() if k.startswith("coverage_")}
    plot_reliability_diagram(coverage, quantiles, output_path=path)
    saved_plots["reliability_diagram"] = path
    plt.close()

    # 4. Coverage bar chart
    path = output_dir / "coverage.png"
    plot_coverage(coverage, quantiles, output_path=path)
    saved_plots["coverage"] = path
    plt.close()

    return saved_plots
