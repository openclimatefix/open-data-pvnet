"""
PVNet Evaluation Module

This module provides utilities for evaluating trained PVNet models including:
- Metrics computation (MAE, RMSE, Pinball Loss, CRPS)
- Visualization utilities for diagnostic plots
"""

from .metrics import (
    compute_mae,
    compute_rmse,
    compute_pinball_loss,
    compute_crps,
    compute_coverage,
    MetricsAccumulator,
)

from .visualization import (
    plot_mae_vs_horizon,
    plot_scatter,
    plot_reliability_diagram,
    plot_coverage,
)

__all__ = [
    # Metrics
    "compute_mae",
    "compute_rmse",
    "compute_pinball_loss",
    "compute_crps",
    "compute_coverage",
    "MetricsAccumulator",
    # Visualization
    "plot_mae_vs_horizon",
    "plot_scatter",
    "plot_reliability_diagram",
    "plot_coverage",
]
