"""
Metrics Module for PVNet Evaluation

This module provides functions for computing evaluation metrics for probabilistic
and deterministic forecasts. All metrics are designed to work with batched PyTorch
tensors and support per-horizon aggregation.

Metrics included:
- Point forecast: MAE, RMSE
- Probabilistic: Pinball Loss, CRPS, Coverage
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, field


def compute_mae(
    predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute Mean Absolute Error.

    Args:
        predictions: Predicted values, shape (batch, horizon) or (batch, horizon, 1)
        targets: Ground truth values, shape (batch, horizon) or (batch, horizon, 1)
        reduction: 'mean', 'none', or 'horizon' for per-horizon mean

    Returns:
        MAE value (scalar if reduction='mean', tensor otherwise)
    """
    # Ensure same shape
    predictions = predictions.squeeze(-1) if predictions.dim() == 3 else predictions
    targets = targets.squeeze(-1) if targets.dim() == 3 else targets

    errors = torch.abs(predictions - targets)

    if reduction == "mean":
        return errors.mean()
    elif reduction == "horizon":
        # Average across batch dimension, keep horizon
        return errors.mean(dim=0)
    else:  # 'none'
        return errors


def compute_rmse(
    predictions: torch.Tensor, targets: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute Root Mean Squared Error.

    Args:
        predictions: Predicted values, shape (batch, horizon)
        targets: Ground truth values, shape (batch, horizon)
        reduction: 'mean', 'none', or 'horizon' for per-horizon mean

    Returns:
        RMSE value (scalar if reduction='mean', tensor otherwise)
    """
    predictions = predictions.squeeze(-1) if predictions.dim() == 3 else predictions
    targets = targets.squeeze(-1) if targets.dim() == 3 else targets

    squared_errors = (predictions - targets) ** 2

    if reduction == "mean":
        return torch.sqrt(squared_errors.mean())
    elif reduction == "horizon":
        return torch.sqrt(squared_errors.mean(dim=0))
    else:  # 'none'
        return torch.sqrt(squared_errors)


def compute_pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    quantiles: List[float],
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """
    Compute Pinball (Quantile) Loss for multiple quantiles.

    The pinball loss for quantile τ is:
        L_τ(y, q) = τ(y - q) if y >= q else (1-τ)(q - y)

    Args:
        predictions: Quantile predictions, shape (batch, horizon, num_quantiles)
        targets: Ground truth values, shape (batch, horizon) or (batch, horizon, 1)
        quantiles: List of quantile values (e.g., [0.1, 0.5, 0.9])
        reduction: 'mean' for overall mean, 'quantile' for per-quantile mean

    Returns:
        Dictionary with 'overall' and per-quantile losses
    """
    # Expand targets to match predictions shape
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)  # (batch, horizon, 1)

    results = {}
    all_losses = []

    for i, tau in enumerate(quantiles):
        pred_q = predictions[..., i]  # (batch, horizon)
        target = targets.squeeze(-1)  # (batch, horizon)

        # Compute pinball loss
        error = target - pred_q
        loss = torch.where(error >= 0, tau * error, (tau - 1) * error)

        if reduction == "mean":
            results[f"pinball_{tau:.2f}"] = loss.mean()
        else:
            results[f"pinball_{tau:.2f}"] = loss

        all_losses.append(loss)

    # Overall pinball loss (average across all quantiles)
    stacked = torch.stack(all_losses, dim=-1)
    results["pinball_overall"] = stacked.mean()

    return results


def compute_crps(
    predictions: torch.Tensor, targets: torch.Tensor, quantiles: List[float]
) -> torch.Tensor:
    """
    Compute Continuous Ranked Probability Score (CRPS) from quantile predictions.

    Uses the quantile approximation of CRPS as the integral of pinball losses.
    CRPS generalizes MAE to probabilistic forecasts.

    Args:
        predictions: Quantile predictions, shape (batch, horizon, num_quantiles)
        targets: Ground truth values, shape (batch, horizon)
        quantiles: List of quantile values corresponding to predictions

    Returns:
        Mean CRPS value
    """
    # Expand targets
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)

    quantiles_tensor = torch.tensor(quantiles, device=predictions.device)

    # Compute pinball loss for each quantile
    errors = targets - predictions  # (batch, horizon, num_quantiles)
    pinball = torch.where(
        errors >= 0, quantiles_tensor * errors, (quantiles_tensor - 1) * errors
    )

    # CRPS is approximated as 2 * mean(pinball losses)
    # The factor of 2 comes from the integration over [0,1]
    crps = 2 * pinball.mean()

    return crps


def compute_coverage(
    predictions: torch.Tensor, targets: torch.Tensor, quantiles: List[float]
) -> Dict[str, float]:
    """
    Compute coverage probability for each quantile.

    Coverage for quantile τ is the fraction of observations that fall
    below the τ-th quantile prediction. Ideally, coverage(τ) = τ.

    Args:
        predictions: Quantile predictions, shape (batch, horizon, num_quantiles)
        targets: Ground truth values, shape (batch, horizon)
        quantiles: List of quantile values

    Returns:
        Dictionary mapping quantile to observed coverage
    """
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)

    results = {}

    for i, tau in enumerate(quantiles):
        pred_q = predictions[..., i : i + 1]  # (batch, horizon, 1)
        # Coverage = fraction of targets below predicted quantile
        below = (targets <= pred_q).float()
        coverage = below.mean().item()
        results[f"coverage_{tau:.2f}"] = coverage

    return results


@dataclass
class MetricsAccumulator:
    """
    Accumulates metrics across batches for final aggregation.

    This class handles batch-wise metric accumulation and provides
    methods for computing final aggregated metrics.

    Attributes:
        quantiles: List of quantile values the model predicts
        horizon_minutes: List of forecast horizon values in minutes
    """

    quantiles: List[float] = field(
        default_factory=lambda: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    )
    horizon_minutes: Optional[List[int]] = None

    # Accumulators
    _predictions: List[torch.Tensor] = field(default_factory=list)
    _targets: List[torch.Tensor] = field(default_factory=list)
    _n_samples: int = 0

    def __post_init__(self):
        """Initialize accumulators."""
        self._predictions = []
        self._targets = []
        self._n_samples = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Add batch predictions and targets to accumulator.

        Args:
            predictions: Model output, shape (batch, horizon, num_quantiles)
            targets: Ground truth, shape (batch, horizon) or (batch, horizon, 1)
        """
        # Move to CPU and detach from computation graph
        self._predictions.append(predictions.detach().cpu())
        self._targets.append(targets.detach().cpu())
        self._n_samples += predictions.shape[0]

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated data.

        Returns:
            Dictionary containing all computed metrics
        """
        if not self._predictions:
            return {}

        # Concatenate all batches
        all_preds = torch.cat(self._predictions, dim=0)
        all_targets = torch.cat(self._targets, dim=0)

        # Ensure targets are 2D
        if all_targets.dim() == 3:
            all_targets = all_targets.squeeze(-1)

        results = {}

        # Find median index for point forecast metrics
        median_idx = (
            self.quantiles.index(0.5)
            if 0.5 in self.quantiles
            else len(self.quantiles) // 2
        )
        point_forecast = all_preds[..., median_idx]

        # Point forecast metrics
        results["mae"] = compute_mae(point_forecast, all_targets).item()
        results["rmse"] = compute_rmse(point_forecast, all_targets).item()

        # Probabilistic metrics
        pinball_results = compute_pinball_loss(all_preds, all_targets, self.quantiles)
        for k, v in pinball_results.items():
            results[k] = v.item() if torch.is_tensor(v) else v

        # CRPS
        results["crps"] = compute_crps(all_preds, all_targets, self.quantiles).item()

        # Coverage
        coverage_results = compute_coverage(all_preds, all_targets, self.quantiles)
        results.update(coverage_results)

        # Metadata
        results["n_samples"] = self._n_samples

        return results

    def compute_per_horizon(self) -> Dict[str, List[float]]:
        """
        Compute metrics broken down by forecast horizon.

        Returns:
            Dictionary with per-horizon metric lists
        """
        if not self._predictions:
            return {}

        all_preds = torch.cat(self._predictions, dim=0)
        all_targets = torch.cat(self._targets, dim=0)

        if all_targets.dim() == 3:
            all_targets = all_targets.squeeze(-1)

        median_idx = (
            self.quantiles.index(0.5)
            if 0.5 in self.quantiles
            else len(self.quantiles) // 2
        )
        point_forecast = all_preds[..., median_idx]

        # Per-horizon MAE and RMSE
        mae_per_horizon = compute_mae(point_forecast, all_targets, reduction="horizon")
        rmse_per_horizon = compute_rmse(
            point_forecast, all_targets, reduction="horizon"
        )

        n_horizons = mae_per_horizon.shape[0]

        results = {
            "horizon_idx": list(range(n_horizons)),
            "mae": mae_per_horizon.tolist(),
            "rmse": rmse_per_horizon.tolist(),
        }

        if self.horizon_minutes is not None:
            results["horizon_minutes"] = self.horizon_minutes[:n_horizons]

        return results

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._predictions = []
        self._targets = []
        self._n_samples = 0
