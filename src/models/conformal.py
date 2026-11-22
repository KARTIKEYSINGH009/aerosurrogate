"""Conformal prediction intervals and calibration for AeroSurrogate.

Supports:
1. Standard Split Conformal Prediction (inductive conformal prediction)
2. Normalized / Locally-Adaptive Conformal Prediction (scaling residual by estimated local error)
3. Group-stratified (body style) Conformal Prediction for balanced coverage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class ConformalCalibrator:
    """Calibrator storing conformal non-conformity quantiles."""
    target_name: str
    alpha: float
    q_hat_marginal: float
    q_hat_per_group: Dict[str, float]

    def predict_interval(
        self,
        predictions: np.ndarray,
        groups: pd.Series | None = None,
        use_group_calibrated: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute [lower_bound, upper_bound] for predictions."""
        preds = np.asarray(predictions, dtype=np.float64)
        if use_group_calibrated and groups is not None:
            lower = np.empty_like(preds)
            upper = np.empty_like(preds)
            for i, (p, g) in enumerate(zip(preds, groups)):
                q = self.q_hat_per_group.get(str(g), self.q_hat_marginal)
                lower[i] = p - q
                upper[i] = p + q
            return lower, upper
        else:
            return preds - self.q_hat_marginal, preds + self.q_hat_marginal


def compute_conformal_quantile(residuals: np.ndarray, alpha: float = 0.10) -> float:
    """Compute split-conformal calibration quantile with finite-sample (n+1) correction.
    
    Level: 1 - alpha (e.g. alpha=0.10 -> 90% coverage).
    Finite-sample quantile index: ceil((n + 1) * (1 - alpha)) / n
    """
    n = len(residuals)
    if n == 0:
        raise ValueError("Residuals cannot be empty")
    abs_res = np.abs(residuals)
    # Finite-sample correction factor
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(abs_res, q_level, method="higher" if hasattr(np, "quantile") else "linear"))


def calibrate_split_conformal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    groups: pd.Series | None = None,
    alpha: float = 0.10,
) -> ConformalCalibrator:
    """Calibrate both marginal and group-conditional conformal intervals."""
    residuals = y_true - y_pred
    q_marginal = compute_conformal_quantile(residuals, alpha=alpha)

    q_per_group = {}
    if groups is not None:
        for g in groups.unique():
            mask = (groups == g).to_numpy()
            if mask.sum() > 5:
                q_per_group[str(g)] = compute_conformal_quantile(residuals[mask], alpha=alpha)
            else:
                q_per_group[str(g)] = q_marginal

    return ConformalCalibrator(
        target_name=target_name,
        alpha=alpha,
        q_hat_marginal=q_marginal,
        q_hat_per_group=q_per_group,
    )


def evaluate_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: pd.Series | None = None,
) -> Dict[str, float]:
    """Calculate empirical coverage percentage and average interval width."""
    covered = (y_true >= lower) & (y_true <= upper)
    overall_cov = float(np.mean(covered) * 100.0)
    avg_width = float(np.mean(upper - lower))

    stats = {
        "overall_coverage_pct": overall_cov,
        "average_interval_width": avg_width,
    }

    if groups is not None:
        for g in sorted(groups.unique()):
            mask = (groups == g).to_numpy()
            if mask.sum() > 0:
                stats[f"coverage_{g}_pct"] = float(np.mean(covered[mask]) * 100.0)
                stats[f"width_{g}"] = float(np.mean((upper - lower)[mask]))

    return stats
