"""
Quadratic Weighted Kappa (QWK) threshold optimization.

For ordinal classification, QWK penalizes predictions farther from
the true value. If the competition uses QWK, optimizing ordinal
thresholds on stacked probabilities can significantly improve scores.

This implements the approach from Kim (MD)'s notebook: optimize
the boundaries between ESI classes on the probability simplex.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute QWK between true and predicted ordinal labels."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _apply_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Convert class probabilities to ordinal predictions using thresholds.

    Instead of argmax, compute expected value (weighted by probabilities)
    and apply ordinal thresholds to bin into classes 1-5.
    """
    # Expected ESI value (probability-weighted)
    classes = np.arange(1, proba.shape[1] + 1)
    expected = proba @ classes

    # Apply thresholds: boundaries between classes
    # thresholds = [t1, t2, t3, t4] where t1 < t2 < t3 < t4
    preds = np.ones(len(expected), dtype=int)
    for i, t in enumerate(sorted(thresholds)):
        preds[expected > t] = i + 2  # class 2, 3, 4, 5

    return preds


def optimize_thresholds(proba: np.ndarray,
                        y_true: np.ndarray,
                        ) -> tuple[np.ndarray, float]:
    """Find optimal ordinal thresholds to maximize QWK.

    Args:
        proba: (n_samples, 5) probability matrix
        y_true: (n_samples,) true ESI labels (1-5)

    Returns:
        (optimal_thresholds, best_qwk)
    """
    def neg_qwk(thresholds):
        preds = _apply_thresholds(proba, thresholds)
        return -quadratic_weighted_kappa(y_true, preds)

    # Initial thresholds: evenly spaced at 1.5, 2.5, 3.5, 4.5
    x0 = np.array([1.5, 2.5, 3.5, 4.5])

    # Try multiple starting points
    best_result = None
    best_score = -1.0

    for offsets in [
        [0, 0, 0, 0],
        [-0.2, -0.1, 0.1, 0.2],
        [0.2, 0.1, -0.1, -0.2],
        [-0.3, 0, 0, 0.3],
    ]:
        x_start = x0 + np.array(offsets)
        result = minimize(
            neg_qwk,
            x_start,
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 0.001, "fatol": 0.0001},
        )
        score = -result.fun
        if score > best_score:
            best_score = score
            best_result = result

    optimal = np.sort(best_result.x)
    print(f"QWK optimization: {best_score:.4f}")
    print(f"  Thresholds: {optimal}")

    return optimal, best_score


def predict_with_thresholds(proba: np.ndarray,
                            thresholds: np.ndarray) -> np.ndarray:
    """Apply optimized thresholds to probabilities."""
    return _apply_thresholds(proba, thresholds)
