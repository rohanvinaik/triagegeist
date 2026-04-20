"""Tests for src.qwk_optimizer — kill-targeted per LintGate prescriptions."""
from __future__ import annotations

import numpy as np
import pytest

from src.qwk_optimizer import (
    _apply_thresholds, optimize_thresholds, predict_with_thresholds,
    quadratic_weighted_kappa,
)


# ---------- quadratic_weighted_kappa ----------

def test_qwk_perfect_agreement():
    """Perfect agreement → kappa=1.0."""
    y = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)


def test_qwk_swap_args_not_commutative_in_sign():
    """QWK IS commutative in y_true/y_pred for the quadratic weights."""
    y1 = np.array([1, 2, 3, 4, 5])
    y2 = np.array([2, 2, 3, 4, 4])
    # cohen_kappa_score with quadratic weights is symmetric
    assert (quadratic_weighted_kappa(y1, y2)
            == pytest.approx(quadratic_weighted_kappa(y2, y1)))


def test_qwk_off_by_one_beats_off_by_two():
    """Quadratic weights penalize distant predictions more."""
    y_true = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    y_close = np.array([1, 3, 3, 3, 5, 1, 3, 3, 3, 5])  # off-by-one on 4 rows
    y_far = np.array([1, 4, 3, 2, 5, 1, 4, 3, 2, 5])    # off-by-two on 4 rows
    assert quadratic_weighted_kappa(y_true, y_close) > \
           quadratic_weighted_kappa(y_true, y_far)


# ---------- _apply_thresholds ----------

def test_apply_thresholds_value_center():
    """Probability concentrated on class 3, expected=3.0. With default
    thresholds [1.5, 2.5, 3.5, 4.5], expected=3.0 → class 3."""
    proba = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])  # 100% on class 3
    thresholds = np.array([1.5, 2.5, 3.5, 4.5])
    preds = _apply_thresholds(proba, thresholds)
    assert preds[0] == 3


def test_apply_thresholds_value_class_1_and_5():
    """Extremes map correctly."""
    proba = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # class 1
        [0.0, 0.0, 0.0, 0.0, 1.0],  # class 5
    ])
    thresholds = np.array([1.5, 2.5, 3.5, 4.5])
    preds = _apply_thresholds(proba, thresholds)
    assert preds[0] == 1
    assert preds[1] == 5


def test_apply_thresholds_boundary_moves_class():
    """Shifting a threshold by 0.5 changes which class gets assigned."""
    # Expected value = 2.4 (mostly class 2, some class 3)
    proba = np.array([[0.0, 0.6, 0.4, 0.0, 0.0]])
    t_low = np.array([1.5, 2.0, 3.5, 4.5])   # boundary at 2.0: 2.4>2.0 → class 3
    t_high = np.array([1.5, 2.5, 3.5, 4.5])  # boundary at 2.5: 2.4<2.5 → class 2
    assert _apply_thresholds(proba, t_low)[0] == 3
    assert _apply_thresholds(proba, t_high)[0] == 2


def test_apply_thresholds_arithmetic_sum_weight():
    """Expected value is literally proba @ [1,2,3,4,5]. If arithmetic is
    mutated (e.g. + → -), the expected value flips sign/magnitude → wrong bin."""
    # proba concentrated between class 4 and 5 → expected=4.5
    # With thresholds [1.5, 2.5, 3.5, 4.5], 4.5 is exactly on boundary
    proba = np.array([[0.0, 0.0, 0.0, 0.5, 0.5]])
    thresholds = np.array([1.5, 2.5, 3.5, 4.5])
    preds = _apply_thresholds(proba, thresholds)
    # expected = 4.5, 4.5 > 4.5 is False, so preds stays at class 4
    assert preds[0] == 4


def test_apply_thresholds_swap_preserves_sort():
    """Thresholds are internally sorted; swapping threshold order shouldn't
    change output."""
    proba = np.array([[0.1, 0.2, 0.3, 0.3, 0.1]])
    t_normal = np.array([1.5, 2.5, 3.5, 4.5])
    t_swapped = np.array([4.5, 3.5, 2.5, 1.5])  # same after sort
    np.testing.assert_array_equal(
        _apply_thresholds(proba, t_normal),
        _apply_thresholds(proba, t_swapped),
    )


# ---------- optimize_thresholds ----------

def test_optimize_thresholds_returns_monotonic():
    """Optimizer output should be sorted ascending."""
    rng = np.random.default_rng(0)
    y = rng.integers(1, 6, size=100)
    proba = np.zeros((100, 5))
    for i, lab in enumerate(y):
        # Put 0.7 mass on true, spread rest
        proba[i, lab - 1] = 0.7
        for j in range(5):
            if j != lab - 1:
                proba[i, j] = 0.3 / 4
    thr, score = optimize_thresholds(proba, y)
    assert np.all(np.diff(thr) > 0)  # monotonically increasing


def test_optimize_thresholds_value_perfect_probas_gives_high_kappa():
    """Perfect probas → QWK ≈ 1.0."""
    y = np.array([1, 2, 3, 4, 5] * 4)  # 20 samples
    proba = np.zeros((20, 5))
    for i, lab in enumerate(y):
        proba[i, lab - 1] = 1.0
    thr, score = optimize_thresholds(proba, y)
    assert score > 0.99


def test_optimize_thresholds_beats_naive_argmax():
    """Threshold optimization should match or beat argmax on QWK."""
    rng = np.random.default_rng(42)
    # Generate noisy probabilities
    y = rng.integers(1, 6, size=200)
    proba = np.zeros((200, 5))
    for i, lab in enumerate(y):
        proba[i] = rng.dirichlet(np.ones(5) * 0.3)
        proba[i, lab - 1] += 0.5
        proba[i] /= proba[i].sum()
    thr, score = optimize_thresholds(proba, y)
    argmax_pred = proba.argmax(axis=1) + 1
    argmax_qwk = quadratic_weighted_kappa(y, argmax_pred)
    assert score >= argmax_qwk - 0.001  # tuned is never worse


# ---------- predict_with_thresholds ----------

def test_predict_with_thresholds_matches_apply():
    """Thin wrapper over _apply_thresholds."""
    proba = np.array([[0.1, 0.2, 0.4, 0.2, 0.1]])
    thr = np.array([1.5, 2.5, 3.5, 4.5])
    np.testing.assert_array_equal(
        predict_with_thresholds(proba, thr),
        _apply_thresholds(proba, thr),
    )
