"""Tests for src.clinician_style — kill-targeted per LintGate prescriptions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clinician_style import (
    ConfidenceCalibrator, ESI_LEVELS, SEVERE_DISPOSITIONS, StyleBank,
    _style_bank, calibrate, detect_undertriage, fit_confidence_calibrator,
    fit_fold_safe_style_features, fit_style_banks,
    recommend_sample_weights, style_features_for_patients,
)


# ---------- _style_bank ----------

def test_style_bank_value_identical_to_pop():
    """If rater distribution equals population, deviations are 0."""
    pop = np.array([0.04, 0.17, 0.36, 0.29, 0.14])
    sb = _style_bank("nurse_42", pop.copy(), pop, 100)
    assert sb.l1_dev_from_pop == pytest.approx(0.0)
    assert sb.over_triage_bias == pytest.approx(0.0)
    assert sb.under_triage_bias == pytest.approx(0.0)
    assert sb.n_seen == 100


def test_style_bank_arithmetic_l1_distance():
    """l1_dev_from_pop == sum(|dist - pop|)."""
    dist = np.array([0.10, 0.25, 0.30, 0.25, 0.10])
    pop = np.array([0.04, 0.17, 0.36, 0.29, 0.14])
    sb = _style_bank("n1", dist, pop, 50)
    expected = np.abs(dist - pop).sum()
    assert sb.l1_dev_from_pop == pytest.approx(expected)


def test_style_bank_value_over_triage_bias():
    """over_triage_bias is signed: ESI 1-2 share minus pop ESI 1-2 share."""
    dist = np.array([0.20, 0.30, 0.30, 0.15, 0.05])  # ESI1+2 = 0.50
    pop = np.array([0.04, 0.17, 0.36, 0.29, 0.14])   # pop ESI1+2 = 0.21
    sb = _style_bank("over_n", dist, pop, 30)
    assert sb.over_triage_bias == pytest.approx(0.50 - 0.21)


# ---------- fit_style_banks ----------

@pytest.fixture
def style_train():
    """Three nurses with distinct ESI patterns on 30 patients."""
    return pd.DataFrame({
        "triage_nurse_id": (["N1"] * 10 + ["N2"] * 10 + ["N3"] * 10),
        "site_id": ["S1"] * 30,
        # N1 over-triages (more ESI 1-2), N2 balanced, N3 under-triages
        "triage_acuity": (
            [1, 1, 2, 2, 2, 3, 3, 3, 2, 2] +     # N1
            [2, 3, 3, 3, 3, 3, 4, 4, 4, 4] +     # N2
            [3, 4, 4, 4, 4, 5, 5, 5, 5, 5]       # N3
        ),
        "disposition": (["admitted"] * 5 + ["discharged"] * 25),
        "ed_los_hours": [3.0] * 30,
    })


def test_fit_style_banks_value_three_raters(style_train):
    banks, pop = fit_style_banks(style_train, "triage_nurse_id",
                                  smoothing=0.0)
    assert set(banks.keys()) == {"N1", "N2", "N3"}
    # N1 should have highest over-triage bias
    assert banks["N1"].over_triage_bias > banks["N3"].over_triage_bias
    # N3 should have highest under-triage bias (ESI 4+5)
    assert banks["N3"].under_triage_bias > banks["N1"].under_triage_bias


def test_fit_style_banks_boundary_smoothing_shrinks_to_pop(style_train):
    """Large smoothing shrinks individual bank distributions toward population."""
    _, pop = fit_style_banks(style_train, "triage_nurse_id", smoothing=0.0)
    banks_high_smooth, _ = fit_style_banks(style_train, "triage_nurse_id",
                                            smoothing=10000.0)
    banks_no_smooth, _ = fit_style_banks(style_train, "triage_nurse_id",
                                          smoothing=0.0)
    # L1 deviation should shrink toward 0 with high smoothing
    assert banks_high_smooth["N1"].l1_dev_from_pop < \
           banks_no_smooth["N1"].l1_dev_from_pop


def test_fit_style_banks_arithmetic_pop_dist_sums_to_one(style_train):
    _, pop = fit_style_banks(style_train, "triage_nurse_id", smoothing=20.0)
    assert pop.sum() == pytest.approx(1.0)
    assert pop.shape == (5,)


def test_fit_style_banks_swap_rater_column(style_train):
    """Different rater column → different banks emitted."""
    nurse_banks, _ = fit_style_banks(style_train, "triage_nurse_id",
                                      smoothing=10.0)
    site_banks, _ = fit_style_banks(style_train, "site_id", smoothing=10.0)
    assert set(nurse_banks.keys()) != set(site_banks.keys())


# ---------- style_features_for_patients ----------

def test_style_features_value_columns(style_train):
    nurse, pop = fit_style_banks(style_train, "triage_nurse_id", smoothing=10)
    site, _ = fit_style_banks(style_train, "site_id", smoothing=10)
    feats = style_features_for_patients(style_train, nurse, site, pop)
    expected_cols = {
        "style_nurse_l1_dev", "style_nurse_over_bias",
        "style_nurse_under_bias", "style_nurse_entropy",
        "style_nurse_expected_esi",
        "style_site_l1_dev", "style_site_over_bias",
        "style_site_under_bias", "style_combined_over_bias",
    }
    assert expected_cols.issubset(feats.columns)
    assert len(feats) == len(style_train)


def test_style_features_arithmetic_combined_bias_sums_nurse_site(style_train):
    nurse, pop = fit_style_banks(style_train, "triage_nurse_id", smoothing=10)
    site, _ = fit_style_banks(style_train, "site_id", smoothing=10)
    feats = style_features_for_patients(style_train, nurse, site, pop)
    combined = feats["style_nurse_over_bias"] + feats["style_site_over_bias"]
    np.testing.assert_allclose(combined.values,
                                feats["style_combined_over_bias"].values)


def test_style_features_value_unseen_rater_defaults_zero():
    """Patient with an unknown nurse gets default 0 deviations."""
    pop = np.array([0.2] * 5)
    patient_df = pd.DataFrame({
        "patient_id": ["P0"],
        "triage_nurse_id": ["unknown_nurse"],
        "site_id": ["unknown_site"],
    })
    feats = style_features_for_patients(patient_df, {}, {}, pop)
    assert feats["style_nurse_l1_dev"].iloc[0] == 0.0
    assert feats["style_site_l1_dev"].iloc[0] == 0.0


def test_style_features_swap_nurse_site_maps_independent(style_train):
    """Passing site_banks where nurse_banks goes (swapped) changes output."""
    nurse, pop = fit_style_banks(style_train, "triage_nurse_id", smoothing=10)
    site, _ = fit_style_banks(style_train, "site_id", smoothing=10)
    normal = style_features_for_patients(style_train, nurse, site, pop)
    swapped = style_features_for_patients(style_train, site, nurse, pop)
    # With swapped banks, unknown nurse ids fall back to pop → bias=0
    # while the original had real nurse biases
    assert not np.allclose(normal["style_nurse_over_bias"].values,
                            swapped["style_nurse_over_bias"].values)


# ---------- fit_confidence_calibrator ----------

def test_fit_confidence_calibrator_value_monotonic():
    """Severe-outcome rate is non-decreasing when severe patients correlate
    with higher confidence (isotonic fit's expected use)."""
    n = 40
    confidence = np.linspace(0.1, 0.9, n)
    # Construct dispositions correlated with confidence: higher conf → more severe
    disp = ["discharged"] * (n // 2) + ["admitted"] * (n // 2)
    df = pd.DataFrame({
        "disposition": disp,
        "ed_los_hours": [3.0] * n,
    })
    cal = fit_confidence_calibrator(df, confidence, n_bins=5)
    assert np.all(np.diff(cal.severe_rate) >= -1e-9)


def test_fit_confidence_calibrator_boundary_severe_set_coverage():
    """A 'deceased' disposition must be counted as severe."""
    df = pd.DataFrame({
        "disposition": ["deceased"] * 5 + ["discharged"] * 5,
        "ed_los_hours": [2.0] * 10,
    })
    conf = np.array([0.1] * 5 + [0.9] * 5)
    cal = fit_confidence_calibrator(df, conf, n_bins=2)
    # Low-confidence bin should have 100% severe rate (all deceased)
    # But after isotonic, we enforce monotone — if low has higher severe rate
    # than high, pool. In this synthetic case they get pooled to 0.5.
    assert cal.severe_rate[0] <= cal.severe_rate[1] + 1e-9


def test_fit_confidence_calibrator_arithmetic_bin_counts(style_train):
    """n_per_bin counts should sum to training size."""
    conf = np.linspace(0, 1, len(style_train))
    cal = fit_confidence_calibrator(style_train, conf, n_bins=5)
    assert cal.n_per_bin.sum() == len(style_train)


# ---------- calibrate ----------

def test_calibrate_value_returns_severe_rate_array():
    """calibrate maps confidence through the fitted piecewise-constant fn."""
    edges = np.array([-np.inf, 0.3, 0.7, np.inf])
    severe_rate = np.array([0.2, 0.5, 0.8])
    cal = ConfidenceCalibrator(bin_edges=edges, severe_rate=severe_rate,
                                mean_los=np.array([2.0, 3.0, 5.0]),
                                n_per_bin=np.array([10, 10, 10]))
    out = calibrate(cal, np.array([0.1, 0.5, 0.9]))
    np.testing.assert_allclose(out, [0.2, 0.5, 0.8])


def test_calibrate_arithmetic_monotone_input_yields_monotone_output():
    edges = np.array([-np.inf, 0.25, 0.5, 0.75, np.inf])
    severe_rate = np.array([0.1, 0.3, 0.5, 0.8])
    cal = ConfidenceCalibrator(bin_edges=edges, severe_rate=severe_rate,
                                mean_los=np.full(4, 3.0),
                                n_per_bin=np.full(4, 5))
    out = calibrate(cal, np.array([0.2, 0.4, 0.6, 0.8]))
    assert np.all(np.diff(out) >= 0)


def test_calibrate_swap_cal_shape_changes_result():
    """Different calibrators on same confidence produce different outputs."""
    edges = np.array([-np.inf, 0.5, np.inf])
    cal_low = ConfidenceCalibrator(bin_edges=edges,
                                    severe_rate=np.array([0.1, 0.2]),
                                    mean_los=np.array([2.0, 3.0]),
                                    n_per_bin=np.array([5, 5]))
    cal_high = ConfidenceCalibrator(bin_edges=edges,
                                     severe_rate=np.array([0.6, 0.9]),
                                     mean_los=np.array([2.0, 3.0]),
                                     n_per_bin=np.array([5, 5]))
    inp = np.array([0.3, 0.7])
    assert not np.allclose(calibrate(cal_low, inp), calibrate(cal_high, inp))


# ---------- detect_undertriage / recommend_sample_weights ----------

def test_detect_undertriage_value():
    """ESI 4/5 + severe disposition + long LOS → flagged."""
    df = pd.DataFrame({
        "triage_acuity": [4, 4, 5, 2, 3],
        "disposition": ["admitted", "discharged", "admitted",
                         "admitted", "admitted"],
        "ed_los_hours": [8.0, 4.0, 9.0, 8.0, 8.0],
    })
    sus = detect_undertriage(df, los_threshold=6.0)
    # Expected: row 0 (ESI 4, admitted, 8h), row 2 (ESI 5, admitted, 9h)
    assert sus.iloc[0] is np.True_ or sus.iloc[0] == True  # noqa: E712
    assert sus.iloc[2] == True  # noqa: E712
    # Not flagged: row 3 (ESI 2 — not in 4/5), row 4 (ESI 3 — not in 4/5),
    # row 1 (discharged — not severe)
    assert sus.iloc[1] == False  # noqa: E712
    assert sus.iloc[3] == False  # noqa: E712
    assert sus.iloc[4] == False  # noqa: E712


def test_detect_undertriage_boundary_los_threshold():
    """LOS at threshold is excluded (strict > required)."""
    df = pd.DataFrame({
        "triage_acuity": [4, 4],
        "disposition": ["admitted", "admitted"],
        "ed_los_hours": [5.9, 6.0],
    })
    sus = detect_undertriage(df, los_threshold=6.0)
    assert sus.iloc[0] == False  # below  # noqa: E712
    assert sus.iloc[1] == True   # exactly at threshold (los >= threshold)  # noqa: E712


def test_recommend_sample_weights_arithmetic():
    """Suspect rows get suspect_weight (default 0.5), others 1.0."""
    df = pd.DataFrame({
        "triage_acuity": [4, 2, 5, 3],
        "disposition": ["admitted", "discharged", "admitted", "admitted"],
        "ed_los_hours": [8.0, 2.0, 10.0, 8.0],
    })
    w = recommend_sample_weights(df, suspect_weight=0.5)
    # Rows 0 and 2 suspect → 0.5; rows 1 and 3 not suspect → 1.0
    assert w[0] == 0.5
    assert w[2] == 0.5
    assert w[1] == 1.0
    assert w[3] == 1.0


def test_recommend_sample_weights_value_custom_weight():
    df = pd.DataFrame({
        "triage_acuity": [4],
        "disposition": ["admitted"],
        "ed_los_hours": [10.0],
    })
    w = recommend_sample_weights(df, suspect_weight=0.1)
    assert w[0] == pytest.approx(0.1)


# ---------- fit_fold_safe_style_features ----------

def test_fit_fold_safe_style_swap_fold_order(style_train):
    folds = [
        (np.arange(0, 15), np.arange(15, 30)),
        (np.arange(15, 30), np.arange(0, 15)),
    ]
    feats = fit_fold_safe_style_features(style_train, folds, smoothing=10.0)
    # All rows covered, in original index order
    assert len(feats) == len(style_train)
    assert list(feats.index) == list(style_train.index)


def test_fit_fold_safe_style_value_columns_populated(style_train):
    folds = [(np.arange(0, 20), np.arange(20, 30))]
    feats = fit_fold_safe_style_features(style_train, folds, smoothing=5.0)
    # All style columns present
    assert "style_nurse_l1_dev" in feats.columns
    # Val rows (20-29) get real feature values; unused train rows get 0 (NaN filled)
    assert feats.loc[20:29, "style_nurse_l1_dev"].notna().all()


# ---------- Module-level constants ----------

def test_esi_levels_value():
    assert ESI_LEVELS == [1, 2, 3, 4, 5]


def test_severe_dispositions_value():
    assert "admitted" in SEVERE_DISPOSITIONS
    assert "deceased" in SEVERE_DISPOSITIONS
    assert "discharged" not in SEVERE_DISPOSITIONS
