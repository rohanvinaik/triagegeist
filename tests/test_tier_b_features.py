"""Tests for src.tier_b_features — kill-targeted per LintGate prescriptions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.tier_b_features import (
    TierBArtifacts, _top_bank_confidence, build_tier_b_features,
    fit_tier_b_artifacts,
)


# ---------- _top_bank_confidence ----------

class _FakeSignal:
    def __init__(self, confidence: float):
        self.confidence = confidence


class _FakeDecomp:
    def __init__(self, confs: list[float]):
        self.signals = {f"bank_{i}": _FakeSignal(c)
                        for i, c in enumerate(confs)}


def test_top_bank_confidence_value_max_picked():
    """Function takes max across all signals per decomposition."""
    decomps = [
        _FakeDecomp([0.1, 0.5, 0.9, 0.3]),
        _FakeDecomp([0.2, 0.2, 0.2]),
        _FakeDecomp([0.7]),
    ]
    out = _top_bank_confidence(decomps)
    np.testing.assert_allclose(out, [0.9, 0.2, 0.7])


def test_top_bank_confidence_value_empty_returns_zero():
    """Decomp with no signals → 0."""
    decomps = [_FakeDecomp([])]
    out = _top_bank_confidence(decomps)
    assert out[0] == 0.0


def test_top_bank_confidence_value_order_preserved():
    """Output array order matches input list order."""
    decomps = [_FakeDecomp([0.3]), _FakeDecomp([0.8]), _FakeDecomp([0.1])]
    out = _top_bank_confidence(decomps)
    np.testing.assert_allclose(out, [0.3, 0.8, 0.1])


# ---------- fit_tier_b_artifacts / build_tier_b_features integration ----------
#
# These touch many moving parts (banks, complaints_batch, style, calibrator).
# We exercise the end-to-end path with a tiny realistic dataset.

@pytest.fixture
def tier_b_fixture():
    rng = np.random.default_rng(0)
    n = 50
    train = pd.DataFrame({
        "patient_id": [f"P{i:03d}" for i in range(n)],
        "triage_nurse_id": (["N1"] * 25 + ["N2"] * 25),
        "site_id": ["S1"] * n,
        "triage_acuity": rng.integers(2, 5, size=n),
        "disposition": rng.choice(["admitted", "discharged"], size=n),
        "ed_los_hours": rng.uniform(1.0, 10.0, size=n),
        "age": rng.integers(20, 80, size=n),
        "age_group": ["adult"] * n,
        "sex": ["M"] * n,
        "arrival_mode": ["walk-in"] * n,
        "arrival_hour": [12] * n,
        "arrival_day": ["Mon"] * n,
        "arrival_month": [6] * n,
        "arrival_season": ["summer"] * n,
        "shift": ["day"] * n,
        "mental_status_triage": ["alert"] * n,
        "pain_location": ["none"] * n,
        "language": ["en"] * n,
        "insurance_type": ["private"] * n,
        "transport_origin": ["self"] * n,
        "chief_complaint_system": ["general"] * n,
        "systolic_bp": rng.integers(100, 140, size=n),
        "diastolic_bp": rng.integers(60, 90, size=n),
        "mean_arterial_pressure": rng.integers(70, 110, size=n),
        "pulse_pressure": rng.integers(30, 60, size=n),
        "heart_rate": rng.integers(60, 100, size=n),
        "respiratory_rate": rng.integers(12, 22, size=n),
        "temperature_c": rng.uniform(36.5, 37.5, size=n),
        "spo2": rng.integers(95, 100, size=n),
        "gcs_total": [15] * n,
        "pain_score": rng.integers(0, 8, size=n),
        "weight_kg": [70.0] * n,
        "height_cm": [170.0] * n,
        "bmi": [24.0] * n,
        "shock_index": rng.uniform(0.5, 0.9, size=n),
        "news2_score": rng.uniform(0, 4, size=n),
        "num_prior_ed_visits_12m": [0] * n,
        "num_prior_admissions_12m": [0] * n,
        "num_active_medications": rng.integers(0, 5, size=n),
        "num_comorbidities": rng.integers(0, 3, size=n),
    })
    complaints = pd.DataFrame({
        "patient_id": train.patient_id.values,
        "chief_complaint_raw": ["chest pain"] * n,
        "chief_complaint_system": ["cardiovascular"] * n,
    })
    history = pd.DataFrame({
        "patient_id": train.patient_id.values,
        **{f"hx_{c}": [0] * n for c in [
            "hypertension", "diabetes_type2", "diabetes_type1", "asthma",
            "copd", "heart_failure", "atrial_fibrillation", "ckd",
            "liver_disease", "malignancy", "obesity", "depression",
            "anxiety", "dementia", "epilepsy", "hypothyroidism",
            "hyperthyroidism", "hiv", "coagulopathy", "immunosuppressed",
            "pregnant", "substance_use_disorder", "coronary_artery_disease",
            "stroke_prior", "peripheral_vascular_disease",
        ]},
    })
    return train, complaints, history


def test_fit_tier_b_artifacts_value_returns_frozen_dataclass(tier_b_fixture):
    train, complaints, history = tier_b_fixture
    artifacts = fit_tier_b_artifacts(train, complaints, history,
                                      smoothing=10.0)
    assert isinstance(artifacts, TierBArtifacts)
    # nurse_banks must have both raters
    assert "N1" in artifacts.nurse_banks
    assert "N2" in artifacts.nurse_banks
    assert "S1" in artifacts.site_banks
    assert artifacts.pop_dist.shape == (5,)


def test_fit_tier_b_artifacts_swap_smoothing_changes_banks(tier_b_fixture):
    train, complaints, history = tier_b_fixture
    low_smooth = fit_tier_b_artifacts(train, complaints, history,
                                       smoothing=0.1)
    high_smooth = fit_tier_b_artifacts(train, complaints, history,
                                        smoothing=1000.0)
    # Heavy smoothing → distribution closer to pop, lower L1 deviation
    assert low_smooth.nurse_banks["N1"].l1_dev_from_pop >= \
           high_smooth.nurse_banks["N1"].l1_dev_from_pop


def test_fit_tier_b_artifacts_logical_has_calibrator(tier_b_fixture):
    """Calibrator bin_edges must be monotone-ish."""
    train, complaints, history = tier_b_fixture
    art = fit_tier_b_artifacts(train, complaints, history)
    cal = art.confidence_calibrator
    # Severe rate after isotonic enforcement is non-decreasing
    assert np.all(np.diff(cal.severe_rate) >= -1e-9)
    assert cal.n_per_bin.sum() == len(train)


def test_build_tier_b_features_value_all_columns(tier_b_fixture):
    train, complaints, history = tier_b_fixture
    artifacts = fit_tier_b_artifacts(train, complaints, history)
    feats = build_tier_b_features(train, complaints, history, artifacts)
    expected = {
        "temporal_trajectory_code", "temporal_has_marker",
        "temporal_chronic", "temporal_news2_deviation",
        "temporal_paradox_flag",
        "style_nurse_l1_dev", "style_nurse_over_bias",
        "style_nurse_under_bias", "style_nurse_entropy",
        "style_nurse_expected_esi",
        "style_site_l1_dev", "style_site_over_bias",
        "style_site_under_bias", "style_combined_over_bias",
        "tier_b_calibrated_severe_prob",
    }
    assert expected.issubset(feats.columns)
    assert len(feats) == len(train)


def test_build_tier_b_features_logical_severe_prob_in_bounds(tier_b_fixture):
    """Calibrated severe probability is a probability in [0, 1]."""
    train, complaints, history = tier_b_fixture
    art = fit_tier_b_artifacts(train, complaints, history)
    feats = build_tier_b_features(train, complaints, history, art)
    p = feats["tier_b_calibrated_severe_prob"].values
    assert p.min() >= 0.0
    assert p.max() <= 1.0


def test_build_tier_b_features_swap_apply_df_affects_output(tier_b_fixture):
    """Features on different patient subsets differ."""
    train, complaints, history = tier_b_fixture
    art = fit_tier_b_artifacts(train, complaints, history)
    first_half = train.iloc[:25]
    second_half = train.iloc[25:]
    feats1 = build_tier_b_features(first_half, complaints, history, art)
    feats2 = build_tier_b_features(second_half, complaints, history, art)
    # Nurse bias differs between halves (N1 in first, N2 in second)
    assert not np.allclose(feats1["style_nurse_over_bias"].values,
                            feats2["style_nurse_over_bias"].values)
