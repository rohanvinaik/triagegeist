"""Tests for src.temporal_bank — kill-targeted per LintGate prescriptions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.temporal_bank import (
    TRAJECTORY_CODES, _extract_base_condition, _extract_trajectory,
    build_temporal_features, fit_cohort_expectations,
    fit_fold_safe_temporal_features,
)


# ---------- _extract_trajectory ----------

@pytest.mark.parametrize("text,expected", [
    ("chest pain, worsening over hours", "rapid_worsening"),
    ("abdominal pain, worsening", "gradual_worsening"),
    ("fever, onset today", "onset_today"),
    ("cough, since yesterday", "subacute"),
    ("joint pain, chronic", "chronic"),
    ("headache, for 3 days", "days_duration"),
    ("nausea, constant", "constant"),
    ("wheeze, intermittent", "intermittent"),
    ("chest pain", "unspecified"),
    ("", "unspecified"),
])
def test_extract_trajectory_value(text, expected):
    assert _extract_trajectory(text) == expected


def test_extract_trajectory_logical_worsening_vs_rapid():
    """Rapid check: 'worsening over hours' must win over generic 'worsening'."""
    assert _extract_trajectory("foo, worsening over hours") == "rapid_worsening"
    assert _extract_trajectory("foo, worsening") == "gradual_worsening"


def test_extract_trajectory_type_non_string_returns_unspecified():
    """Non-string input must not raise; returns default."""
    assert _extract_trajectory(None) == "unspecified"
    assert _extract_trajectory(42) == "unspecified"


# ---------- _extract_base_condition ----------

@pytest.mark.parametrize("text,expected", [
    ("chest pain", "chest pain"),
    ("severe chest pain", "chest pain"),
    ("chest pain, worsening", "chest pain"),
    ("moderate headache, intermittent", "headache"),
    ("acute abdomen", "abdomen"),
    ("mild nausea, onset today", "nausea"),
    ("", "unknown"),
])
def test_extract_base_condition_value(text, expected):
    assert _extract_base_condition(text) == expected


def test_extract_base_condition_logical_modifier_strip():
    """Modifier prefix stripping iterates — nested prefixes both removed."""
    # Loop strips each matching prefix in order: "severe " then "acute "
    assert _extract_base_condition("severe acute chest pain") == "chest pain"


def test_extract_base_condition_type_non_string():
    assert _extract_base_condition(None) == "unknown"
    assert _extract_base_condition(3.14) == "unknown"


# ---------- fit_cohort_expectations ----------

@pytest.fixture
def simple_train():
    return pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(6)],
        "age_group": ["adult"] * 3 + ["elderly"] * 3,
        "news2_score": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "heart_rate": [80, 90, 100, 110, 120, 130],
        "respiratory_rate": [16, 18, 20, 22, 24, 26],
        "spo2": [98, 97, 96, 95, 94, 93],
        "temperature_c": [37.0, 37.2, 37.4, 37.6, 37.8, 38.0],
        "shock_index": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    })


@pytest.fixture
def simple_complaints():
    return pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(6)],
        "chief_complaint_raw": [
            "chest pain", "chest pain", "chest pain",
            "shortness of breath", "shortness of breath", "shortness of breath",
        ],
    })


def test_fit_cohort_expectations_value(simple_train, simple_complaints):
    cohort = fit_cohort_expectations(simple_train, simple_complaints)
    # 2 (base, age_group) pairs: (chest pain, adult) and (sob, elderly)
    assert len(cohort) == 2
    # (chest pain, adult) NEWS2 = (2+3+4)/3 = 3.0
    chest_adult = cohort[(cohort["base"] == "chest pain") &
                          (cohort["age_group"] == "adult")]
    assert chest_adult["news2_score"].iloc[0] == pytest.approx(3.0)
    # (sob, elderly) NEWS2 = (5+6+7)/3 = 6.0
    sob_elderly = cohort[(cohort["base"] == "shortness of breath") &
                          (cohort["age_group"] == "elderly")]
    assert sob_elderly["news2_score"].iloc[0] == pytest.approx(6.0)


# ---------- build_temporal_features ----------

def test_build_temporal_features_value_columns(simple_train, simple_complaints):
    cohort = fit_cohort_expectations(simple_train, simple_complaints)
    feats = build_temporal_features(simple_train, simple_complaints, cohort)
    required = {"temporal_trajectory_code", "temporal_has_marker",
                "temporal_chronic", "temporal_expected_news2",
                "temporal_news2_deviation", "temporal_paradox_flag"}
    assert required.issubset(feats.columns)
    assert len(feats) == len(simple_train)


def test_build_temporal_features_boundary_paradox_flag():
    """Paradox fires when cohort mean ≥ 5.0 AND own NEWS2 ≤ 2.0."""
    train = pd.DataFrame({
        "patient_id": ["P0", "P1"],
        "age_group": ["adult", "adult"],
        "news2_score": [6.0, 1.0],
        "heart_rate": [100, 100],
        "respiratory_rate": [20, 20],
        "spo2": [96, 96],
        "temperature_c": [37.0, 37.0],
        "shock_index": [0.7, 0.7],
    })
    complaints = pd.DataFrame({
        "patient_id": ["P0", "P1"],
        "chief_complaint_raw": ["sepsis severe", "sepsis severe"],
    })
    # Train cohort has P0 at news2=6, P1 at news2=1. Mean = 3.5 — below threshold.
    # Bump to fire paradox: set both to high NEWS2, use a separate apply frame
    train_high = train.copy()
    train_high["news2_score"] = [6.0, 7.0]
    cohort = fit_cohort_expectations(train_high, complaints)
    # cohort mean for (sepsis, adult) = 6.5 — above paradox_cohort_threshold=5
    # Now build features with a low-NEWS2 patient from the same cohort
    apply_df = pd.DataFrame({
        "patient_id": ["Q0", "Q1"],
        "age_group": ["adult", "adult"],
        "news2_score": [1.0, 8.0],   # Q0 is paradox, Q1 is not
    })
    apply_complaints = pd.DataFrame({
        "patient_id": ["Q0", "Q1"],
        "chief_complaint_raw": ["sepsis severe", "sepsis severe"],
    })
    feats = build_temporal_features(apply_df, apply_complaints, cohort)
    assert feats["temporal_paradox_flag"].iloc[0] == 1  # low NEWS2 in severe cohort
    assert feats["temporal_paradox_flag"].iloc[1] == 0


def test_build_temporal_features_arithmetic_deviation_sign():
    """deviation = own - expected. Own above cohort → positive; below → negative."""
    train = pd.DataFrame({
        "patient_id": ["P0", "P1"],
        "age_group": ["adult", "adult"],
        "news2_score": [5.0, 5.0],  # cohort mean = 5
        "heart_rate": [100, 100],
        "respiratory_rate": [20, 20],
        "spo2": [96, 96],
        "temperature_c": [37.0, 37.0],
        "shock_index": [0.7, 0.7],
    })
    complaints = pd.DataFrame({
        "patient_id": ["P0", "P1"],
        "chief_complaint_raw": ["chest pain", "chest pain"],
    })
    cohort = fit_cohort_expectations(train, complaints)

    apply_df = pd.DataFrame({
        "patient_id": ["Q0", "Q1"],
        "age_group": ["adult", "adult"],
        "news2_score": [7.0, 2.0],  # above and below cohort mean
    })
    apply_c = pd.DataFrame({
        "patient_id": ["Q0", "Q1"],
        "chief_complaint_raw": ["chest pain", "chest pain"],
    })
    feats = build_temporal_features(apply_df, apply_c, cohort)
    assert feats["temporal_news2_deviation"].iloc[0] == pytest.approx(2.0)
    assert feats["temporal_news2_deviation"].iloc[1] == pytest.approx(-3.0)


def test_build_temporal_features_swap_trajectory_code_matches_map():
    """Trajectory text → TRAJECTORY_CODES mapping must be stable."""
    train = pd.DataFrame({
        "patient_id": ["P0"],
        "age_group": ["adult"],
        "news2_score": [3.0],
        "heart_rate": [80],
        "respiratory_rate": [16],
        "spo2": [98],
        "temperature_c": [37.0],
        "shock_index": [0.5],
    })
    complaints = pd.DataFrame({
        "patient_id": ["P0"],
        "chief_complaint_raw": ["chest pain, chronic"],
    })
    cohort = fit_cohort_expectations(train, complaints)
    feats = build_temporal_features(train, complaints, cohort)
    assert feats["temporal_trajectory_code"].iloc[0] == TRAJECTORY_CODES["chronic"]
    assert feats["temporal_chronic"].iloc[0] == 1
    assert feats["temporal_has_marker"].iloc[0] == 1


# ---------- fit_fold_safe_temporal_features ----------

def test_fit_fold_safe_swap_fold_order_preserves_rows(simple_train,
                                                      simple_complaints):
    """OOF features must be indexed back to original order regardless of
    fold ordering."""
    fold_indices = [
        (np.array([0, 1, 2]), np.array([3, 4, 5])),
        (np.array([3, 4, 5]), np.array([0, 1, 2])),
    ]
    feats = fit_fold_safe_temporal_features(simple_train, simple_complaints,
                                             fold_indices)
    # All 6 rows covered, in original sort order
    assert len(feats) == 6
    assert list(feats.index) == sorted(feats.index)
