"""Tests for src.feature_engine — kill-targeted per LintGate prescriptions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.banks import (
    Bank, BankDecomposition, BankSignal, decompose_dataframe,
)
from src.complaint_lexicon import classify_complaints_batch
from src.feature_engine import (
    LEAKAGE_COLS, _build_bank_features, _build_clinical_features,
    _extract_text_features, build_features,
)


# ---------- _extract_text_features ----------

def test_extract_text_features_value_keyword_groups():
    """Keyword groups fire when matching substrings appear."""
    texts = pd.Series([
        "severe chest pain",       # pain, cardiac, severity_high
        "mild cough",              # respiratory, severity_low
        "fever and nausea",        # infection, gi
    ])
    feats = _extract_text_features(texts)
    assert feats.loc[0, "cc_pain"] == 1
    assert feats.loc[0, "cc_cardiac"] == 1
    assert feats.loc[0, "cc_severity_high"] == 1
    assert feats.loc[1, "cc_respiratory"] == 1
    assert feats.loc[1, "cc_severity_low"] == 1
    assert feats.loc[2, "cc_infection"] == 1
    assert feats.loc[2, "cc_gi"] == 1


def test_extract_text_features_value_word_count_len():
    texts = pd.Series(["chest pain"])
    feats = _extract_text_features(texts)
    assert feats.loc[0, "cc_len"] == 10  # "chest pain" = 10 chars
    assert feats.loc[0, "cc_word_count"] == 2


def test_extract_text_features_value_no_cc_condition_prior():
    """Declined shortcut: cc_base_condition must NOT be emitted anymore."""
    feats = _extract_text_features(pd.Series(["chest pain, worsening"]))
    assert "cc_base_condition" not in feats.columns
    assert "cc_condition_prior" not in feats.columns


def test_extract_text_features_logical_missing_text_returns_zeros():
    """NaN/empty strings give 0 keyword counts."""
    feats = _extract_text_features(pd.Series([None, "", float("nan")]))
    assert feats.loc[0, "cc_pain"] == 0
    assert feats.loc[1, "cc_word_count"] == 0


# ---------- _build_clinical_features ----------

def _make_clinical_df(**overrides):
    defaults = {
        "heart_rate": [80, 120, 60, 100, 85],
        "respiratory_rate": [16, 24, 10, 18, 14],
        "spo2": [98, 92, 97, 95, 99],
        "systolic_bp": [120, 85, 180, 110, 130],
        "temperature_c": [37.0, 39.0, 35.0, 37.5, 38.6],
        "gcs_total": [15, 12, 15, 15, 15],
        "news2_score": [0.0, 7.0, 3.0, 2.0, 1.0],
        "age": [30, 70, 45, 5, 90],
        "age_group": ["adult", "elderly", "adult", "peds", "elderly"],
        "pain_score": [3, 8, 0, -1, 5],
        "num_active_medications": [2, 8, 1, 0, 5],
        "num_prior_ed_visits_12m": [0, 3, 1, 0, 2],
        "hx_heart_failure": [0, 1, 0, 0, 1],
        "hx_immunosuppressed": [0, 0, 1, 0, 0],
        "hx_coagulopathy": [0, 0, 0, 0, 1],
        "mental_status_triage": ["alert", "confused", "alert",
                                  "agitated", "drowsy"],
        "diastolic_bp": [80, 50, 110, 70, 85],
        "shock_index": [0.67, 1.41, 0.33, 0.91, 0.65],
    }
    defaults.update(overrides)
    return pd.DataFrame(defaults)


def test_build_clinical_features_arithmetic_hr_rr_product():
    df = _make_clinical_df(heart_rate=[80, 120], respiratory_rate=[16, 24],
                            spo2=[98, 92], systolic_bp=[120, 85],
                            temperature_c=[37.0, 39.0], gcs_total=[15, 12],
                            news2_score=[0.0, 7.0], age=[30, 70],
                            age_group=["adult", "elderly"],
                            pain_score=[3, 8],
                            num_active_medications=[2, 8],
                            num_prior_ed_visits_12m=[0, 3],
                            hx_heart_failure=[0, 1],
                            hx_immunosuppressed=[0, 0],
                            hx_coagulopathy=[0, 0],
                            mental_status_triage=["alert", "confused"],
                            diastolic_bp=[80, 50],
                            shock_index=[0.67, 1.41])
    feats = _build_clinical_features(df)
    assert feats.loc[0, "hr_rr_product"] == pytest.approx(80 * 16)
    assert feats.loc[1, "hr_rr_product"] == pytest.approx(120 * 24)


def test_build_clinical_features_arithmetic_gcs_deficit():
    df = _make_clinical_df()
    feats = _build_clinical_features(df)
    # GCS 15 → deficit 0; GCS 12 → deficit 3
    assert feats.loc[0, "gcs_deficit"] == 0
    assert feats.loc[1, "gcs_deficit"] == 3


def test_build_clinical_features_boundary_vital_instability_count():
    """vital_instability counts deranged vitals."""
    # Row 1 in _make_clinical_df is designed as highly deranged:
    # HR=120 (>100 +1), RR=24 (>22 +1), SpO2=92 (<94 +1),
    # SBP=85 (<90 +1), Temp=39.0 (>38.5 +1), GCS=12 (<15 +1) → 6
    df = _make_clinical_df()
    feats = _build_clinical_features(df)
    assert feats.loc[1, "vital_instability"] == 6


def test_build_clinical_features_logical_immuno_fever():
    """immuno_x_fever fires only when both conditions hold."""
    df = _make_clinical_df()
    feats = _build_clinical_features(df)
    # Row 0: immuno=0, temp=37.0 → 0
    # Row 2: immuno=1, temp=35.0 → 0 (no fever)
    # Row 4: immuno=0, temp=38.6 → 0 (no immuno)
    assert feats.loc[0, "immuno_x_fever"] == 0
    assert feats.loc[2, "immuno_x_fever"] == 0
    assert feats.loc[4, "immuno_x_fever"] == 0


def test_build_clinical_features_logical_age_bin_fine():
    """Age bins are ordinal and increasing with age."""
    df = _make_clinical_df()
    feats = _build_clinical_features(df)
    # age [30, 70, 45, 5, 90] → bins should be increasing with age
    # age 5 → lowest bin, age 90 → highest
    assert feats.loc[3, "age_bin_fine"] < feats.loc[0, "age_bin_fine"]
    assert feats.loc[4, "age_bin_fine"] > feats.loc[0, "age_bin_fine"]


def test_build_clinical_features_value_mental_severity_ordinal():
    """mental_status: alert=0 < confused=3."""
    df = _make_clinical_df()
    feats = _build_clinical_features(df)
    assert feats.loc[0, "mental_severity"] == 0   # alert
    assert feats.loc[1, "mental_severity"] == 3   # confused
    assert feats.loc[3, "mental_severity"] == 1   # agitated
    assert feats.loc[4, "mental_severity"] == 2   # drowsy


# ---------- _build_bank_features ----------

def _make_decomp(patient_id="P1"):
    """Build a BankDecomposition with 3 signals."""
    d = BankDecomposition(patient_id=patient_id)
    d.add(BankSignal(bank=Bank.SEVERITY, esi_estimate=3.0, confidence=0.8,
                     esi_floor=0, esi_ceiling=6, evidence="news2=3"))
    d.add(BankSignal(bank=Bank.CONSCIOUSNESS, esi_estimate=2.0, confidence=0.9,
                     esi_floor=0, esi_ceiling=6, evidence="gcs=14"))
    d.add(BankSignal(bank=Bank.DEMOGRAPHIC, esi_estimate=4.5, confidence=0.3,
                     esi_floor=0, esi_ceiling=6, evidence="adult"))
    return d


def test_build_bank_features_value_esi_and_conf_columns():
    feats = _build_bank_features([_make_decomp()])
    assert feats.loc["P1", "bank_severity_esi"] == 3.0
    assert feats.loc["P1", "bank_severity_conf"] == 0.8
    assert feats.loc["P1", "bank_consciousness_esi"] == 2.0


def test_build_bank_features_arithmetic_kuramoto_r_in_zero_one():
    """Kuramoto order parameter r ∈ [0, 1]."""
    feats = _build_bank_features([_make_decomp()])
    r = feats.loc["P1", "bank_r_total"]
    assert 0.0 <= r <= 1.0


def test_build_bank_features_boundary_phase_dev_in_neg_pi_pi():
    """Signed phase deviation wrapped to [-π, π]."""
    feats = _build_bank_features([_make_decomp()])
    for col in ["bank_severity_dev", "bank_consciousness_dev",
                "bank_demographic_dev"]:
        dev = feats.loc["P1", col]
        assert -np.pi - 1e-9 <= dev <= np.pi + 1e-9


def test_build_bank_features_logical_low_conf_bank_dev_zero():
    """Banks with confidence <= 0.05 get zero phase deviation."""
    d = BankDecomposition(patient_id="P1")
    d.add(BankSignal(bank=Bank.SEVERITY, esi_estimate=3.0, confidence=0.8,
                     esi_floor=0, esi_ceiling=6, evidence="x"))
    d.add(BankSignal(bank=Bank.ARRIVAL, esi_estimate=4.0, confidence=0.0,
                     esi_floor=0, esi_ceiling=6, evidence="noise"))
    feats = _build_bank_features([d])
    assert feats.loc["P1", "bank_arrival_dev"] == 0.0


def test_build_bank_features_value_coherence_spread():
    """coherence_spread = |r_core - r_chronic|."""
    feats = _build_bank_features([_make_decomp()])
    r_core = feats.loc["P1", "bank_r_physiologic_core"]
    r_chronic = feats.loc["P1", "bank_r_chronic_profile"]
    spread = feats.loc["P1", "bank_coherence_spread"]
    assert spread == pytest.approx(abs(r_core - r_chronic))


# ---------- build_features (integration) ----------

@pytest.fixture
def build_fixture():
    rng = np.random.default_rng(7)
    n = 10
    train = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(n)],
        "site_id": ["S1"] * n,
        "triage_nurse_id": ["N1"] * n,
        "arrival_mode": ["walk-in"] * n,
        "arrival_hour": [12] * n,
        "arrival_day": ["Mon"] * n,
        "arrival_month": [6] * n,
        "arrival_season": ["summer"] * n,
        "shift": ["day"] * n,
        "age": rng.integers(20, 80, size=n),
        "age_group": ["adult"] * n,
        "sex": ["M"] * n,
        "language": ["en"] * n,
        "insurance_type": ["private"] * n,
        "transport_origin": ["self"] * n,
        "pain_location": ["none"] * n,
        "mental_status_triage": ["alert"] * n,
        "chief_complaint_system": ["general"] * n,
        "num_prior_ed_visits_12m": [0] * n,
        "num_prior_admissions_12m": [0] * n,
        "num_active_medications": rng.integers(0, 5, size=n),
        "num_comorbidities": rng.integers(0, 3, size=n),
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
        "disposition": ["discharged"] * n,    # leakage col
        "ed_los_hours": [2.0] * n,             # leakage col
        "triage_acuity": rng.integers(2, 5, size=n),
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


def test_build_features_logical_leakage_cols_removed(build_fixture):
    """Leakage columns MUST be absent from output features."""
    train, complaints, history = build_fixture
    cc = classify_complaints_batch(complaints)
    merged = train.merge(history, on="patient_id", how="left",
                          suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    decomps = decompose_dataframe(merged, cc)
    feats = build_features(train, complaints, history, decomps)
    for leak in LEAKAGE_COLS:
        assert leak not in feats.columns, f"{leak} leaked into features"


def test_build_features_value_banks_merge(build_fixture):
    """Bank features get merged via patient_id."""
    train, complaints, history = build_fixture
    cc = classify_complaints_batch(complaints)
    merged = train.merge(history, on="patient_id", how="left",
                          suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    decomps = decompose_dataframe(merged, cc)
    feats = build_features(train, complaints, history, decomps)
    assert "bank_severity_esi" in feats.columns
    assert "bank_r_total" in feats.columns
    assert len(feats) == len(train)


def test_build_features_arithmetic_pain_imputation(build_fixture):
    """pain_score=-1 rows get imputed from same-age-group median."""
    train, complaints, history = build_fixture
    # Force one row to pain_score=-1
    train = train.copy()
    train.loc[0, "pain_score"] = -1
    feats = build_features(train, complaints, history, None)
    # After imputation, row 0's pain_score should be >= 0 (median of others)
    assert feats.loc[0, "pain_score"] >= 0


def test_build_features_value_text_keywords_present(build_fixture):
    train, complaints, history = build_fixture
    feats = build_features(train, complaints, history, None)
    assert "cc_pain" in feats.columns
    assert "cc_cardiac" in feats.columns
    # All rows have "chest pain" → cc_cardiac=1 everywhere
    assert feats["cc_cardiac"].sum() == len(train)
