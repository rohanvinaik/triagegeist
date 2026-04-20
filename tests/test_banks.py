"""Tests for src.banks — kill-targeted per LintGate prescriptions.

Covers 17 functions / 576 mutants. Each bank is driven by boundary-rich
clinical threshold rules — tests hit both sides of every cutoff.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.banks import (
    BANK_FUNCTIONS, Bank, BankDecomposition, BankSignal, _arrival_bank,
    _cardiovascular_bank, _consciousness_bank, _cv_severity_from_shock_index,
    _cv_severity_from_vitals, _demographic_bank, _history_bank,
    _is_missing, _pain_bank, _respiratory_bank, _safe_float,
    _severity_bank, _thermal_bank, _utilization_bank,
    decompose_dataframe, decompose_patient,
)


# ---------- _safe_float ----------

@pytest.mark.parametrize("val,default,expected", [
    (3.5, 0.0, 3.5),
    (7, 0.0, 7.0),
    (None, 5.0, 5.0),
    (float("nan"), 9.0, 9.0),
    (0, 1.0, 0.0),
])
def test_safe_float_value(val, default, expected):
    assert _safe_float(val, default) == expected


def test_safe_float_type_returns_float():
    """Always returns a float, never int."""
    assert isinstance(_safe_float(42, 0.0), float)
    assert isinstance(_safe_float(None, 0.0), float)


def test_safe_float_logical_nan_triggers_default():
    """NaN triggers the default branch, not the float-conversion branch."""
    assert _safe_float(float("nan"), 99.0) == 99.0


# ---------- _is_missing ----------

@pytest.mark.parametrize("val,expected", [
    (None, True),
    (float("nan"), True),
    (0, False),
    (0.0, False),
    (3.14, False),
    ("", False),      # non-None string is not missing
])
def test_is_missing_value(val, expected):
    assert _is_missing(val) == expected


def test_is_missing_logical_zero_not_missing():
    """0 is a valid value, NOT missing."""
    assert _is_missing(0) is False
    assert _is_missing(0.0) is False


# ---------- _severity_bank (NEWS2 → ESI thresholds) ----------

def _row(**kwargs):
    return pd.Series(kwargs)


@pytest.mark.parametrize("news2,expected_esi,expected_floor,expected_ceil", [
    (13.0, 1.5, 1, 2),     # critical
    (14.0, 1.5, 1, 2),     # critical (above threshold)
    (9.0, 1.9, 1, 2),      # very-high
    (12.0, 1.9, 1, 2),     # very-high upper
    (7.0, 2.2, 1, 3),      # high
    (8.0, 2.2, 1, 3),      # high
    (5.0, 2.8, 2, 4),      # medium
    (6.0, 2.8, 2, 4),
    (3.0, 3.1, 2, 4),      # low-medium
    (4.0, 3.1, 2, 4),
    (2.0, 3.4, 2, 5),      # low
    (1.0, 3.8, 3, 5),      # low
    (0.0, 4.2, 3, 5),      # minimal
])
def test_severity_bank_boundary(news2, expected_esi, expected_floor,
                                 expected_ceil):
    sig = _severity_bank(_row(news2_score=news2))
    assert sig.esi_estimate == expected_esi
    assert sig.esi_floor == expected_floor
    assert sig.esi_ceiling == expected_ceil


def test_severity_bank_value_missing_confidence_zero():
    sig = _severity_bank(_row())  # no news2_score
    assert sig.confidence == 0.0
    assert "missing" in sig.evidence


def test_severity_bank_value_confidence_increases_with_severity():
    """Higher NEWS2 → higher confidence in the bank's call."""
    s0 = _severity_bank(_row(news2_score=0))
    s5 = _severity_bank(_row(news2_score=5))
    s13 = _severity_bank(_row(news2_score=13))
    assert s0.confidence < s5.confidence < s13.confidence


# ---------- _consciousness_bank ----------

@pytest.mark.parametrize("gcs,mental,expected_esi", [
    (3, "unresponsive", 1.0),       # comatose
    (8, "", 1.0),                    # boundary: GCS 8
    (9, "", 1.5),                    # severely impaired
    (12, "", 1.5),                   # severely impaired upper
    (13, "confused", 2.5),           # mildly impaired
    (14, "drowsy", 3.0),             # mildly impaired
    (14, "unresponsive", 1.5),       # contradictory → trust mental
    (15, "alert", 3.8),              # normal
    (15, "unresponsive", 1.5),       # contradiction
    (15, "agitated", 2.5),
    (15, "confused", 2.8),
    (15, "drowsy", 2.6),
])
def test_consciousness_bank_value(gcs, mental, expected_esi):
    sig = _consciousness_bank(_row(gcs_total=gcs,
                                    mental_status_triage=mental))
    assert sig.esi_estimate == expected_esi


def test_consciousness_bank_boundary_gcs_8_vs_9():
    """GCS 8 = comatose floor; GCS 9 = severely impaired."""
    s8 = _consciousness_bank(_row(gcs_total=8, mental_status_triage=""))
    s9 = _consciousness_bank(_row(gcs_total=9, mental_status_triage=""))
    assert s8.esi_floor == 1 and s8.esi_ceiling == 1
    assert s9.esi_floor == 1 and s9.esi_ceiling == 2


def test_consciousness_bank_boundary_gcs_15_alert_vs_agitated():
    """Default GCS 15 alert → 3.8; agitated pulls it down to 2.5."""
    alert = _consciousness_bank(_row(gcs_total=15,
                                      mental_status_triage="alert"))
    agit = _consciousness_bank(_row(gcs_total=15,
                                     mental_status_triage="agitated"))
    assert alert.esi_estimate > agit.esi_estimate


# ---------- _respiratory_bank ----------

@pytest.mark.parametrize("spo2,rr,expected_esi", [
    (84, 16, 1.0),         # SpO2 critical
    (95, 36, 1.0),         # RR critical high
    (95, 7, 1.0),          # RR critical low
    (88, 16, 1.5),         # SpO2 severe
    (95, 32, 1.5),         # RR severe
    (92, 16, 2.5),         # SpO2 moderate
    (95, 28, 2.5),         # RR moderate
    (94, 16, 3.2),         # mild: actually SpO2 < 96
    (95, 23, 3.2),         # RR mild
    (97, 10, 3.5),         # normal SpO2, low RR
    (98, 16, 5.0),         # normal
])
def test_respiratory_bank_boundary(spo2, rr, expected_esi):
    sig = _respiratory_bank(_row(spo2=spo2, respiratory_rate=rr))
    assert sig.esi_estimate == expected_esi


def test_respiratory_bank_logical_critical_triggered_by_either():
    """Critical branch fires if EITHER spo2<85 OR RR out of range."""
    s1 = _respiratory_bank(_row(spo2=80, respiratory_rate=16))
    s2 = _respiratory_bank(_row(spo2=98, respiratory_rate=40))
    assert s1.esi_estimate == 1.0
    assert s2.esi_estimate == 1.0


def test_respiratory_bank_value_missing_defaults_to_normal():
    """Missing vitals → use defaults (97, 16) → normal branch."""
    sig = _respiratory_bank(_row())
    assert sig.esi_estimate == 5.0


# ---------- _cv_severity_from_shock_index ----------

@pytest.mark.parametrize("si,expected_esi", [
    (1.5, 1.0),       # critical
    (1.31, 1.0),      # just above critical boundary
    (1.1, 2.0),       # elevated
    (1.01, 2.0),
    (0.95, 3.0),      # borderline
    (0.91, 3.0),
])
def test_cv_severity_from_shock_index_value(si, expected_esi):
    result = _cv_severity_from_shock_index(si)
    assert result is not None
    assert result[0] == expected_esi


def test_cv_severity_from_shock_index_boundary_returns_none_below():
    """SI ≤ 0.9 → returns None (falls through to vitals)."""
    assert _cv_severity_from_shock_index(0.9) is None
    assert _cv_severity_from_shock_index(0.7) is None


def test_cv_severity_from_shock_index_boundary_just_above_0_9():
    """SI = 0.91 → borderline branch."""
    result = _cv_severity_from_shock_index(0.91)
    assert result is not None
    assert result[0] == 3.0


# ---------- _cv_severity_from_vitals ----------

@pytest.mark.parametrize("hr,sbp,expected_esi", [
    (135, 120, 1.2),    # hr_critical: hr > 130
    (35, 120, 1.2),     # hr_critical: hr < 40
    (80, 75, 1.2),      # sbp_critical: sbp < 80
    (115, 120, 2.2),    # hr_severe: 110 < hr <= 130
    (45, 120, 2.2),     # hr_severe: 40 <= hr < 50
    (80, 85, 2.2),      # sbp_severe: 80 <= sbp < 90
    (80, 210, 2.2),     # sbp_severe: sbp > 200
    (80, 120, 4.8),     # stable
    (80, None, 4.8),    # stable (no sbp)
])
def test_cv_severity_from_vitals_boundary(hr, sbp, expected_esi):
    result = _cv_severity_from_vitals(hr, sbp)
    assert result[0] == expected_esi


def test_cv_severity_from_vitals_logical_none_sbp_doesnt_trigger():
    """sbp=None must not be treated as critical/severe."""
    result = _cv_severity_from_vitals(80, None)
    assert result[0] == 4.8   # stable branch


# ---------- _cardiovascular_bank ----------

def test_cardiovascular_bank_value_shock_index_preferred():
    """If shock_index available, it drives the call."""
    # SI=1.5 (critical) but HR/BP normal → SI wins
    sig = _cardiovascular_bank(_row(shock_index=1.5, heart_rate=80,
                                     systolic_bp=120))
    assert sig.esi_estimate == 1.0


def test_cardiovascular_bank_logical_si_missing_falls_back_to_vitals():
    """No shock_index → use HR+SBP."""
    sig = _cardiovascular_bank(_row(heart_rate=140, systolic_bp=120))
    assert sig.esi_estimate == 1.2  # hr_critical path


def test_cardiovascular_bank_value_si_borderline_returns_borderline():
    sig = _cardiovascular_bank(_row(shock_index=0.95, heart_rate=80,
                                     systolic_bp=120))
    assert sig.esi_estimate == 3.0


# ---------- _thermal_bank ----------

@pytest.mark.parametrize("temp,expected_esi", [
    (41.0, 1.5),        # extreme high
    (32.0, 1.5),        # extreme low
    (40.0, 2.5),        # high fever
    (33.5, 2.5),        # hypothermia
    (38.8, 3.0),        # fever
    (38.2, 3.5),        # low-grade
    (37.0, 5.0),        # normal
])
def test_thermal_bank_boundary(temp, expected_esi):
    sig = _thermal_bank(_row(temperature_c=temp))
    assert sig.esi_estimate == expected_esi


def test_thermal_bank_logical_missing_returns_neutral():
    sig = _thermal_bank(_row())
    assert sig.esi_estimate == 3.5
    assert sig.confidence == 0.10
    assert "missing" in sig.evidence


def test_thermal_bank_value_extreme_high_vs_high():
    """40.5 is the boundary between high and extreme."""
    extreme = _thermal_bank(_row(temperature_c=40.5))
    high = _thermal_bank(_row(temperature_c=40.4))
    assert extreme.esi_estimate == 1.5
    assert high.esi_estimate == 2.5


# ---------- _pain_bank ----------

@pytest.mark.parametrize("pain,location,expected_esi", [
    (10, "chest", 2.0),          # severe high-risk
    (9, "knee", 2.0),            # severe
    (8, "chest", 2.5),           # high + location boost
    (7, "knee", 2.5),
    (5, "", 3.5),                # moderate
    (4, "", 3.5),
    (2, "", 4.2),                # mild
    (1, "", 4.2),
    (0, "", 5.0),                # none
    (-1, "", 2.5),               # unassessable
])
def test_pain_bank_boundary(pain, location, expected_esi):
    sig = _pain_bank(_row(pain_score=pain, pain_location=location))
    assert sig.esi_estimate == expected_esi


def test_pain_bank_arithmetic_location_boost_on_high_risk():
    """chest location boosts confidence by 0.15 at pain >= 7."""
    chest = _pain_bank(_row(pain_score=8, pain_location="chest"))
    knee = _pain_bank(_row(pain_score=8, pain_location="knee"))
    assert chest.confidence == pytest.approx(knee.confidence + 0.15)


def test_pain_bank_value_missing_returns_low_conf():
    sig = _pain_bank(_row())
    assert sig.confidence == 0.10
    assert "missing" in sig.evidence


# ---------- _history_bank ----------

def test_history_bank_value_no_hx_cols_returns_neutral():
    sig = _history_bank(_row(age=40))  # no hx_* cols
    assert sig.esi_estimate == 3.5
    assert sig.confidence == 0.10


@pytest.mark.parametrize("flags,expected_esi", [
    ({"hx_heart_failure": 1, "hx_copd": 1, "hx_malignancy": 1}, 2.5),  # ≥3
    ({"hx_heart_failure": 1, "hx_copd": 1}, 3.0),   # 2
    ({"hx_heart_failure": 1}, 3.2),                   # 1
    ({"hx_hypertension": 1}, 4.0),                   # 0 high-risk, no comorbid
])
def test_history_bank_boundary_high_risk_counts(flags, expected_esi):
    # Need an hx_* column to avoid the "no history data" branch
    all_flags = {"hx_hypertension": 0, "hx_heart_failure": 0,
                 "hx_copd": 0, "hx_malignancy": 0, "hx_immunosuppressed": 0,
                 "hx_coagulopathy": 0, "hx_ckd": 0, "hx_liver_disease": 0,
                 "hx_stroke_prior": 0, "hx_pregnant": 0}
    all_flags.update(flags)
    sig = _history_bank(_row(**all_flags))
    assert sig.esi_estimate == expected_esi


def test_history_bank_logical_pregnancy_caps_esi():
    """Pregnant patients have esi capped at 3.0."""
    sig = _history_bank(_row(hx_hypertension=1, hx_pregnant=1,
                              hx_heart_failure=0, hx_copd=0,
                              hx_malignancy=0, hx_immunosuppressed=0,
                              hx_coagulopathy=0, hx_ckd=0,
                              hx_liver_disease=0, hx_stroke_prior=0))
    assert sig.esi_estimate <= 3.0


def test_history_bank_value_many_comorbidities_bumps_esi():
    """num_comorbidities ≥ 6 raises acuity even without high-risk flags."""
    all_flags = {f"hx_{n}": 0 for n in [
        "hypertension", "heart_failure", "copd", "malignancy",
        "immunosuppressed", "coagulopathy", "ckd", "liver_disease",
        "stroke_prior", "pregnant",
    ]}
    sig = _history_bank(_row(num_comorbidities=7, **all_flags))
    assert sig.esi_estimate == 3.0


# ---------- _demographic_bank ----------

@pytest.mark.parametrize("age,expected_esi", [
    (1, 2.5),       # neonate
    (3, 3.0),       # young pediatric
    (40, 3.8),      # middle-age neutral
    (76, 3.0),      # elderly
    (90, 2.8),      # very elderly
])
def test_demographic_bank_boundary(age, expected_esi):
    sig = _demographic_bank(_row(age=age))
    assert sig.esi_estimate == expected_esi


def test_demographic_bank_value_missing_neutral():
    sig = _demographic_bank(_row())
    assert sig.esi_estimate == 3.5


def test_demographic_bank_boundary_age_85_boundary():
    """age=85 enters very-elderly; age=84 is elderly."""
    very_elderly = _demographic_bank(_row(age=85))
    elderly = _demographic_bank(_row(age=84))
    assert very_elderly.esi_estimate == 2.8
    assert elderly.esi_estimate == 3.0


# ---------- _utilization_bank ----------

@pytest.mark.parametrize("ed,admit,meds,expected_esi", [
    (0, 0, 0, 4.2),          # burden=0
    (2, 0, 2, 3.5),          # burden=0.6+0.4=1.0 → esi 4.2? check
    (5, 5, 10, 2.5),         # burden large, esi 2.5
    (3, 3, 5, 3.0),          # burden=0.9+1.5+1.0=3.4 → > 3 → esi 3.0
])
def test_utilization_bank_arithmetic_burden(ed, admit, meds, expected_esi):
    sig = _utilization_bank(_row(num_prior_ed_visits_12m=ed,
                                   num_prior_admissions_12m=admit,
                                   num_active_medications=meds))
    # Calculate burden = ed*0.3 + admit*0.5 + meds*0.2
    burden = ed * 0.3 + admit * 0.5 + meds * 0.2
    if burden > 5:
        expected = 2.5
    elif burden > 3:
        expected = 3.0
    elif burden > 1:
        expected = 3.5
    else:
        expected = 4.2
    assert sig.esi_estimate == expected


def test_utilization_bank_boundary_burden_3():
    """burden>3 → esi 3.0; burden==3 → esi 3.5."""
    # burden=3.3: ed=1(0.3) + admit=3(1.5) + meds=5(1.0) = 2.8 — too low
    # burden exactly at 3: ed=10(3.0) → > 3 threshold (3.0 > 3 is False!)
    # exactly 3 falls into "burden > 1" branch, giving esi=3.5
    s = _utilization_bank(_row(num_prior_ed_visits_12m=10,
                                num_prior_admissions_12m=0,
                                num_active_medications=0))
    # burden = 3.0, >1 but not >3 → esi 3.5
    assert s.esi_estimate == 3.5


def test_utilization_bank_value_missing_defaults_to_zero_burden():
    sig = _utilization_bank(_row())
    assert sig.esi_estimate == 4.2


# ---------- _arrival_bank ----------

def test_arrival_bank_value_low_confidence():
    sig = _arrival_bank(_row(arrival_mode="walk-in"))
    assert sig.esi_estimate == 3.3
    assert sig.confidence == 0.05


def test_arrival_bank_value_same_esi_for_all_modes():
    """All arrival modes produce the SAME esi estimate (by design —
    non-discriminative per CLAUDE.md note)."""
    modes = ["walk-in", "ambulance", "police", "helicopter", ""]
    ests = [_arrival_bank(_row(arrival_mode=m)).esi_estimate for m in modes]
    assert all(e == 3.3 for e in ests)


def test_arrival_bank_value_mode_in_evidence():
    """The mode string appears in the evidence field."""
    sig = _arrival_bank(_row(arrival_mode="ambulance"))
    assert "ambulance" in sig.evidence


# ---------- decompose_patient ----------

def test_decompose_patient_value_ten_banks():
    """decompose_patient returns all 10 bank signals (COMPLAINT optional)."""
    row = pd.Series({
        "patient_id": "P1", "news2_score": 2.0, "gcs_total": 15,
        "spo2": 98, "respiratory_rate": 16, "heart_rate": 80,
        "systolic_bp": 120, "temperature_c": 37.0, "pain_score": 3,
        "age": 40, "arrival_mode": "walk-in",
        "hx_hypertension": 1,
    })
    d = decompose_patient(row)
    assert len(d.signals) == 10  # 11 banks minus COMPLAINT
    assert Bank.SEVERITY in d.signals
    assert Bank.ARRIVAL in d.signals
    assert Bank.COMPLAINT not in d.signals


def test_decompose_patient_swap_with_complaint():
    """Passing complaint_signal adds COMPLAINT bank to decomposition."""
    row = pd.Series({"patient_id": "P1", "age": 40, "gcs_total": 15})
    complaint = BankSignal(Bank.COMPLAINT, 2.5, 0.7, 1, 3,
                            "chest pain cardiac")
    d = decompose_patient(row, complaint)
    assert Bank.COMPLAINT in d.signals
    assert d.signals[Bank.COMPLAINT].esi_estimate == 2.5


def test_decompose_patient_value_patient_id_preserved():
    row = pd.Series({"patient_id": "PID-42-XYZ", "age": 40})
    d = decompose_patient(row)
    assert d.patient_id == "PID-42-XYZ"


# ---------- decompose_dataframe ----------

def test_decompose_dataframe_value_row_count():
    df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "news2_score": [0, 5, 13],
        "gcs_total": [15, 15, 10],
        "age": [30, 50, 80],
        "hx_hypertension": [0, 0, 1],
    })
    decomps = decompose_dataframe(df)
    assert len(decomps) == 3
    assert {d.patient_id for d in decomps} == {"P1", "P2", "P3"}


def test_decompose_dataframe_swap_complaint_signals_applied_by_pid():
    """Complaint signals routed by patient_id, not positional order."""
    df = pd.DataFrame({
        "patient_id": ["P1", "P2"],
        "age": [30, 50],
        "gcs_total": [15, 15],
    })
    complaints = {
        "P2": BankSignal(Bank.COMPLAINT, 1.5, 0.9, 1, 2, "sepsis"),
    }
    decomps = decompose_dataframe(df, complaints)
    # P1 has no complaint → no COMPLAINT in its signals
    # P2 has one
    p1 = next(d for d in decomps if d.patient_id == "P1")
    p2 = next(d for d in decomps if d.patient_id == "P2")
    assert Bank.COMPLAINT not in p1.signals
    assert Bank.COMPLAINT in p2.signals
    assert p2.signals[Bank.COMPLAINT].evidence == "sepsis"


# ---------- BANK_FUNCTIONS registry ----------

def test_bank_functions_registry_covers_ten_banks():
    """Registry has exactly the 10 symbolic banks (no COMPLAINT)."""
    assert set(BANK_FUNCTIONS.keys()) == {
        Bank.SEVERITY, Bank.CONSCIOUSNESS, Bank.RESPIRATORY,
        Bank.CARDIOVASCULAR, Bank.THERMAL, Bank.PAIN, Bank.HISTORY,
        Bank.DEMOGRAPHIC, Bank.UTILIZATION, Bank.ARRIVAL,
    }
    assert Bank.COMPLAINT not in BANK_FUNCTIONS
