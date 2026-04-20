"""
Clinical bank decomposition for ESI triage prediction.

Maps patient features into orthogonal clinical dimensions (banks),
each producing an ESI range estimate with confidence. This is the
data geometry layer — the core insight from NCEMS ported to triage.

Banks are orthogonal clinical axes:
- SEVERITY: NEWS2 composite score (dominant signal, r=0.815)
- CONSCIOUSNESS: GCS + mental status
- RESPIRATORY: SpO2 + respiratory rate
- CARDIOVASCULAR: HR, BP, MAP, shock index
- THERMAL: Temperature
- PAIN: Pain score + location
- COMPLAINT: Chief complaint text → severity category
- HISTORY: Comorbidity risk profile (25 binary hx_* flags)
- DEMOGRAPHIC: Age extremes, sex
- UTILIZATION: Prior ED visits, admissions, active medications
- ARRIVAL: Mode of arrival, transport origin
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Bank(Enum):
    SEVERITY = "severity"
    CONSCIOUSNESS = "consciousness"
    RESPIRATORY = "respiratory"
    CARDIOVASCULAR = "cardiovascular"
    THERMAL = "thermal"
    PAIN = "pain"
    COMPLAINT = "complaint"
    HISTORY = "history"
    DEMOGRAPHIC = "demographic"
    UTILIZATION = "utilization"
    ARRIVAL = "arrival"


@dataclass(frozen=True)
class BankSignal:
    """A single bank's ESI estimate with confidence and provenance."""
    bank: Bank
    esi_estimate: float        # continuous ESI estimate (1.0-5.0)
    confidence: float          # 0.0-1.0
    esi_floor: int             # hard minimum ESI (1-5), 0 = no constraint
    esi_ceiling: int           # hard maximum ESI (1-5), 6 = no constraint
    evidence: str              # human-readable provenance


@dataclass
class BankDecomposition:
    """All bank signals for a single patient."""
    patient_id: str
    signals: dict[Bank, BankSignal] = field(default_factory=dict)

    def add(self, signal: BankSignal):
        self.signals[signal.bank] = signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val, default: float) -> float:
    """Extract a float from a Series value, returning default if missing/NaN."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return float(val)


def _is_missing(val) -> bool:
    """Check if a value is None or NaN."""
    if val is None:
        return True
    return isinstance(val, float) and math.isnan(val)


# ---------------------------------------------------------------------------
# Per-bank ESI inference functions
# ---------------------------------------------------------------------------

def _severity_bank(row: pd.Series) -> BankSignal:
    """NEWS2 score → ESI estimate. This is the dominant signal (r=0.815)."""
    raw = row.get("news2_score")
    if _is_missing(raw):
        return BankSignal(Bank.SEVERITY, 3.0, 0.0, 0, 6, "news2 missing")
    news2 = float(raw)

    # NEWS2 → ESI mapping CALIBRATED from 80K training examples
    # Each estimate = empirical mean ESI for that NEWS2 range
    if news2 >= 13:
        return BankSignal(Bank.SEVERITY, 1.5, 0.95, 1, 2,
                          f"news2={news2:.0f} critical")
    if news2 >= 9:
        return BankSignal(Bank.SEVERITY, 1.9, 0.92, 1, 2,
                          f"news2={news2:.0f} very-high")
    if news2 >= 7:
        return BankSignal(Bank.SEVERITY, 2.2, 0.88, 1, 3,
                          f"news2={news2:.0f} high")
    if news2 >= 5:
        return BankSignal(Bank.SEVERITY, 2.8, 0.80, 2, 4,
                          f"news2={news2:.0f} medium")
    if news2 >= 3:
        return BankSignal(Bank.SEVERITY, 3.1, 0.70, 2, 4,
                          f"news2={news2:.0f} low-medium")
    if news2 >= 2:
        return BankSignal(Bank.SEVERITY, 3.4, 0.65, 2, 5,
                          f"news2={news2:.0f} low")
    if news2 >= 1:
        return BankSignal(Bank.SEVERITY, 3.8, 0.60, 3, 5,
                          f"news2={news2:.0f} low")
    # NEWS2 = 0: empirical mean = 4.23
    return BankSignal(Bank.SEVERITY, 4.2, 0.55, 3, 5,
                      f"news2={news2:.0f} minimal")


def _consciousness_bank(row: pd.Series) -> BankSignal:
    """GCS + mental status → consciousness-based ESI."""
    gcs = int(_safe_float(row.get("gcs_total"), 15.0))
    mental = row.get("mental_status_triage", "")

    # GCS ≤ 8: comatose → ESI 1 (resuscitation)
    if gcs <= 8:
        return BankSignal(Bank.CONSCIOUSNESS, 1.0, 0.98, 1, 1,
                          f"gcs={gcs} comatose")
    # GCS 9-12: severely impaired
    if gcs <= 12:
        return BankSignal(Bank.CONSCIOUSNESS, 1.5, 0.90, 1, 2,
                          f"gcs={gcs} severely impaired")
    # GCS 13-14: mildly impaired
    if gcs <= 14:
        conf = 0.75
        esi = 2.5
        if mental == "unresponsive":
            esi = 1.5
            conf = 0.92
        elif mental == "confused":
            esi = 2.5
            conf = 0.75
        elif mental == "drowsy":
            esi = 3.0
            conf = 0.65
        return BankSignal(Bank.CONSCIOUSNESS, esi, conf, 1, 3,
                          f"gcs={gcs} mental={mental}")
    # GCS 15: normal
    if mental == "unresponsive":
        # GCS 15 but unresponsive is contradictory → trust mental status
        return BankSignal(Bank.CONSCIOUSNESS, 1.5, 0.85, 1, 2,
                          "gcs=15 but unresponsive")
    if mental == "agitated":
        return BankSignal(Bank.CONSCIOUSNESS, 2.5, 0.55, 2, 4,
                          "gcs=15 agitated")
    if mental == "confused":
        return BankSignal(Bank.CONSCIOUSNESS, 2.8, 0.60, 2, 4,
                          "gcs=15 confused")
    if mental == "drowsy":
        return BankSignal(Bank.CONSCIOUSNESS, 2.6, 0.55, 2, 4,
                          "gcs=15 drowsy")
    # alert + GCS 15 — empirical mean ESI = 3.82
    return BankSignal(Bank.CONSCIOUSNESS, 3.8, 0.30, 3, 5,
                      "gcs=15 alert (neutral)")


def _respiratory_bank(row: pd.Series) -> BankSignal:
    """SpO2 + respiratory rate → respiratory status ESI."""
    spo2 = _safe_float(row.get("spo2"), 97.0)
    rr = _safe_float(row.get("respiratory_rate"), 16.0)

    # Critical: SpO2 < 85% or RR < 8 or RR > 35
    if spo2 < 85 or rr < 8 or rr > 35:
        return BankSignal(Bank.RESPIRATORY, 1.0, 0.95, 1, 1,
                          f"spo2={spo2:.0f} rr={rr:.0f} critical")
    # Severe: SpO2 85-89% or RR 30-35
    if spo2 < 90 or rr > 30:
        return BankSignal(Bank.RESPIRATORY, 1.5, 0.88, 1, 2,
                          f"spo2={spo2:.0f} rr={rr:.0f} severe")
    # Moderate: SpO2 90-93% or RR 25-30
    if spo2 < 94 or rr > 25:
        return BankSignal(Bank.RESPIRATORY, 2.5, 0.75, 2, 3,
                          f"spo2={spo2:.0f} rr={rr:.0f} moderate")
    # Mild: SpO2 94-95% or RR 22-25
    if spo2 < 96 or rr > 22:
        return BankSignal(Bank.RESPIRATORY, 3.2, 0.55, 2, 4,
                          f"spo2={spo2:.0f} rr={rr:.0f} mild")
    # Normal: SpO2 ≥ 96% and RR 12-20
    if rr < 12:
        return BankSignal(Bank.RESPIRATORY, 3.5, 0.45, 3, 5,
                          f"spo2={spo2:.0f} rr={rr:.0f} low-rr")
    return BankSignal(Bank.RESPIRATORY, 5.0, 0.35, 3, 5,
                      f"spo2={spo2:.0f} rr={rr:.0f} normal")


def _cv_severity_from_shock_index(si: float) -> tuple[float, float, int, int, str] | None:
    """Classify cardiovascular severity from shock index alone."""
    if si > 1.3:
        return (1.0, 0.92, 1, 2, f"shock_index={si:.2f} critical")
    if si > 1.0:
        return (2.0, 0.80, 1, 3, f"shock_index={si:.2f} elevated")
    if si > 0.9:
        return (3.0, 0.55, 2, 4, f"shock_index={si:.2f} borderline")
    return None


def _cv_severity_from_vitals(hr: float, sbp: float | None) -> tuple[float, float, int, int, str]:
    """Classify cardiovascular severity from HR and SBP."""
    hr_critical = hr > 130 or hr < 40
    hr_severe = hr > 110 or hr < 50
    sbp_critical = sbp is not None and sbp < 80
    sbp_severe = sbp is not None and (sbp < 90 or sbp > 200)

    if hr_critical or sbp_critical:
        return (1.2, 0.90, 1, 2, f"hr={hr:.0f} sbp={sbp} critical")
    if hr_severe or sbp_severe:
        return (2.2, 0.70, 1, 3, f"hr={hr:.0f} sbp={sbp} severe")
    return (4.8, 0.30, 3, 5, f"hr={hr:.0f} sbp={sbp} stable")


def _cardiovascular_bank(row: pd.Series) -> BankSignal:
    """HR, BP, MAP, shock index → hemodynamic status ESI."""
    hr = _safe_float(row.get("heart_rate"), 80.0)
    sbp = None if _is_missing(row.get("systolic_bp")) else float(row["systolic_bp"])
    si_raw = row.get("shock_index")

    # Shock index is most informative if available
    if not _is_missing(si_raw):
        result = _cv_severity_from_shock_index(float(si_raw))
        if result:
            return BankSignal(Bank.CARDIOVASCULAR, *result)

    # Fall back to HR + SBP
    esi, conf, floor, ceil, evidence = _cv_severity_from_vitals(hr, sbp)
    return BankSignal(Bank.CARDIOVASCULAR, esi, conf, floor, ceil, evidence)


def _thermal_bank(row: pd.Series) -> BankSignal:
    """Temperature → thermal/infection status ESI."""
    raw = row.get("temperature_c")
    if _is_missing(raw):
        return BankSignal(Bank.THERMAL, 3.5, 0.10, 0, 6, "temp missing")
    temp = float(raw)

    if temp >= 40.5 or temp < 33.0:
        return BankSignal(Bank.THERMAL, 1.5, 0.85, 1, 2,
                          f"temp={temp:.1f} extreme")
    if temp >= 39.5 or temp < 34.0:
        return BankSignal(Bank.THERMAL, 2.5, 0.65, 2, 3,
                          f"temp={temp:.1f} high fever/hypothermia")
    if temp >= 38.5:
        return BankSignal(Bank.THERMAL, 3.0, 0.50, 2, 4,
                          f"temp={temp:.1f} fever")
    if temp >= 38.0:
        return BankSignal(Bank.THERMAL, 3.5, 0.35, 3, 5,
                          f"temp={temp:.1f} low-grade fever")
    return BankSignal(Bank.THERMAL, 5.0, 0.15, 3, 5,
                      f"temp={temp:.1f} normal")


def _pain_bank(row: pd.Series) -> BankSignal:
    """Pain score + location → pain-based ESI contribution."""
    raw = row.get("pain_score")
    location = row.get("pain_location", "")

    if _is_missing(raw):
        return BankSignal(Bank.PAIN, 3.5, 0.10, 0, 6, "pain missing")
    pain = float(raw)

    # pain_score = -1 means "unable to assess" (common in altered mental status)
    if pain < 0:
        return BankSignal(Bank.PAIN, 2.5, 0.30, 0, 6,
                          "pain unassessable (altered?)")

    # High-risk locations amplify pain signal
    high_risk_locations = {"chest", "head", "abdomen", "pelvis"}
    location_boost = 0.15 if str(location).lower() in high_risk_locations else 0.0

    if pain >= 9:
        return BankSignal(Bank.PAIN, 2.0, 0.60 + location_boost, 1, 3,
                          f"pain={pain:.0f} loc={location} severe")
    if pain >= 7:
        return BankSignal(Bank.PAIN, 2.5, 0.45 + location_boost, 2, 4,
                          f"pain={pain:.0f} loc={location} high")
    if pain >= 4:
        return BankSignal(Bank.PAIN, 3.5, 0.30, 3, 5,
                          f"pain={pain:.0f} moderate")
    if pain >= 1:
        return BankSignal(Bank.PAIN, 4.2, 0.25, 3, 5,
                          f"pain={pain:.0f} mild")
    return BankSignal(Bank.PAIN, 5.0, 0.30, 4, 5,
                      f"pain={pain:.0f} none")


def _history_bank(row: pd.Series) -> BankSignal:
    """Comorbidity profile → risk-adjusted ESI modifier."""
    hx_cols = [c for c in row.index if c.startswith("hx_")]
    if not hx_cols:
        return BankSignal(Bank.HISTORY, 3.5, 0.10, 0, 6, "no history data")

    hx_count = sum(1 for c in hx_cols if row.get(c) == 1)
    num_comorbidities = int(_safe_float(row.get("num_comorbidities"), float(hx_count)))

    # High-risk comorbidities that escalate triage
    high_risk = sum(1 for c in [
        "hx_heart_failure", "hx_copd", "hx_malignancy",
        "hx_immunosuppressed", "hx_coagulopathy", "hx_ckd",
        "hx_liver_disease", "hx_stroke_prior"
    ] if row.get(c) == 1)

    # Pregnancy is a special modifier
    pregnant = row.get("hx_pregnant", 0) == 1

    if high_risk >= 3:
        esi = 2.5
        conf = 0.45
    elif high_risk >= 2:
        esi = 3.0
        conf = 0.35
    elif high_risk >= 1:
        esi = 3.2
        conf = 0.25
    elif int(num_comorbidities) >= 6:
        esi = 3.0
        conf = 0.30
    else:
        esi = 4.0
        conf = 0.15

    if pregnant:
        esi = min(esi, 3.0)
        conf = max(conf, 0.30)

    return BankSignal(Bank.HISTORY, esi, conf, 0, 6,
                      f"comorbidities={num_comorbidities} high_risk={high_risk}")


def _demographic_bank(row: pd.Series) -> BankSignal:
    """Age extremes and demographics → ESI modifier."""
    raw = row.get("age")
    if _is_missing(raw):
        return BankSignal(Bank.DEMOGRAPHIC, 3.5, 0.10, 0, 6, "age missing")
    age = int(raw)

    # Pediatric (< 5 years): automatic acuity concern
    if age < 2:
        return BankSignal(Bank.DEMOGRAPHIC, 2.5, 0.50, 1, 3,
                          f"age={age} neonate/infant")
    if age < 5:
        return BankSignal(Bank.DEMOGRAPHIC, 3.0, 0.35, 2, 4,
                          f"age={age} young pediatric")
    # Geriatric (≥ 80): higher acuity concern
    if age >= 85:
        return BankSignal(Bank.DEMOGRAPHIC, 2.8, 0.40, 1, 4,
                          f"age={age} very elderly")
    if age >= 75:
        return BankSignal(Bank.DEMOGRAPHIC, 3.0, 0.30, 2, 4,
                          f"age={age} elderly")
    # Middle age — neutral
    return BankSignal(Bank.DEMOGRAPHIC, 3.8, 0.10, 0, 6,
                      f"age={age} neutral")


def _utilization_bank(row: pd.Series) -> BankSignal:
    """Prior ED visits, admissions, medications → chronic disease burden."""
    prior_ed = int(_safe_float(row.get("num_prior_ed_visits_12m"), 0.0))
    prior_admit = int(_safe_float(row.get("num_prior_admissions_12m"), 0.0))
    num_meds = int(_safe_float(row.get("num_active_medications"), 0.0))

    # Heavy utilizers with many medications → higher acuity tendency
    burden = prior_ed * 0.3 + prior_admit * 0.5 + num_meds * 0.2
    if burden > 5:
        esi = 2.5
        conf = 0.35
    elif burden > 3:
        esi = 3.0
        conf = 0.25
    elif burden > 1:
        esi = 3.5
        conf = 0.15
    else:
        esi = 4.2
        conf = 0.10

    return BankSignal(Bank.UTILIZATION, esi, conf, 0, 6,
                      f"ed={prior_ed} admit={prior_admit} meds={num_meds}")


def _arrival_bank(row: pd.Series) -> BankSignal:
    """Arrival mode and transport origin → acuity signal.

    CALIBRATION NOTE: Empirically all arrival modes have mean ESI ≈ 3.32
    in training data. Arrival mode has near-zero discriminative power
    (likely synthetic data artifact). Bank confidence set very low.
    """
    mode = str(row.get("arrival_mode", "")).lower()
    # All modes empirically ≈ 3.32 — minimal differentiation
    return BankSignal(Bank.ARRIVAL, 3.3, 0.05, 0, 6,
                      f"mode={mode} (non-discriminative)")


# ---------------------------------------------------------------------------
# Bank registry
# ---------------------------------------------------------------------------

BANK_FUNCTIONS = {
    Bank.SEVERITY: _severity_bank,
    Bank.CONSCIOUSNESS: _consciousness_bank,
    Bank.RESPIRATORY: _respiratory_bank,
    Bank.CARDIOVASCULAR: _cardiovascular_bank,
    Bank.THERMAL: _thermal_bank,
    Bank.PAIN: _pain_bank,
    Bank.HISTORY: _history_bank,
    Bank.DEMOGRAPHIC: _demographic_bank,
    Bank.UTILIZATION: _utilization_bank,
    Bank.ARRIVAL: _arrival_bank,
    # COMPLAINT bank is handled separately via complaint_lexicon.py
}


def decompose_patient(row: pd.Series, complaint_signal: BankSignal | None = None) -> BankDecomposition:
    """Decompose a single patient into orthogonal bank signals."""
    patient_id = str(row.get("patient_id", "unknown"))
    decomp = BankDecomposition(patient_id=patient_id)

    for bank, func in BANK_FUNCTIONS.items():
        signal = func(row)
        decomp.add(signal)

    if complaint_signal is not None:
        decomp.add(complaint_signal)

    return decomp


def decompose_dataframe(df: pd.DataFrame,
                        complaint_signals: dict[str, BankSignal] | None = None
                        ) -> list[BankDecomposition]:
    """Decompose all patients in a DataFrame."""
    results = []
    for idx, row in df.iterrows():
        pid = str(row.get("patient_id", idx))
        cs = complaint_signals.get(pid) if complaint_signals else None
        results.append(decompose_patient(row, cs))
    return results
