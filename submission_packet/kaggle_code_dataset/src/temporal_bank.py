"""
Temporal bank — snapshot-bootstrapped trajectory signals.

Built from two independent sources:

1. **Trajectory marker extraction** from `chief_complaint_raw`. The synthetic
   dataset encodes trajectory markers as a standardized vocabulary:
   {worsening over hours, worsening, onset today, since yesterday, constant,
   intermittent, for N days, chronic}. On this dataset the markers have weak
   direct ESI correlation (all mean ESI 3.30-3.34 except chronic at 4.41) —
   but they're kept as features for real-world generalizability and as audit
   signals in the clinician-facing output.

2. **Population-conditional vitals deviation** — the appendicitis-stable
   signature. For each (complaint_base, age_group), compute E[NEWS2]. A
   patient whose NEWS2 is much LOWER than their complaint+age cohort expects
   is suspicious: the complaint category implies acuity but the instantaneous
   vitals look reassuring. This is clinically the "symptoms are stable but
   the clock is ticking" pattern. Empirically, on 2026-04-17 train data,
   **34 patients match "severe-complaint mean NEWS2 ≥ 5 AND own NEWS2 ≤ 2"
   and 100% of them are ESI 2** — a small but high-precision paradox class.

This bank emits:
  - `temporal_trajectory_code` (categorical): encoded marker
  - `temporal_has_marker` (0/1): any marker found
  - `temporal_chronic` (0/1): chronic flag (the only high-signal marker)
  - `temporal_news2_deviation`: patient's NEWS2 minus cohort mean
  - `temporal_paradox_flag`: (complaint_expected_NEWS2 ≥ 5) AND (patient NEWS2 ≤ 2)
  - `temporal_expected_news2`: cohort mean for this (complaint_base, age_group)

Fits are deterministic — no target leakage — because cohort means are computed
from vitals only, not from the triage label.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Trajectory marker extraction
# ---------------------------------------------------------------------------

TRAJECTORY_CODES = {
    "rapid_worsening": 1,
    "gradual_worsening": 2,
    "onset_today": 3,
    "subacute": 4,
    "days_duration": 5,
    "constant": 6,
    "intermittent": 7,
    "chronic": 8,
    "unspecified": 0,
}


def _extract_trajectory(text: str) -> str:
    """Return trajectory class for a chief complaint string."""
    if not isinstance(text, str) or not text:
        return "unspecified"
    t = text.lower()
    if "worsening over hours" in t:
        return "rapid_worsening"
    if "worsening" in t:
        return "gradual_worsening"
    if "onset today" in t:
        return "onset_today"
    if "since yesterday" in t:
        return "subacute"
    if "chronic" in t:
        return "chronic"
    if re.search(r"for \d+ days?", t):
        return "days_duration"
    if "constant" in t:
        return "constant"
    if "intermittent" in t:
        return "intermittent"
    return "unspecified"


def _extract_base_condition(text: str) -> str:
    """Extract the base condition (pre-modifier) from a chief complaint.

    Duplicates feature_engine's helper to keep this module standalone.
    """
    if not isinstance(text, str) or not text:
        return "unknown"
    base = re.split(r"[,\uff0c]", text)[0].strip().lower()
    for prefix in ["severe ", "moderate ", "mild ", "minor ", "acute ", "light "]:
        if base.startswith(prefix):
            base = base[len(prefix):]
    return base.strip() or "unknown"


# ---------------------------------------------------------------------------
# Population-conditional cohort statistics
# ---------------------------------------------------------------------------

def fit_cohort_expectations(
    train_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    vital_cols: tuple[str, ...] = (
        "news2_score", "heart_rate", "respiratory_rate", "spo2",
        "temperature_c", "shock_index",
    ),
) -> pd.DataFrame:
    """Compute E[vital | base_condition, age_group] over train data.

    Returns a DataFrame keyed by (base, age_group) with one column per vital.
    Caller passes this to `build_temporal_features` at inference time.
    """
    merged = train_df.merge(complaints_df[["patient_id", "chief_complaint_raw"]],
                            on="patient_id", how="left")
    merged["base"] = merged["chief_complaint_raw"].apply(_extract_base_condition)
    grp = merged.groupby(["base", "age_group"], observed=True)
    cohort = grp[list(vital_cols)].mean()
    # Global fallback: mean per vital across all training data
    global_means = merged[list(vital_cols)].mean()
    for col in vital_cols:
        cohort[col] = cohort[col].fillna(global_means[col])
    return cohort.reset_index()


# ---------------------------------------------------------------------------
# Per-patient temporal features
# ---------------------------------------------------------------------------

def build_temporal_features(
    df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    cohort_expectations: pd.DataFrame,
    paradox_news2_threshold: float = 2.0,
    paradox_cohort_threshold: float = 5.0,
) -> pd.DataFrame:
    """Emit temporal-bank feature columns for the given patients.

    Args:
        df: patient rows with `patient_id`, `news2_score`, `age_group`, etc.
        complaints_df: chief complaint text for these patients.
        cohort_expectations: output of `fit_cohort_expectations` on train data.
        paradox_news2_threshold: "low NEWS2" cutoff for paradox detection.
        paradox_cohort_threshold: "cohort typically severe" cutoff.

    Returns:
        DataFrame indexed like `df` with columns:
            temporal_trajectory_code, temporal_has_marker, temporal_chronic,
            temporal_expected_news2, temporal_news2_deviation,
            temporal_paradox_flag
    """
    d = df.merge(complaints_df[["patient_id", "chief_complaint_raw"]],
                 on="patient_id", how="left")
    traj = d["chief_complaint_raw"].apply(_extract_trajectory)
    base = d["chief_complaint_raw"].apply(_extract_base_condition)
    age_group = d["age_group"].astype(str)

    # Join cohort expectations
    cohort_key = pd.DataFrame({"base": base.values,
                               "age_group": age_group.values})
    joined = cohort_key.merge(cohort_expectations, on=["base", "age_group"],
                              how="left")
    exp_news2 = joined["news2_score"].values

    # Fallback for unseen (base, age_group) pairs → use global mean
    global_mean = cohort_expectations["news2_score"].mean()
    exp_news2 = np.where(np.isnan(exp_news2), global_mean, exp_news2)

    own_news2 = d["news2_score"].fillna(global_mean).values
    deviation = own_news2 - exp_news2

    paradox = ((exp_news2 >= paradox_cohort_threshold) &
               (own_news2 <= paradox_news2_threshold)).astype(int)

    out = pd.DataFrame({
        "temporal_trajectory_code": traj.map(TRAJECTORY_CODES).fillna(0).astype(int).values,
        "temporal_has_marker": (traj != "unspecified").astype(int).values,
        "temporal_chronic": (traj == "chronic").astype(int).values,
        "temporal_expected_news2": exp_news2,
        "temporal_news2_deviation": deviation,
        "temporal_paradox_flag": paradox,
    }, index=df.index)
    return out


# ---------------------------------------------------------------------------
# Fold-safe wrapper
# ---------------------------------------------------------------------------

def fit_fold_safe_temporal_features(
    train_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Build OOF temporal features across CV folds without leakage.

    Cohort expectations for each fold are fit on that fold's train partition
    only. No target column is consulted during fitting — but cohort statistics
    are dataset-specific, so fold isolation keeps the features reproducible
    across identical CV configurations.
    """
    pieces = []
    for tr, va in fold_indices:
        fold_train = train_df.iloc[tr]
        fold_val = train_df.iloc[va]
        cohort = fit_cohort_expectations(fold_train, complaints_df)
        val_feats = build_temporal_features(fold_val, complaints_df, cohort)
        pieces.append(val_feats)
    return pd.concat(pieces).sort_index()
