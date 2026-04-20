"""
Tier-B feature-integration layer.

Wraps the temporal bank, clinician style banks, and confidence calibrator
into a single fold-safe builder that emits model-ready feature columns.

Usage in CV (fold-safe):
    fitted = fit_tier_b_artifacts(train_fold, complaints, history)
    tr_feats = build_tier_b_features(train_fold, complaints, history, fitted)
    va_feats = build_tier_b_features(val_fold,   complaints, history, fitted)
    # Merge tr_feats/va_feats with the rest of the feature matrix.

Usage at submission time:
    fitted = fit_tier_b_artifacts(train, complaints, history)
    train_feats = build_tier_b_features(train, complaints, history, fitted)
    test_feats  = build_tier_b_features(test,  complaints, history, fitted)

Key design choice: pure computation, no hidden globals. Each fold gets its
own `TierBArtifacts` frozen dataclass holding the style banks, cohort
expectation table, and confidence calibrator. The caller controls fold
isolation explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .banks import decompose_dataframe
from .clinician_style import (
    StyleBank, calibrate, fit_confidence_calibrator,
    fit_style_banks, style_features_for_patients,
    ConfidenceCalibrator,
)
from .complaint_lexicon import classify_complaints_batch
from .temporal_bank import build_temporal_features, fit_cohort_expectations


@dataclass(frozen=True)
class TierBArtifacts:
    """Everything fit from a training partition that Tier-B needs at apply time."""
    nurse_banks: dict[str, StyleBank]
    site_banks: dict[str, StyleBank]
    pop_dist: np.ndarray
    cohort_expectations: pd.DataFrame
    confidence_calibrator: ConfidenceCalibrator


def _top_bank_confidence(decomps) -> np.ndarray:
    out = np.zeros(len(decomps))
    for i, d in enumerate(decomps):
        confs = [s.confidence for s in d.signals.values()]
        out[i] = max(confs) if confs else 0.0
    return out


def fit_tier_b_artifacts(train_df: pd.DataFrame,
                          complaints_df: pd.DataFrame,
                          history_df: pd.DataFrame,
                          *, smoothing: float = 20.0) -> TierBArtifacts:
    """Fit all Tier-B artifacts from a training partition.

    No side effects — returns a frozen dataclass. Safe for fold-local use.
    """
    nurse, pop = fit_style_banks(train_df, "triage_nurse_id",
                                  smoothing=smoothing)
    site, _ = fit_style_banks(train_df, "site_id", smoothing=smoothing)
    cohort = fit_cohort_expectations(train_df, complaints_df)

    # Confidence calibrator needs bank decomposition on training rows
    merged = train_df.merge(history_df, on="patient_id", how="left",
                             suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    cc = classify_complaints_batch(
        complaints_df[complaints_df.patient_id.isin(train_df.patient_id)])
    decomps = decompose_dataframe(merged, cc)
    top_conf = _top_bank_confidence(decomps)
    cal = fit_confidence_calibrator(train_df, top_conf, n_bins=10)

    return TierBArtifacts(
        nurse_banks=nurse, site_banks=site, pop_dist=pop,
        cohort_expectations=cohort, confidence_calibrator=cal,
    )


def build_tier_b_features(
    df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    history_df: pd.DataFrame,
    artifacts: TierBArtifacts,
) -> pd.DataFrame:
    """Emit Tier-B model-ready features for the given patients.

    Returned columns (prefixed to avoid collisions):
      temporal_trajectory_code, temporal_has_marker, temporal_chronic,
      temporal_news2_deviation, temporal_paradox_flag,
      style_nurse_over_bias, style_nurse_under_bias, style_nurse_expected_esi,
      style_nurse_entropy, style_site_l1_dev, style_combined_over_bias,
      style_site_over_bias, style_site_under_bias, style_nurse_l1_dev,
      tier_b_calibrated_severe_prob

    All features are safe at test time — none depend on the target.
    `tier_b_calibrated_severe_prob` is a post-triage severe-outcome
    probability estimated from the pipeline's top-bank confidence; it's
    fit on train-time disposition/ed_los_hours but applied as a forward
    calculation from patient banks only.
    """
    # Temporal bank
    temp = build_temporal_features(df, complaints_df,
                                    artifacts.cohort_expectations)
    temp = temp[["temporal_trajectory_code", "temporal_has_marker",
                 "temporal_chronic", "temporal_news2_deviation",
                 "temporal_paradox_flag"]]

    # Clinician style
    style = style_features_for_patients(df, artifacts.nurse_banks,
                                         artifacts.site_banks,
                                         artifacts.pop_dist)

    # Calibrated severe-outcome probability — needs bank decomposition
    merged = df.merge(history_df, on="patient_id", how="left",
                       suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    cc = classify_complaints_batch(
        complaints_df[complaints_df.patient_id.isin(df.patient_id)])
    decomps = decompose_dataframe(merged, cc)
    top_conf = _top_bank_confidence(decomps)
    severe_prob = calibrate(artifacts.confidence_calibrator, top_conf)

    # Ensure alignment on df.index
    temp.index = df.index
    style.index = df.index
    severe_series = pd.Series(severe_prob, index=df.index,
                               name="tier_b_calibrated_severe_prob")

    return pd.concat([temp, style, severe_series], axis=1)
