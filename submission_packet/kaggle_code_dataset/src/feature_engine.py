"""
Feature engineering pipeline for CatBoost model.

Builds domain-informed features from raw patient data + bank signals.
The model handles the 10-15% residual that deterministic rules and
coherence can't resolve with high confidence.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Chief complaint text features
# ---------------------------------------------------------------------------

_KEYWORD_GROUPS: dict[str, list[str]] = {
    "pain": ["pain", "ache", "sore", "tender", "hurt"],
    "respiratory": ["breath", "dyspnea", "cough", "wheeze", "asthma"],
    "cardiac": ["chest", "palpitation", "syncope", "cardiac"],
    "neuro": ["headache", "seizure", "stroke", "dizzy", "numb", "weakness"],
    "gi": ["nausea", "vomit", "diarrhea", "abdominal", "bleeding"],
    "psych": ["suicidal", "anxiety", "agitated", "psychosis", "hallucin"],
    "trauma": ["fall", "injury", "fracture", "laceration", "wound", "trauma"],
    "infection": ["fever", "rigor", "sepsis", "abscess", "cellulitis"],
    "severity_high": ["severe", "acute", "sudden", "worst", "uncontrolled"],
    "severity_low": ["mild", "chronic", "intermittent", "stable", "minor"],
}


def _extract_text_features(texts: pd.Series) -> pd.DataFrame:
    """Extract keyword-count features from chief complaint text."""
    filled = texts.fillna("").astype(str).str.lower()
    features = {}

    features["cc_len"] = filled.str.len()
    features["cc_word_count"] = filled.str.split().str.len().fillna(0).astype(int)

    for group_name, keywords in _KEYWORD_GROUPS.items():
        pattern = "|".join(re.escape(k) for k in keywords)
        features[f"cc_{group_name}"] = filled.str.contains(pattern, regex=True).astype(int)

    return pd.DataFrame(features, index=texts.index)


# ---------------------------------------------------------------------------
# Clinical interaction features
# ---------------------------------------------------------------------------

def _build_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build domain-informed clinical interaction features."""
    features = {}

    # Vitals interactions
    hr = df.get("heart_rate", pd.Series(dtype=float))
    rr = df.get("respiratory_rate", pd.Series(dtype=float))
    spo2 = df.get("spo2", pd.Series(dtype=float))
    sbp = df.get("systolic_bp", pd.Series(dtype=float))
    temp = df.get("temperature_c", pd.Series(dtype=float))
    gcs = df.get("gcs_total", pd.Series(dtype=float))
    news2 = df.get("news2_score", pd.Series(dtype=float))
    age = df.get("age", pd.Series(dtype=float))
    pain = df.get("pain_score", pd.Series(dtype=float))

    # HR × RR interaction (both elevated = distress)
    features["hr_rr_product"] = hr * rr
    features["hr_rr_ratio"] = hr / (rr + 1e-3)

    # SpO2 × RR (compensatory breathing)
    features["spo2_rr_gap"] = spo2 - rr  # healthy: ~80; distressed: <70

    # Age × NEWS2 (elderly with high NEWS2 = worse)
    features["age_news2"] = age * news2

    # GCS deficit (15 - GCS, so higher = worse)
    features["gcs_deficit"] = 15 - gcs

    # Temperature deviation from normal (37°C)
    features["temp_deviation"] = (temp - 37.0).abs()

    # Pain-adjusted severity (pain + NEWS2 combined)
    features["pain_news2"] = pain.clip(lower=0) + news2

    # Vital instability score (count of deranged vitals)
    features["vital_instability"] = (
        (hr > 100).astype(int)
        + (hr < 60).astype(int)
        + (rr > 22).astype(int)
        + (rr < 10).astype(int)
        + (spo2 < 94).astype(int)
        + (sbp < 90).astype(int)
        + (sbp > 180).astype(int)
        + (temp > 38.5).astype(int)
        + (temp < 35.5).astype(int)
        + (gcs < 15).astype(int)
    )

    # History burden (sum of high-risk comorbidities)
    hx_cols = [c for c in df.columns if c.startswith("hx_")]
    if hx_cols:
        features["hx_total"] = df[hx_cols].sum(axis=1)
        high_risk_hx = [
            "hx_heart_failure", "hx_copd", "hx_malignancy",
            "hx_immunosuppressed", "hx_coagulopathy", "hx_ckd",
            "hx_liver_disease", "hx_stroke_prior",
        ]
        present_high_risk = [c for c in high_risk_hx if c in df.columns]
        if present_high_risk:
            features["hx_high_risk"] = df[present_high_risk].sum(axis=1)

    # ESI 4 vs 5 discriminators (resource prediction features)
    num_meds = df.get("num_active_medications", pd.Series(0, index=df.index))
    prior_ed = df.get("num_prior_ed_visits_12m", pd.Series(0, index=df.index))

    features["resource_proxy"] = num_meds * 0.3 + prior_ed * 0.2
    features["low_acuity_score"] = (
        (news2 <= 1).astype(int)
        + (gcs == 15).astype(int)
        + (pain.clip(lower=0) <= 3).astype(int)
        + (spo2 >= 96).astype(int)
    )
    features["news2_squared"] = news2 ** 2
    features["age_bin_fine"] = pd.cut(age, bins=[0, 2, 5, 18, 40, 65, 80, 100],
                                       labels=False)

    # Missing vitals count (strong predictor per competitor analysis)
    features["n_vitals_missing"] = (
        df.get("systolic_bp", pd.Series(dtype=float)).isna().astype(int)
        + df.get("diastolic_bp", pd.Series(dtype=float)).isna().astype(int)
        + df.get("respiratory_rate", pd.Series(dtype=float)).isna().astype(int)
        + df.get("temperature_c", pd.Series(dtype=float)).isna().astype(int)
        + df.get("shock_index", pd.Series(dtype=float)).isna().astype(int)
    )
    features["bp_missing"] = df.get("systolic_bp", pd.Series(dtype=float)).isna().astype(int)
    features["pain_not_recorded"] = (pain < 0).astype(int)

    # Clinical interaction features (from competitor analysis)
    immuno = df.get("hx_immunosuppressed", pd.Series(0, index=df.index))
    features["immuno_x_fever"] = (immuno == 1).astype(int) * (temp > 38.0).astype(int)

    cardiac_risk_cols = ["hx_heart_failure", "hx_coronary_artery_disease",
                         "hx_atrial_fibrillation"]
    present_cardiac = [c for c in cardiac_risk_cols if c in df.columns]
    if present_cardiac:
        cardiac = df[present_cardiac].sum(axis=1)
        features["cardiac_x_tachy"] = (cardiac > 0).astype(int) * (hr > 100).astype(int)

    # Mental status severity (ordinal encoding)
    mental = df.get("mental_status_triage", pd.Series("", index=df.index))
    ms_map = {"alert": 0, "agitated": 1, "drowsy": 2, "confused": 3, "unresponsive": 4}
    features["mental_severity"] = mental.map(ms_map).fillna(0).astype(int)
    features["pain_x_ams"] = (pain >= 7).astype(int) * (features["mental_severity"] >= 2).astype(int)

    # Coagulopathy × bleeding/trauma complaint keywords
    coag = df.get("hx_coagulopathy", pd.Series(0, index=df.index))
    features["coag_risk"] = (coag == 1).astype(int)

    return pd.DataFrame(features, index=df.index)


# ---------------------------------------------------------------------------
# Bank signal features (geometry → model input)
# ---------------------------------------------------------------------------

def _build_bank_features(bank_decisions: list) -> pd.DataFrame:
    """Convert bank decomposition signals into model features.

    Each bank contributes:
    - esi_estimate (continuous 1.0-5.0)
    - confidence (0.0-1.0)
    - signed phase deviation (θ_i − ψ) — Kuramoto-style disagreement from consensus.
      Validated by analysis/foundation_report.json: `dev_demographic` alone has
      MI=0.62 with target, higher than any raw bank ESI estimate.

    Also emits scalar order parameter `bank_r` and subset order parameters
    for clinically meaningful bank groupings.
    """
    # Clinically meaningful subsets (validated in foundation analysis)
    SUBSETS = {
        "physiologic_core": {"severity", "consciousness", "respiratory",
                             "cardiovascular"},
        "chronic_profile": {"history", "demographic", "utilization"},
        "complaint_context": {"complaint", "pain", "thermal"},
    }

    rows = []
    for decomp in bank_decisions:
        row = {}

        # Phase-space embedding: ESI [1,5] → θ ∈ [0, π]. Half-circle avoids
        # wrapping ESI 5 back onto ESI 1.
        esi_arr, conf_arr, names = [], [], []
        for bank, signal in decomp.signals.items():
            prefix = f"bank_{bank.value}"
            row[f"{prefix}_esi"] = signal.esi_estimate
            row[f"{prefix}_conf"] = signal.confidence
            if signal.confidence > 0.05:
                theta = (signal.esi_estimate - 1.0) / 4.0 * np.pi
                esi_arr.append(theta)
                conf_arr.append(signal.confidence)
                names.append(bank.value)

        # Kuramoto order parameter across confident banks
        if esi_arr:
            thetas = np.asarray(esi_arr, dtype=float)
            w = np.asarray(conf_arr, dtype=float)
            z = (np.exp(1j * thetas) * w).sum() / w.sum()
            r = float(np.abs(z))
            psi = float(np.angle(z))
            row["bank_r_total"] = r
            row["bank_psi"] = psi
        else:
            r, psi = 0.0, 0.0
            row["bank_r_total"] = 0.0
            row["bank_psi"] = 0.0

        # Per-bank signed phase deviation θ_i − ψ, wrapped to [−π, π].
        # Zero-filled for banks with conf ≤ 0.05 (noise banks don't dissent).
        for bank, signal in decomp.signals.items():
            prefix = f"bank_{bank.value}"
            if signal.confidence > 0.05:
                theta = (signal.esi_estimate - 1.0) / 4.0 * np.pi
                dev = (theta - psi + np.pi) % (2 * np.pi) - np.pi
                row[f"{prefix}_dev"] = dev
            else:
                row[f"{prefix}_dev"] = 0.0

        # Subset order parameters (clinically grouped banks agree or don't)
        for sub_name, members in SUBSETS.items():
            sub_thetas, sub_w = [], []
            for bank, signal in decomp.signals.items():
                if bank.value in members and signal.confidence > 0.05:
                    sub_thetas.append((signal.esi_estimate - 1.0) / 4.0 * np.pi)
                    sub_w.append(signal.confidence)
            if sub_thetas:
                sz = (np.exp(1j * np.asarray(sub_thetas)) *
                      np.asarray(sub_w)).sum() / sum(sub_w)
                row[f"bank_r_{sub_name}"] = float(np.abs(sz))
            else:
                row[f"bank_r_{sub_name}"] = 0.0

        # Disagreement structure: how far is the core from the chronic?
        # If physiologic says severe but demographic says non-severe → interesting
        r_core = row.get("bank_r_physiologic_core", 0.0)
        r_chronic = row.get("bank_r_chronic_profile", 0.0)
        row["bank_coherence_spread"] = abs(r_core - r_chronic)

        row["patient_id"] = decomp.patient_id
        rows.append(row)
    return pd.DataFrame(rows).set_index("patient_id")


# ---------------------------------------------------------------------------
# Full feature pipeline
# ---------------------------------------------------------------------------

LEAKAGE_COLS = ["disposition", "ed_los_hours", "triage_acuity"]
ID_COLS = ["patient_id"]
TEXT_COLS = ["chief_complaint_raw"]


def build_features(df: pd.DataFrame,
                   complaints_df: pd.DataFrame | None = None,
                   history_df: pd.DataFrame | None = None,
                   bank_decompositions: list | None = None,
                   ) -> pd.DataFrame:
    """Build the full feature matrix for model training/inference.

    Merges:
    1. Raw structured features (vitals, demographics, arrival)
    2. Chief complaint text features (keyword counts only; no target encoding)
    3. Clinical interaction features
    4. Bank signal features (data geometry → model input)

    NOTE on declined shortcut: the Kaggle synthetic data has near-deterministic
    chief_complaint → triage_acuity mapping (2,296 unique base conditions,
    target std=0.0 within every common condition). Target encoding on the
    base condition is therefore a label lookup — fold-safe but clinically
    meaningless. A full-stack benchmark with fold-safe target encoding on
    this feature yielded QWK ≈ 0.999 / mF1 ≈ 0.995, entirely driven by the
    encoded complaint string. We decline the shortcut: chief complaint text
    is used only via the deterministic clinical lexicon (complaint_lexicon.py)
    to emit a COMPLAINT bank signal. This preserves a signal that would
    transfer to a real ED dataset, where complaint → acuity is noisy.
    """
    # Start with raw features, drop leakage + target + text cols
    drop = [c for c in LEAKAGE_COLS + TEXT_COLS if c in df.columns]
    result = df.drop(columns=drop, errors="ignore").copy()

    # LEAKAGE GUARD: verify leakage columns were removed
    leakage_remaining = [c for c in ["disposition", "ed_los_hours"]
                         if c in result.columns]
    assert not leakage_remaining, (
        f"Leakage columns survived drop: {leakage_remaining}"
    )

    # Age-group pain imputation: pain_score=-1 means "unable to assess"
    # Impute with age_group median (clinically appropriate — pain expression
    # varies by age, and inability to assess correlates with altered status)
    if "pain_score" in result.columns and "age_group" in result.columns:
        pain_missing_mask = result["pain_score"] < 0
        if pain_missing_mask.any():
            age_medians = result.loc[~pain_missing_mask].groupby(
                "age_group")["pain_score"].median()
            for ag, med in age_medians.items():
                mask = pain_missing_mask & (result["age_group"] == ag)
                result.loc[mask, "pain_score"] = med

    # Merge auxiliary tables
    if complaints_df is not None:
        cc_cols = complaints_df[["patient_id", "chief_complaint_raw"]].copy()
        result = result.merge(cc_cols, on="patient_id", how="left")

    if history_df is not None:
        result = result.merge(history_df, on="patient_id", how="left",
                              suffixes=("", "_hx_dup"))
        dup_cols = [c for c in result.columns if c.endswith("_hx_dup")]
        result = result.drop(columns=dup_cols)

    # Text features
    text_col = "chief_complaint_raw"
    if text_col in result.columns:
        text_feats = _extract_text_features(result[text_col])
        result = pd.concat([result, text_feats], axis=1)
        result = result.drop(columns=[text_col])
    elif complaints_df is not None and text_col in complaints_df.columns:
        merged_text = result.merge(
            complaints_df[["patient_id", text_col]], on="patient_id", how="left"
        )
        text_feats = _extract_text_features(merged_text[text_col])
        result = pd.concat([result, text_feats], axis=1)

    # Clinical interaction features
    clinical_feats = _build_clinical_features(result)
    result = pd.concat([result, clinical_feats], axis=1)

    # Bank signal features
    if bank_decompositions:
        bank_feats = _build_bank_features(bank_decompositions)
        result = result.merge(bank_feats, left_on="patient_id",
                              right_index=True, how="left")

    return result
