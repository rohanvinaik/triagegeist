#!/usr/bin/env python3
"""
Cheat-sheet probe — do discarded signals earn their keep on the HARD residual?

Full-population MI killed three proposed compositional banks (AgeNormalizedShock,
ChronicBurdenResidual, ThermoImmuneProduct) and showed weak direct ESI signal
from temporal trajectory markers. That's the wrong question for an
LLM-enrichment context.

The right question: conditional on a patient already being in the hard
residual (low ensemble confidence, low bank coherence), do these signals
carry information the easy banks already consumed?

This script computes MI of dropped/weak signals on three subsets:
  A. Full training population (80K).
  B. Bottom quartile by top-bank confidence (~20K).
  C. Bottom decile by top-bank confidence (~8K) — this approximates the
     LLM-triggered set.

Signals tested:
  - Compositional banks: age_normalized_shock, chronic_burden_residual,
    thermo_immune_product
  - Temporal: trajectory_code, has_marker, chronic, news2_deviation,
    paradox_flag
  - Clinician style: nurse_over_bias, nurse_under_bias, nurse_expected_esi,
    site_l1_dev
  - Phase deviations (already validated globally): for completeness

Output: analysis/cheat_sheet_probe.json plus a printed delta table ranking
signals by conditional-MI-lift on the hard residual.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.banks import Bank, decompose_dataframe
from src.clinician_style import fit_style_banks, style_features_for_patients
from src.complaint_lexicon import classify_complaints_batch
from src.temporal_bank import build_temporal_features, fit_cohort_expectations

DATA = PROJECT_ROOT / "data" / "extracted"
OUT = PROJECT_ROOT / "analysis" / "cheat_sheet_probe.json"


def _max_bank_conf(decomps) -> np.ndarray:
    out = np.zeros(len(decomps))
    for i, d in enumerate(decomps):
        confs = [s.confidence for s in d.signals.values()]
        out[i] = max(confs) if confs else 0.0
    return out


def _phase_deviations(decomps) -> pd.DataFrame:
    """Per-bank signed phase deviation from consensus (full-pop validated)."""
    rows = []
    for d in decomps:
        thetas, ws = [], []
        for bank, s in d.signals.items():
            if s.confidence > 0.05:
                thetas.append((s.esi_estimate - 1) / 4 * np.pi)
                ws.append(s.confidence)
        if not thetas:
            rows.append({b.value: 0.0 for b in Bank})
            continue
        t = np.asarray(thetas); w = np.asarray(ws)
        z = (np.exp(1j * t) * w).sum() / w.sum()
        psi = np.angle(z)
        row = {}
        for bank, s in d.signals.items():
            theta = (s.esi_estimate - 1) / 4 * np.pi
            dev = (theta - psi + np.pi) % (2 * np.pi) - np.pi
            row[bank.value] = float(dev) if s.confidence > 0.05 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).add_prefix("dev_")


def _compositional_banks(m: pd.DataFrame) -> pd.DataFrame:
    """The three compositional banks proposed+killed on full population."""
    age = m["age"].astype(float).clip(lower=1)
    gcs = m["gcs_total"].fillna(15).astype(float).clip(lower=3, upper=15)
    si = m["shock_index"].astype(float)
    news2 = m["news2_score"].astype(float)
    temp = m["temperature_c"].astype(float)
    meds = m["num_active_medications"].fillna(0).astype(float)
    n_co = m["num_comorbidities"].fillna(0).astype(float).clip(lower=0)
    immuno = (m.get("hx_immunosuppressed", pd.Series(0, index=m.index))
              .fillna(0).astype(int))

    ans = si * (age / 40.0) ** 0.5 * (1.0 + (15 - gcs) / 15.0)
    cond_mean = m.groupby(
        ["age_group",
         pd.cut(n_co, bins=[-0.5, 0.5, 2.5, 4.5, 10.5],
                labels=["0", "1-2", "3-4", "5+"])],
        observed=True,
    )["news2_score"].transform("mean")
    cbr = news2 - cond_mean
    poly = np.maximum(0, np.log1p(np.clip(meds - 3, a_min=0, a_max=None)))
    tip = (temp - 37.0) ** 2 * (1.0 + 2.0 * immuno) * poly

    return pd.DataFrame({
        "age_normalized_shock": ans,
        "chronic_burden_residual": cbr,
        "thermo_immune_product": tip,
    }, index=m.index)


def _mi_on_subset(X: pd.DataFrame, y: np.ndarray, mask: np.ndarray,
                  sample: int = 20000, seed: int = 42) -> pd.Series:
    Xs = X[mask].fillna(X.median())
    ys = y[mask]
    if len(Xs) > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xs), size=sample, replace=False)
        Xs = Xs.iloc[idx]; ys = ys[idx]
    mi = mutual_info_classif(Xs.values, ys, n_neighbors=3, random_state=seed)
    return pd.Series(mi, index=X.columns, name="MI")


def main():
    print("Loading data...")
    train = pd.read_csv(DATA / "train.csv")
    complaints = pd.read_csv(DATA / "chief_complaints.csv")
    history = pd.read_csv(DATA / "patient_history.csv")
    y = train.triage_acuity.values

    print("Decomposing banks on train (12s)...")
    merged = train.merge(history, on="patient_id", how="left",
                         suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    cc = classify_complaints_batch(
        complaints[complaints.patient_id.isin(train.patient_id)])
    decomps = decompose_dataframe(merged, cc)
    train_aligned = train.set_index("patient_id").loc[[d.patient_id for d in decomps]].reset_index()
    y = train_aligned.triage_acuity.values

    print("Assembling signal matrices...")
    comp = _compositional_banks(train_aligned.merge(history, on="patient_id",
                                                     how="left", suffixes=("", "_d")))
    temp_feats = build_temporal_features(
        train_aligned, complaints,
        fit_cohort_expectations(train, complaints))
    nurse, pop = fit_style_banks(train, "triage_nurse_id", smoothing=20.0)
    site, _ = fit_style_banks(train, "site_id", smoothing=20.0)
    style = style_features_for_patients(train_aligned, nurse, site, pop)
    phase = _phase_deviations(decomps)

    signal_matrix = pd.concat(
        [comp,
         temp_feats[["temporal_trajectory_code", "temporal_has_marker",
                     "temporal_chronic", "temporal_news2_deviation",
                     "temporal_paradox_flag"]],
         style[["style_nurse_over_bias", "style_nurse_under_bias",
                "style_nurse_expected_esi", "style_site_l1_dev"]],
         phase],
        axis=1,
    )
    print(f"Signal matrix: {signal_matrix.shape}")

    # Difficulty axis: max-bank confidence (lower = harder for symbolic system)
    top_conf = _max_bank_conf(decomps)
    p25 = np.quantile(top_conf, 0.25)
    p10 = np.quantile(top_conf, 0.10)
    print(f"top_conf quartile=0.25 → {p25:.3f}  decile=0.10 → {p10:.3f}")

    subsets = {
        "full_population": np.ones(len(top_conf), dtype=bool),
        "bottom_quartile_conf": top_conf <= p25,
        "bottom_decile_conf": top_conf <= p10,
    }
    for name, mask in subsets.items():
        print(f"  {name}: n={mask.sum()}")

    print("\nComputing MI on each subset (3 × ~30s each)...")
    mi_results = {}
    for name, mask in subsets.items():
        mi_results[name] = _mi_on_subset(signal_matrix, y, mask)

    # Compute lifts (conditional MI − full-pop MI)
    lift_q25 = (mi_results["bottom_quartile_conf"]
                - mi_results["full_population"])
    lift_p10 = (mi_results["bottom_decile_conf"]
                - mi_results["full_population"])

    table = pd.DataFrame({
        "full_pop_MI": mi_results["full_population"].round(4),
        "q25_MI": mi_results["bottom_quartile_conf"].round(4),
        "p10_MI": mi_results["bottom_decile_conf"].round(4),
        "lift_q25": lift_q25.round(4),
        "lift_p10": lift_p10.round(4),
    })
    table_sorted = table.sort_values("lift_p10", ascending=False)

    print("\n=== MI BY SUBSET, SORTED BY LIFT ON BOTTOM DECILE ===")
    print(table_sorted.to_string())

    print("\n=== SIGNALS WITH LARGEST CONDITIONAL-MI LIFT (hard residual) ===")
    winners = table_sorted[table_sorted.lift_p10 > 0.02]
    print(winners.to_string())

    # Save
    out = {
        "subsets": {name: int(m.sum()) for name, m in subsets.items()},
        "threshold_top_conf_q25": float(p25),
        "threshold_top_conf_p10": float(p10),
        "mi_table": table.round(4).to_dict(),
        "ranked_by_p10_lift": table_sorted.round(4).to_dict(orient="index"),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
