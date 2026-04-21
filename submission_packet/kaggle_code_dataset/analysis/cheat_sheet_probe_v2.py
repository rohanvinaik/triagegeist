#!/usr/bin/env python3
"""
Cheat-sheet probe v2 — test multiple definitions of "hard residual".

v1 used low max-bank-confidence, which surfaces patients where NO bank fires
strongly (typically stable ESI 4-5 patients). That's the wrong kind of hard.
The actual LLM-trigger set is patients where banks DISAGREE or the MODEL
CAN'T DECIDE.

Four hard-residual definitions tested:
  A. Low max-bank-confidence (v1) — "banks say nothing".
  B. Low Kuramoto r (banks disagree) — "banks fight each other".
  C. Large consensus-vs-reality gap — bank-weighted ESI differs from true ESI
     by ≥ 1. The LLM would actually have to disagree with bank consensus here.
  D. High phase-deviation entropy — banks not merely disagreeing but
     disagreeing in complex multi-directional ways.

Output: table of full-pop MI vs conditional MI on each subset. Signals that
lift meaningfully on B/C/D earn LLM-context inclusion for hard cases.
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
OUT = PROJECT_ROOT / "analysis" / "cheat_sheet_probe_v2.json"


def _bank_level_stats(decomps):
    """Compute top_conf, Kuramoto r, consensus ESI, phase-entropy per patient."""
    top_conf = np.zeros(len(decomps))
    r_vec = np.zeros(len(decomps))
    consensus_esi = np.zeros(len(decomps))
    phase_entropy = np.zeros(len(decomps))

    for i, d in enumerate(decomps):
        confs, thetas, esis = [], [], []
        for _, s in d.signals.items():
            if s.confidence > 0.05:
                confs.append(s.confidence)
                thetas.append((s.esi_estimate - 1) / 4 * np.pi)
                esis.append(s.esi_estimate)
        all_confs = [s.confidence for s in d.signals.values()]
        top_conf[i] = max(all_confs) if all_confs else 0.0
        if not thetas:
            continue
        t = np.asarray(thetas); w = np.asarray(confs); e = np.asarray(esis)
        z = (np.exp(1j * t) * w).sum() / w.sum()
        r_vec[i] = abs(z)
        consensus_esi[i] = (w * e).sum() / w.sum()
        # Entropy of phase distribution, quantized into 5 angular bins
        # (harder metric than r — captures multi-modal disagreement)
        psi = np.angle(z)
        dev = (t - psi + np.pi) % (2 * np.pi) - np.pi
        bins = np.digitize(dev, np.linspace(-np.pi, np.pi, 6))
        _, counts = np.unique(bins, return_counts=True)
        p = counts / counts.sum()
        phase_entropy[i] = -np.sum(p * np.log(p + 1e-12))

    return top_conf, r_vec, consensus_esi, phase_entropy


def _compositional(m: pd.DataFrame) -> pd.DataFrame:
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


def _phase_deviations(decomps) -> pd.DataFrame:
    rows = []
    for d in decomps:
        thetas, ws = [], []
        for _, s in d.signals.items():
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


def _mi_subset(X: pd.DataFrame, y: np.ndarray, mask: np.ndarray,
               sample: int = 20000, seed: int = 42) -> pd.Series:
    if mask.sum() == 0:
        return pd.Series(0.0, index=X.columns)
    Xs = X[mask].fillna(X.median())
    ys = y[mask]
    if len(Xs) > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(Xs), size=sample, replace=False)
        Xs = Xs.iloc[idx]; ys = ys[idx]
    mi = mutual_info_classif(Xs.values, ys, n_neighbors=3, random_state=seed)
    return pd.Series(mi, index=X.columns, name="MI")


def main():
    train = pd.read_csv(DATA / "train.csv")
    complaints = pd.read_csv(DATA / "chief_complaints.csv")
    history = pd.read_csv(DATA / "patient_history.csv")

    print("Decomposing banks...")
    merged = train.merge(history, on="patient_id", how="left",
                         suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    cc = classify_complaints_batch(
        complaints[complaints.patient_id.isin(train.patient_id)])
    decomps = decompose_dataframe(merged, cc)
    aligned = train.set_index("patient_id").loc[[d.patient_id for d in decomps]].reset_index()
    y = aligned.triage_acuity.values

    print("Computing bank-level stats (top_conf, r, consensus, entropy)...")
    top_conf, r_vec, consensus_esi, phase_ent = _bank_level_stats(decomps)
    # Consensus-vs-reality gap (absolute)
    gap = np.abs(consensus_esi - y)

    print("Building signal matrices...")
    joined = aligned.merge(history, on="patient_id", how="left", suffixes=("", "_d"))
    comp = _compositional(joined)
    temp_feats = build_temporal_features(
        aligned, complaints, fit_cohort_expectations(train, complaints))
    nurse, pop = fit_style_banks(train, "triage_nurse_id", smoothing=20.0)
    site, _ = fit_style_banks(train, "site_id", smoothing=20.0)
    style = style_features_for_patients(aligned, nurse, site, pop)
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
    print(f"Signal matrix shape: {signal_matrix.shape}")

    # Four hard-residual definitions
    q = lambda x, p: float(np.quantile(x, p))
    subsets = {
        "A_full_pop": np.ones(len(y), dtype=bool),
        # A. Low bank confidence — banks say nothing (v1 definition)
        "B_low_topconf_p25": top_conf <= q(top_conf, 0.25),
        # B. Low Kuramoto r — banks disagree
        "C_low_r_p25": r_vec <= q(r_vec, 0.25),
        "C_low_r_p10": r_vec <= q(r_vec, 0.10),
        # C. Consensus far from reality — bank system is wrong
        "D_gap_ge_1": gap >= 1.0,     # bank consensus off by ≥ 1 ESI
        "D_gap_ge_2": gap >= 2.0,     # off by ≥ 2 ESI (catastrophic)
        # D. High phase entropy — complex multi-modal disagreement
        "E_high_entropy_p75": phase_ent >= q(phase_ent, 0.75),
    }
    print("\nSubset sizes:")
    for name, mask in subsets.items():
        print(f"  {name}: n={mask.sum()}")

    print("\nComputing MI per subset...")
    mi = {}
    for name, mask in subsets.items():
        mi[name] = _mi_subset(signal_matrix, y, mask)

    full = mi["A_full_pop"]
    report = pd.DataFrame({"full_pop": full.round(4)})
    for name in list(subsets)[1:]:
        report[name] = mi[name].round(4)
        report[f"lift_{name}"] = (mi[name] - full).round(4)

    print("\n" + "=" * 100)
    print("MI per subset (lift = subset_MI - full_pop_MI; positive = useful for cheat-sheet)")
    print("=" * 100)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(report.sort_values("lift_D_gap_ge_1", ascending=False).to_string())

    # For each signal, find its BEST subset lift
    lift_cols = [c for c in report.columns if c.startswith("lift_")]
    report["best_lift"] = report[lift_cols].max(axis=1)
    report["best_subset"] = report[lift_cols].idxmax(axis=1).str.replace("lift_", "")
    winners = report[report.best_lift > 0.03].sort_values("best_lift", ascending=False)
    print("\n=== CHEAT-SHEET CANDIDATES (best subset lift > 0.03) ===")
    print(winners[["full_pop", "best_subset", "best_lift"]].to_string())

    out = {
        "subsets": {name: int(m.sum()) for name, m in subsets.items()},
        "mi_table": report.round(4).to_dict(),
        "winners": winners.round(4).to_dict(orient="index"),
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
