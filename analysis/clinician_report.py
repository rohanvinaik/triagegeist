#!/usr/bin/env python3
"""
Writeup-ready report on clinician variance + confidence calibration.

Fits the Tier-B clinician style banks and confidence calibrator on the full
80K training set, then generates a human-readable report:
  - Distribution of per-nurse style divergence from population
  - Top 5 over-triaging nurses, top 5 under-triaging nurses
  - Site-level variance
  - Confidence calibration table (coherence quartile → severe-outcome rate)
  - Suspected-undertriage count and class breakdown

Also dumps JSON artifact `analysis/clinician_report.json` for the writeup.

Everything here uses disposition and ed_los_hours ONLY at training time for
CALIBRATION, never as features. The features exposed by clinician_style.py
depend only on patient_id → triage_nurse_id → pre-fit style vector.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.banks import decompose_dataframe
from src.clinician_style import (
    ESI_LEVELS, SEVERE_DISPOSITIONS,
    detect_undertriage, fit_confidence_calibrator, fit_style_banks,
)
from src.complaint_lexicon import classify_complaints_batch

DATA = PROJECT_ROOT / "data" / "extracted"
OUT = PROJECT_ROOT / "analysis" / "clinician_report.json"


def _max_bank_confidence(decomps: list) -> np.ndarray:
    """Per-patient max bank confidence — the "strongest signal fired" measure.

    Foundation analysis (2026-04-17) validated this as the confidence proxy
    that correlates monotonically with outcome severity (severe-outcome rate
    20% → 84% across quartiles). Kuramoto order parameter r is a different
    quantity — "did banks agree" — and does NOT correlate with severity.
    """
    out = np.zeros(len(decomps))
    for i, d in enumerate(decomps):
        confs = [sig.confidence for sig in d.signals.values()]
        out[i] = float(max(confs)) if confs else 0.0
    return out


def main():
    print("Loading data...")
    train = pd.read_csv(DATA / "train.csv")
    complaints = pd.read_csv(DATA / "chief_complaints.csv")
    history = pd.read_csv(DATA / "patient_history.csv")

    print(f"Training rows: {len(train)}")

    # Fit style banks (on FULL train — this is for the writeup, not model training)
    nurse_banks, pop_dist = fit_style_banks(train, "triage_nurse_id",
                                             smoothing=20.0)
    site_banks, _ = fit_style_banks(train, "site_id", smoothing=20.0)

    print(f"\nPopulation ESI distribution: "
          f"{dict(zip(ESI_LEVELS, pop_dist.round(3)))}")

    # Rank nurses by over-triage and under-triage bias
    nurse_df = pd.DataFrame([
        {
            "nurse_id": b.rater_id,
            "n_seen": b.n_seen,
            "l1_dev": b.l1_dev_from_pop,
            "over_triage_bias": b.over_triage_bias,
            "under_triage_bias": b.under_triage_bias,
            **{f"p_esi_{l}": b.distribution[i]
               for i, l in enumerate(ESI_LEVELS)},
        }
        for b in nurse_banks.values()
    ]).sort_values("l1_dev", ascending=False)

    site_df = pd.DataFrame([
        {
            "site_id": b.rater_id,
            "n_seen": b.n_seen,
            "l1_dev": b.l1_dev_from_pop,
            "over_triage_bias": b.over_triage_bias,
            "under_triage_bias": b.under_triage_bias,
            **{f"p_esi_{l}": b.distribution[i]
               for i, l in enumerate(ESI_LEVELS)},
        }
        for b in site_banks.values()
    ]).sort_values("l1_dev", ascending=False)

    print(f"\n=== NURSE VARIANCE (n={len(nurse_df)}) ===")
    print(f"L1 deviation from population: mean={nurse_df.l1_dev.mean():.3f} "
          f"median={nurse_df.l1_dev.median():.3f} "
          f"max={nurse_df.l1_dev.max():.3f}")

    print("\nTop 5 OVER-triaging nurses (higher ESI 1-2 rate than population):")
    top_over = nurse_df.nlargest(5, "over_triage_bias")[
        ["nurse_id", "n_seen", "over_triage_bias", "p_esi_1",
         "p_esi_2", "p_esi_5"]]
    print(top_over.to_string(index=False))

    print("\nTop 5 UNDER-triaging nurses (higher ESI 4-5 rate than population):")
    top_under = nurse_df.nlargest(5, "under_triage_bias")[
        ["nurse_id", "n_seen", "under_triage_bias", "p_esi_1",
         "p_esi_4", "p_esi_5"]]
    print(top_under.to_string(index=False))

    print(f"\n=== SITE VARIANCE (n={len(site_df)}) ===")
    print(site_df.to_string(index=False))

    # Bank decomposition for confidence calibration
    print(f"\nDecomposing banks on train for confidence calibration "
          f"(this takes ~12s)...")
    merged = train.merge(history, on="patient_id", how="left",
                         suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    cc = classify_complaints_batch(
        complaints[complaints.patient_id.isin(train.patient_id)])
    decomps = decompose_dataframe(merged, cc)
    conf = _max_bank_confidence(decomps)

    cal = fit_confidence_calibrator(train, conf, n_bins=10)
    print("\n=== CONFIDENCE CALIBRATION (decile bins) ===")
    for i in range(len(cal.severe_rate)):
        lo = cal.bin_edges[i] if np.isfinite(cal.bin_edges[i]) else -1
        hi = cal.bin_edges[i+1] if np.isfinite(cal.bin_edges[i+1]) else 99
        print(f"  bin{i:2d} [{lo:.3f}–{hi:.3f}]  n={cal.n_per_bin[i]:5d}  "
              f"severe={cal.severe_rate[i]:.3f}  mean_los={cal.mean_los[i]:.2f}h")

    # Training-set undertriage suspects
    suspect = detect_undertriage(train, los_threshold=6.0)
    print(f"\n=== LABEL-NOISE DETECTION ===")
    print(f"Suspected undertriage in training: {suspect.sum()} / {len(train)} "
          f"({100*suspect.mean():.2f}%)")
    print(f"Breakdown by current label:")
    print(train[suspect]["triage_acuity"].value_counts().sort_index().to_string())
    print(f"Breakdown by disposition:")
    print(train[suspect]["disposition"].value_counts().to_string())

    # Save JSON artifact for writeup
    report = {
        "meta": {
            "n_train": int(len(train)),
            "n_nurses": int(len(nurse_df)),
            "n_sites": int(len(site_df)),
            "severe_dispositions": sorted(SEVERE_DISPOSITIONS),
            "population_esi_distribution": {
                str(l): float(p) for l, p in zip(ESI_LEVELS, pop_dist)},
        },
        "nurse_variance": {
            "l1_dev_stats": nurse_df.l1_dev.describe().to_dict(),
            "top_over_triage": top_over.to_dict(orient="records"),
            "top_under_triage": top_under.to_dict(orient="records"),
        },
        "site_variance": {
            "sites": site_df.to_dict(orient="records"),
        },
        "confidence_calibration_bins": [
            {
                "bin": int(i),
                "lo": float(cal.bin_edges[i]) if np.isfinite(cal.bin_edges[i]) else None,
                "hi": float(cal.bin_edges[i+1]) if np.isfinite(cal.bin_edges[i+1]) else None,
                "n": int(cal.n_per_bin[i]),
                "severe_rate": float(cal.severe_rate[i]),
                "mean_los_hours": float(cal.mean_los[i]) if not np.isnan(cal.mean_los[i]) else None,
            }
            for i in range(len(cal.severe_rate))
        ],
        "label_noise": {
            "suspected_undertriage_count": int(suspect.sum()),
            "suspected_undertriage_pct": float(100 * suspect.mean()),
            "by_label": train[suspect]["triage_acuity"].value_counts().sort_index().to_dict(),
            "by_disposition": train[suspect]["disposition"].value_counts().to_dict(),
            "criteria": "label∈{4,5} AND disposition∈severe AND ed_los_hours≥6h",
        },
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
