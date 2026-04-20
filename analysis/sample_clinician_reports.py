#!/usr/bin/env python3
"""
Generate 10-15 sample ClinicianReports for the writeup.

Picks a diverse mix:
  - 2 hard-rule cases (GCS ≤ 8, cardiac arrest) — high confidence, agreement
  - 2 high-coherence low-confidence cases (banks agree, but r is low)
  - 2 temporal-paradox cases (severe-complaint cohort + low NEWS2)
  - 2 dissent cases (physiologic disagrees with demographic/history)
  - 2 chronic-marker cases
  - 2 routine ESI 3 cases for contrast
  - 2 uncertainty-boundary cases (top-2 model prob gap < 0.15)

Each report is rendered as human-readable text AND dumped as JSON for the
writeup or audit trail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.banks import Bank, decompose_dataframe
from src.clinician_output import build_report, dump_reports, render_report
from src.clinician_style import (
    calibrate, fit_confidence_calibrator, fit_style_banks,
    style_features_for_patients,
)
from src.complaint_lexicon import classify_complaints_batch
from src.temporal_bank import build_temporal_features, fit_cohort_expectations

DATA = PROJECT_ROOT / "data" / "extracted"
OUT_JSON = PROJECT_ROOT / "analysis" / "sample_clinician_reports.json"
OUT_TEXT = PROJECT_ROOT / "analysis" / "sample_clinician_reports.txt"


def _top_conf(decomp) -> float:
    confs = [s.confidence for s in decomp.signals.values()]
    return max(confs) if confs else 0.0


def _is_hard_rule(d) -> bool:
    cons = d.signals.get(Bank.CONSCIOUSNESS)
    return bool(cons and "gcs" in cons.evidence and "comatose" in cons.evidence)


def _has_high_coherence(d, threshold: float = 0.9) -> bool:
    thetas, ws = [], []
    for _, s in d.signals.items():
        if s.confidence > 0.05:
            thetas.append((s.esi_estimate - 1) / 4 * np.pi)
            ws.append(s.confidence)
    if not thetas:
        return False
    z = (np.exp(1j * np.asarray(thetas)) * np.asarray(ws)).sum() / sum(ws)
    return abs(z) > threshold


def _has_demo_vs_severity_dissent(d, gap: float = 2.0) -> bool:
    demo = d.signals.get(Bank.DEMOGRAPHIC)
    sev = d.signals.get(Bank.SEVERITY)
    return bool(demo and sev and abs(demo.esi_estimate - sev.esi_estimate) > gap)


def pick_diverse(decomps, train, temporal_feats,
                 per_category: int = 2) -> dict[str, list[int]]:
    """Return index lists per category for a diverse sampling."""
    esi = train.set_index("patient_id").triage_acuity
    buckets: dict[str, list[int]] = {
        "hard_rule": [], "paradox": [], "chronic": [],
        "routine": [], "dissent": [],
    }

    for i, d in enumerate(decomps):
        tf = temporal_feats.iloc[i]
        if len(buckets["paradox"]) < per_category and tf["temporal_paradox_flag"] == 1:
            buckets["paradox"].append(i); continue
        if len(buckets["chronic"]) < per_category and tf["temporal_chronic"] == 1:
            buckets["chronic"].append(i); continue
        if len(buckets["hard_rule"]) < per_category and _is_hard_rule(d):
            buckets["hard_rule"].append(i); continue
        if len(buckets["routine"]) < per_category and \
                esi.loc[d.patient_id] == 3 and _has_high_coherence(d):
            buckets["routine"].append(i); continue
        if len(buckets["dissent"]) < per_category and _has_demo_vs_severity_dissent(d):
            buckets["dissent"].append(i)

    return buckets


def main():
    print("Loading data...")
    train = pd.read_csv(DATA / "train.csv")
    complaints = pd.read_csv(DATA / "chief_complaints.csv")
    history = pd.read_csv(DATA / "patient_history.csv")

    # Fit the full Tier-B infrastructure
    nurse_banks, pop_dist = fit_style_banks(train, "triage_nurse_id",
                                             smoothing=20.0)
    site_banks, _ = fit_style_banks(train, "site_id", smoothing=20.0)
    cohort = fit_cohort_expectations(train, complaints)

    print("Decomposing banks...")
    merged = train.merge(history, on="patient_id", how="left",
                         suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    cc = classify_complaints_batch(
        complaints[complaints.patient_id.isin(train.patient_id)])
    decomps = decompose_dataframe(merged, cc)

    # Confidence calibrator
    top_confs = np.array([_top_conf(d) for d in decomps])
    cal = fit_confidence_calibrator(train, top_confs, n_bins=10)

    # Temporal features
    temporal = build_temporal_features(train, complaints, cohort)

    # Style features per patient
    style = style_features_for_patients(train, nurse_banks, site_banks, pop_dist)

    # Pick diverse patients
    print("Picking diverse patient sample...")
    picks = pick_diverse(decomps, train, temporal)
    all_idx = []
    for cat, idxs in picks.items():
        print(f"  {cat}: {len(idxs)} patients")
        all_idx.extend(idxs)

    print(f"\nTotal selected: {len(all_idx)}")

    # Build reports
    reports = []
    for i in all_idx:
        d = decomps[i]
        pid = d.patient_id
        true_esi = int(train.iloc[i].triage_acuity)

        # Use true ESI as prediction proxy for this demo (in production this
        # would be the model's prediction; here we illustrate the report
        # structure using ground truth).
        calibrated = float(calibrate(cal, np.array([top_confs[i]]))[0])

        rep = build_report(
            decomp=d,
            esi_prediction=true_esi,
            model_confidence=float(top_confs[i]),
            calibrated_severe_prob=calibrated,
            style_info={
                "nurse_over_bias": float(style.iloc[i]["style_nurse_over_bias"]),
                "nurse_under_bias": float(style.iloc[i]["style_nurse_under_bias"]),
                "nurse_expected_esi": float(style.iloc[i]["style_nurse_expected_esi"]),
                "site_l1_dev": float(style.iloc[i]["style_site_l1_dev"]),
            },
            temporal_features=dict(temporal.iloc[i]),
            method="demo (label shown)",
        )
        reports.append(rep)

    # Dump JSON
    dump_reports(reports, OUT_JSON)
    print(f"\nWrote {OUT_JSON}")

    # Write human-readable text
    lines = ["=" * 60,
             "Sample ClinicianReports — for writeup / audit trail demo",
             "=" * 60, ""]
    for cat, idxs in picks.items():
        lines.append(f"--- {cat.upper()} ({len(idxs)}) ---\n")
        for i in idxs:
            # Find the report corresponding to this patient index
            pid = decomps[i].patient_id
            for r in reports:
                if r.patient_id == pid:
                    lines.append(render_report(r))
                    lines.append("")
                    break

    OUT_TEXT.write_text("\n".join(lines))
    print(f"Wrote {OUT_TEXT}")

    # Print a sample
    print("\n=== PREVIEW: first 3 reports ===\n")
    for r in reports[:3]:
        print(render_report(r))
        print()


if __name__ == "__main__":
    main()
