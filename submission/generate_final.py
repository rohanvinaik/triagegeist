#!/usr/bin/env python3
"""
Final submission generator with full audit trail.

Scale 0: Hard rules (deterministic) → Scale 1: Ensemble (CatBoost+LightGBM)
→ Scale 2: QWK threshold optimization → Scale 3: Claude residual
"""

from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.banks import decompose_dataframe
from src.coherence import triage_patient
from src.complaint_lexicon import classify_complaints_batch
from src.feature_engine import build_features
from src.model import prepare_xy
from src.qwk_optimizer import optimize_thresholds, predict_with_thresholds
from src.temporal_bank import build_temporal_features, fit_cohort_expectations

DATA_DIR = PROJECT_ROOT / "data" / "extracted"
SUBMISSION_DIR = Path(__file__).resolve().parent

CB_PARAMS = dict(
    loss_function="MultiClass", eval_metric="TotalF1",
    iterations=2000, learning_rate=0.05, depth=8,
    l2_leaf_reg=5, min_data_in_leaf=10, random_strength=1.0,
    bootstrap_type="Bernoulli", subsample=0.85, colsample_bylevel=0.8,
    auto_class_weights="Balanced",
)
LGB_PARAMS = dict(
    objective="multiclass", num_class=5, n_estimators=1500,
    learning_rate=0.05, max_depth=8, num_leaves=127,
    min_child_samples=15, subsample=0.85, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=5.0, class_weight="balanced",
    verbose=-1, n_jobs=-1,
)


def _run_scale_0(test, history, complaints, test_decomps):
    """Scale 0: Deterministic hard rules."""
    decisions = [triage_patient(d) for d in test_decomps]
    hard_rules = {}
    audit = []
    for dec in decisions:
        if dec.method == "rules" and dec.confidence >= 0.95:
            hard_rules[dec.patient_id] = dec.esi_prediction
            audit.append({
                "patient_id": dec.patient_id, "esi": dec.esi_prediction,
                "confidence": dec.confidence, "evidence": dec.evidence,
            })
    print(f"  Hard rules resolved: {len(hard_rules)}/{len(test)}")
    return hard_rules, audit


def _train_ensemble(X_train, y_train, X_test, cat_cols):
    """Scale 1: Train CatBoost + LightGBM ensemble, return test probabilities."""
    all_proba = []
    for seed in [42, 123, 777]:
        print(f"  CatBoost (seed={seed})...")
        cb = CatBoostClassifier(**CB_PARAMS, random_seed=seed, verbose=500)
        cb.fit(X_train, y_train, cat_features=cat_cols)
        all_proba.append(cb.predict_proba(X_test))
        del cb; gc.collect()

    X_lgb, X_test_lgb = X_train.copy(), X_test.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

    for seed in [42, 314]:
        print(f"  LightGBM (seed={seed})...")
        lgb = LGBMClassifier(**LGB_PARAMS, random_state=seed)
        lgb.fit(X_lgb, y_train)
        all_proba.append(lgb.predict_proba(X_test_lgb))
        del lgb; gc.collect()

    del X_lgb, X_test_lgb; gc.collect()

    weights = [0.6 / 3] * 3 + [0.4 / 2] * 2
    return sum(w * p for w, p in zip(weights, all_proba))


def _get_qwk_thresholds():
    """Scale 2: Use pre-optimized QWK thresholds from CV benchmark.

    These were optimized on 5-fold OOF probabilities (80K training examples).
    Re-computing them requires training 10 additional models (5 CB + 5 LGB),
    so we cache the result from the benchmark run.
    """
    # Pre-computed from optimize_thresholds() on OOF probabilities
    thresholds = np.array([1.56376089, 2.51698442, 3.55305499, 4.58287457])
    oof_qwk = 0.9702
    print(f"  Using pre-optimized thresholds: {thresholds}")
    print(f"  OOF QWK: {oof_qwk:.4f}")
    return thresholds, oof_qwk


def _apply_claude_residual(test_ids, test_proba, final_preds, pred_methods,
                           hard_rules, threshold=0.20):
    """Scale 3: Apply pre-computed Claude sub-agent decisions."""
    llm_decisions = {}
    for batch_file in sorted(SUBMISSION_DIR.glob("decisions_batch_*.json")):
        with open(batch_file) as f:
            for d in json.load(f):
                llm_decisions[d["patient_id"]] = d["esi"]

    n_changed = 0
    audit = []
    for i, pid in enumerate(test_ids):
        if pid in hard_rules or pid not in llm_decisions:
            continue
        sorted_idx = test_proba[i].argsort()[::-1]
        gap = test_proba[i, sorted_idx[0]] - test_proba[i, sorted_idx[1]]
        if gap < threshold:
            old_pred = final_preds[i]
            new_pred = llm_decisions[pid]
            if old_pred != new_pred:
                n_changed += 1
                audit.append({"patient_id": pid, "model_pred": old_pred,
                              "claude_pred": new_pred, "gap": round(float(gap), 4)})
            final_preds[i] = new_pred
            pred_methods[i] = "claude_residual"

    print(f"  Claude decisions: {len(llm_decisions)}, changed: {n_changed}")
    return audit


def main():
    print("=" * 60)
    print("TriageGeist Final Submission Generator")
    print("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    complaints = pd.read_csv(DATA_DIR / "chief_complaints.csv")
    history = pd.read_csv(DATA_DIR / "patient_history.csv")
    audit = {"scales": {}}

    # Decompose both sets
    print("\n--- Scale 0: Bank Decomposition + Hard Rules ---")
    decomps = {}
    for label, df in [("train", train), ("test", test)]:
        merged = df.merge(history, on="patient_id", how="left", suffixes=("", "_dup"))
        merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
        cc = classify_complaints_batch(
            complaints[complaints["patient_id"].isin(df["patient_id"])])
        decomps[label] = decompose_dataframe(merged, cc)

    hard_rules, rule_audit = _run_scale_0(test, history, complaints, decomps["test"])
    audit["scales"]["scale_0_rules"] = {"count": len(hard_rules), "decisions": rule_audit}

    # Features + model
    print("\n--- Scale 1: CatBoost + LightGBM Ensemble ---")
    train_feats = build_features(train, complaints, history, decomps["train"])
    test_feats = build_features(test, complaints, history, decomps["test"])
    del decomps; gc.collect()

    # Option-2 forensic verdict (memory: project_tier_b_forensic_verdict.md):
    # Drop Kuramoto phase-deviation columns — empirically net-zero to slightly
    # negative in 5-fold regime-split CV (paired t: easy Δ=+0.0000 p=1.00,
    # hard Δ=−0.0025 Cohen's d=−0.68). Phase information still renders into
    # the Scale 3 LLM audit context via ClinicianReport.
    phase_cols = [c for c in train_feats.columns
                  if c.startswith("bank_") and c.endswith("_dev")]
    train_feats = train_feats.drop(columns=phase_cols, errors="ignore")
    test_feats = test_feats.drop(columns=phase_cols, errors="ignore")
    print(f"  Dropped {len(phase_cols)} phase-deviation features from ensemble")

    # Add the single Tier-B feature the forensic ablation kept:
    # temporal_news2_deviation = own_NEWS2 − E[NEWS2 | complaint_base, age_group]
    # Carries 91% of the full_stack Tier-B lift; style banks + calibrator +
    # surprisal basis all empirically inert. Clinical theory: cohort-conditional
    # vital surprise (elderly chest pain + unexpectedly low NEWS2 = silent MI,
    # medication masking). Amplitude amplified on synthetic data; disclosed
    # in the writeup.
    cohort = fit_cohort_expectations(train, complaints)
    tr_temp = build_temporal_features(train, complaints, cohort)
    te_temp = build_temporal_features(test, complaints, cohort)
    train_feats["temporal_news2_deviation"] = tr_temp[
        "temporal_news2_deviation"].values
    test_feats["temporal_news2_deviation"] = te_temp[
        "temporal_news2_deviation"].values
    print("  Added temporal_news2_deviation (1 Tier-B feature)")

    X_train, y_train, cat_cols = prepare_xy(train_feats, train["triage_acuity"])
    X_test, _, _ = prepare_xy(test_feats)
    for c in set(X_train.columns) - set(X_test.columns):
        X_test[c] = 0
    X_test = X_test[X_train.columns]

    test_proba = _train_ensemble(X_train, y_train, X_test, cat_cols)

    # QWK thresholds
    print("\n--- Scale 2: QWK Threshold Optimization ---")
    thresholds, oof_qwk = _get_qwk_thresholds()
    audit["scales"]["scale_2_qwk"] = {"thresholds": thresholds.tolist(), "oof_qwk": oof_qwk}
    model_preds = predict_with_thresholds(test_proba, thresholds)

    # Merge rules + model
    test_ids = test["patient_id"].tolist()
    final_preds = [hard_rules[pid] if pid in hard_rules else int(model_preds[i])
                   for i, pid in enumerate(test_ids)]
    pred_methods = ["rules" if pid in hard_rules else "ensemble+qwk"
                    for pid in test_ids]

    # Claude residual
    print("\n--- Scale 3: Claude Residual ---")
    claude_audit = _apply_claude_residual(test_ids, test_proba, final_preds,
                                          pred_methods, hard_rules)
    audit["scales"]["scale_3_claude"] = {"changes": claude_audit[:50]}

    # Write submission
    submission = pd.DataFrame({"patient_id": test_ids, "triage_acuity": final_preds})
    submission.to_csv(SUBMISSION_DIR / "submission.csv", index=False)

    method_counts = Counter(pred_methods)
    print(f"\nSubmission: {submission.shape}")
    print(f"Distribution:\n{submission['triage_acuity'].value_counts().sort_index()}")
    for method, count in method_counts.most_common():
        print(f"  {method}: {count} ({100 * count / len(test_ids):.1f}%)")

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(sample)
    print("Sanity check passed!")

    audit["stats"] = {"method_breakdown": dict(method_counts),
                      "distribution": submission["triage_acuity"].value_counts().sort_index().to_dict(),
                      "n_features": len(X_train.columns)}
    (SUBMISSION_DIR / "submission_audit.json").write_text(
        json.dumps(audit, indent=2, default=str))
    print(f"Audit trail saved.")


if __name__ == "__main__":
    main()
