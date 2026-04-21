#!/usr/bin/env python3
"""Intra-temporal ablation: which of the 5 temporal_* features carries the lift?

Prior forensic work (benchmark_tier_b_ablation.py) showed that temporal
features own ~100% of the Tier-B lift. This script drops ONE temporal feature
at a time to identify the specific laundering channel.

Variants (each is full_stack minus ONE temporal feature):
  - drop_news2_deviation    : suspect #1 (cohort-mean reconstruction pathway)
  - drop_paradox_flag       : suspect #2 (high-precision rare binary, 60 of 80K)
  - drop_trajectory_code    : regex trajectory categorical
  - drop_has_marker         : binary "any marker found"
  - drop_chronic            : binary "chronic marker present"

Same fold seed (42) + r_total gate (bottom 10%) as the main benchmark for
paired-fold comparison. Log format matches regime_stats.py.

Output: analysis/temporal_ablation.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.banks import decompose_dataframe
from src.complaint_lexicon import classify_complaints_batch
from src.feature_engine import build_features
from src.model import prepare_xy
from src.qwk_optimizer import optimize_thresholds, predict_with_thresholds
from src.tier_b_features import build_tier_b_features, fit_tier_b_artifacts

DATA = PROJECT_ROOT / "data" / "extracted"
OUT = PROJECT_ROOT / "analysis" / "temporal_ablation.json"
HARD_FRACTION = 0.10

# Map ablation_name → column to drop
ABLATIONS: dict[str, str] = {
    "drop_news2_deviation": "temporal_news2_deviation",
    "drop_paradox_flag": "temporal_paradox_flag",
    "drop_trajectory_code": "temporal_trajectory_code",
    "drop_has_marker": "temporal_has_marker",
    "drop_chronic": "temporal_chronic",
}


def compute_r_total(decomps):
    out = {}
    for d in decomps:
        thetas, ws = [], []
        for _, sig in d.signals.items():
            if sig.confidence > 0.05:
                theta = (sig.esi_estimate - 1.0) / 4.0 * np.pi
                thetas.append(theta); ws.append(sig.confidence)
        if thetas:
            t = np.asarray(thetas); w = np.asarray(ws)
            z = (np.exp(1j * t) * w).sum() / w.sum()
            out[d.patient_id] = float(np.abs(z))
        else:
            out[d.patient_id] = 0.0
    return out


def _qwk(y_true, y_pred):
    return (float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
            if len(y_true) else float("nan"))


def _f1(y_true, y_pred):
    return (float(f1_score(y_true, y_pred, average="macro"))
            if len(y_true) else float("nan"))


def run_variant(train, complaints, history, decomps_all, hard_mask,
                 *, ablation_name, drop_col, n_splits=5, seed=42):
    y = train.triage_acuity.values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_proba = np.zeros((len(train), 5))
    oof_preds = np.zeros(len(train), dtype=int)
    pid_to_decomp = {d.patient_id: d for d in decomps_all}

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        print(f"[{ablation_name}] fold {fold}/{n_splits}  preparing features ...")

        tr_df = train.iloc[tr_idx].reset_index(drop=True)
        va_df = train.iloc[va_idx].reset_index(drop=True)
        tr_decomps = [pid_to_decomp[p] for p in tr_df.patient_id]
        va_decomps = [pid_to_decomp[p] for p in va_df.patient_id]

        tr_feats = build_features(
            tr_df.drop(columns=["triage_acuity"], errors="ignore"),
            complaints, history, tr_decomps,
        )
        va_feats = build_features(
            va_df.drop(columns=["triage_acuity"], errors="ignore"),
            complaints, history, va_decomps,
        )

        # Full Tier-B, then drop the one temporal column under test
        artifacts = fit_tier_b_artifacts(tr_df, complaints, history)
        tier_b_tr = build_tier_b_features(tr_df.reset_index(drop=True),
                                          complaints, history, artifacts)
        tier_b_va = build_tier_b_features(va_df.reset_index(drop=True),
                                          complaints, history, artifacts)
        if drop_col in tier_b_tr.columns:
            tier_b_tr = tier_b_tr.drop(columns=[drop_col])
            tier_b_va = tier_b_va.drop(columns=[drop_col])
        tier_b_tr.index = tr_feats.index
        tier_b_va.index = va_feats.index
        tr_feats = pd.concat([tr_feats, tier_b_tr], axis=1)
        va_feats = pd.concat([va_feats, tier_b_va], axis=1)

        X_tr, _, cat_cols = prepare_xy(tr_feats)
        X_va, _, _ = prepare_xy(va_feats)
        missing_in_va = set(X_tr.columns) - set(X_va.columns)
        for c in missing_in_va:
            X_va[c] = 0
        X_va = X_va[X_tr.columns]

        m = CatBoostClassifier(
            loss_function="MultiClass", eval_metric="TotalF1",
            iterations=2000, learning_rate=0.05, depth=8,
            l2_leaf_reg=5, min_data_in_leaf=10, random_strength=1.0,
            bootstrap_type="Bernoulli", subsample=0.85,
            colsample_bylevel=0.8,
            random_seed=seed + fold, verbose=0,
            auto_class_weights="Balanced",
        )
        m.fit(X_tr, y[tr_idx], cat_features=cat_cols,
              eval_set=(X_va, y[va_idx]), use_best_model=True,
              early_stopping_rounds=100)
        proba = m.predict_proba(X_va)
        oof_proba[va_idx, :proba.shape[1]] = proba
        oof_preds[va_idx] = m.predict(X_va).reshape(-1).astype(int)

        fold_easy = va_idx[~hard_mask[va_idx]]
        fold_hard = va_idx[hard_mask[va_idx]]
        print(f"[{ablation_name}] fold {fold}  "
              f"all: mF1={_f1(y[va_idx], oof_preds[va_idx]):.4f} "
              f"qwk={_qwk(y[va_idx], oof_preds[va_idx]):.4f}  |  "
              f"easy (n={len(fold_easy)}) qwk={_qwk(y[fold_easy], oof_preds[fold_easy]):.4f}  |  "
              f"hard (n={len(fold_hard)}) qwk={_qwk(y[fold_hard], oof_preds[fold_hard]):.4f}  "
              f"n_feat={X_tr.shape[1]}  elapsed={time.time()-t0:.0f}s")

    easy_mask = ~hard_mask
    res = {
        "label": ablation_name,
        "dropped_col": drop_col,
        "n_easy": int(easy_mask.sum()),
        "n_hard": int(hard_mask.sum()),
        "argmax_macro_f1_all": _f1(y, oof_preds),
        "argmax_qwk_all": _qwk(y, oof_preds),
        "argmax_accuracy_all": float(accuracy_score(y, oof_preds)),
        "argmax_qwk_easy": _qwk(y[easy_mask], oof_preds[easy_mask]),
        "argmax_qwk_hard": _qwk(y[hard_mask], oof_preds[hard_mask]),
    }
    thr, best_qwk = optimize_thresholds(oof_proba, y)
    tuned = predict_with_thresholds(oof_proba, thr)
    res.update({
        "tuned_qwk_all": float(best_qwk),
        "tuned_qwk_easy": _qwk(y[easy_mask], tuned[easy_mask]),
        "tuned_qwk_hard": _qwk(y[hard_mask], tuned[hard_mask]),
        "elapsed_sec": round(time.time() - t0, 1),
    })
    print(f"[{ablation_name}] DONE:")
    print(f"  all:  argmax qwk={res['argmax_qwk_all']:.4f} | tuned qwk={res['tuned_qwk_all']:.4f}")
    print(f"  easy: argmax qwk={res['argmax_qwk_easy']:.4f}")
    print(f"  hard: argmax qwk={res['argmax_qwk_hard']:.4f}")
    return res


def main():
    print("Loading data...")
    train = pd.read_csv(DATA / "train.csv")
    complaints = pd.read_csv(DATA / "chief_complaints.csv")
    history = pd.read_csv(DATA / "patient_history.csv")

    print("Decomposing banks once (12s)...")
    merged = train.merge(history, on="patient_id", how="left",
                         suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    cc = classify_complaints_batch(
        complaints[complaints.patient_id.isin(train.patient_id)])
    decomps = decompose_dataframe(merged, cc)

    print("Computing r_total gate...")
    r_map = compute_r_total(decomps)
    r_total = np.array([r_map[pid] for pid in train.patient_id.values])
    r_threshold = float(np.quantile(r_total, HARD_FRACTION))
    hard_mask = r_total <= r_threshold
    print(f"  hard rows: {int(hard_mask.sum())}  easy rows: {int((~hard_mask).sum())}")

    results = {}
    for name, drop_col in ABLATIONS.items():
        print(f"\n{'=' * 60}")
        print(f"{name}  (dropping {drop_col})")
        print("=" * 60)
        results[name] = run_variant(
            train, complaints, history, decomps, hard_mask,
            ablation_name=name, drop_col=drop_col,
        )

    output = {
        "hard_fraction": HARD_FRACTION,
        "hard_threshold_r_total": r_threshold,
        "n_total": int(len(train)),
        "results": results,
        "notes": (
            "Intra-temporal ablation: drops one temporal_* column at a time. "
            "Compare to full_stack (all temporal in, QWK ~0.9809) and baseline "
            "(no temporal, QWK ~0.9641). The variant whose removal most "
            "collapses the lift identifies the specific laundering channel."
        ),
    }
    OUT.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote {OUT}")

    print("\n=== Summary ===")
    for name, r in results.items():
        print(f"  {name:30s}  easy={r['argmax_qwk_easy']:.4f}  "
              f"hard={r['argmax_qwk_hard']:.4f}  all={r['argmax_qwk_all']:.4f}")


if __name__ == "__main__":
    main()
