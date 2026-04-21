#!/usr/bin/env python3
"""
Measure CV lift from adding Kuramoto phase-deviation features.

Compares:
  A. Baseline — bank ESI + bank confidence features (status quo).
  B. + phase — adds per-bank signed phase deviation, subset r values, and
     scalar total r.

Both trained with identical 5-fold StratifiedKFold, same CatBoost params, same
seed. Reports macro F1, accuracy, QWK (argmax and QWK-optimized), linear kappa.

Writes `analysis/phase_feature_benchmark.json`.
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

DATA = PROJECT_ROOT / "data" / "extracted"
OUT = PROJECT_ROOT / "analysis" / "phase_feature_benchmark.json"


def strip_phase_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return feature matrix with new phase-related cols dropped (baseline A)."""
    drop = [c for c in X.columns if
            c.endswith("_dev") or c.startswith("bank_r_") or
            c == "bank_psi" or c == "bank_coherence_spread"]
    return X.drop(columns=drop)


def run_cv(X: pd.DataFrame, y: np.ndarray, cat_cols: list[str],
           label: str, seed: int = 42) -> dict:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_proba = np.zeros((len(X), 5))
    oof_preds = np.zeros(len(X), dtype=int)

    t0 = time.time()
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        m = CatBoostClassifier(
            loss_function="MultiClass", eval_metric="TotalF1",
            iterations=2000, learning_rate=0.05, depth=8,
            l2_leaf_reg=5, min_data_in_leaf=10, random_strength=1.0,
            bootstrap_type="Bernoulli", subsample=0.85, colsample_bylevel=0.8,
            random_seed=seed + fold, verbose=0,
            auto_class_weights="Balanced",
        )
        m.fit(X.iloc[tr], y[tr], cat_features=cat_cols,
              eval_set=(X.iloc[va], y[va]), use_best_model=True,
              early_stopping_rounds=100)
        proba = m.predict_proba(X.iloc[va])
        oof_proba[va, :proba.shape[1]] = proba
        oof_preds[va] = m.predict(X.iloc[va]).reshape(-1).astype(int)
        f1 = f1_score(y[va], oof_preds[va], average="macro")
        print(f"[{label}] fold {fold}  mF1={f1:.4f}  "
              f"elapsed={time.time()-t0:.0f}s")

    res = {
        "label": label,
        "n_features": int(X.shape[1]),
        "argmax_macro_f1": float(f1_score(y, oof_preds, average="macro")),
        "argmax_accuracy": float(accuracy_score(y, oof_preds)),
        "argmax_qwk": float(cohen_kappa_score(y, oof_preds, weights="quadratic")),
        "argmax_linear_kappa": float(cohen_kappa_score(y, oof_preds, weights="linear")),
    }

    thr, best_qwk = optimize_thresholds(oof_proba, y)
    tuned = predict_with_thresholds(oof_proba, thr)
    res.update({
        "tuned_macro_f1": float(f1_score(y, tuned, average="macro")),
        "tuned_accuracy": float(accuracy_score(y, tuned)),
        "tuned_qwk": float(best_qwk),
        "tuned_linear_kappa": float(cohen_kappa_score(y, tuned, weights="linear")),
        "thresholds": thr.tolist(),
        "elapsed_sec": round(time.time() - t0, 1),
    })
    print(f"[{label}] DONE: argmax mF1={res['argmax_macro_f1']:.4f} "
          f"qwk={res['argmax_qwk']:.4f} | tuned qwk={res['tuned_qwk']:.4f}")
    return res


def main():
    print("Loading data...")
    train = pd.read_csv(DATA / "train.csv")
    complaints = pd.read_csv(DATA / "chief_complaints.csv")
    history = pd.read_csv(DATA / "patient_history.csv")

    print("Building bank decomposition + features (WITH phase)...")
    merged = train.merge(history, on="patient_id", how="left",
                         suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    cc = classify_complaints_batch(
        complaints[complaints.patient_id.isin(train.patient_id)])
    decomps = decompose_dataframe(merged, cc)
    feats_full = build_features(train, complaints, history, decomps)

    X_full, y_arr, cat_cols = prepare_xy(feats_full, train["triage_acuity"])
    X_base = strip_phase_features(X_full)

    print(f"\nFeature counts: base={X_base.shape[1]} full={X_full.shape[1]} "
          f"new={X_full.shape[1] - X_base.shape[1]}")
    new_cols = sorted(set(X_full.columns) - set(X_base.columns))
    print(f"New (phase) features:")
    for c in new_cols:
        print(f"  {c}")

    print("\n" + "=" * 60)
    print("BASELINE A (bank_esi + bank_conf, no phase features)")
    print("=" * 60)
    A = run_cv(X_base, y_arr, cat_cols, "baseline")

    print("\n" + "=" * 60)
    print("VARIANT B (+ phase deviations + subset_r + total_r)")
    print("=" * 60)
    B = run_cv(X_full, y_arr, cat_cols, "with_phase")

    # Deltas
    deltas = {k: B[k] - A[k] for k in A
              if isinstance(A[k], float) and isinstance(B[k], float)
              and k not in ("elapsed_sec",)}

    result = {
        "baseline": A,
        "with_phase": B,
        "delta_B_minus_A": deltas,
        "new_phase_features": new_cols,
    }
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {OUT}")
    print(f"\n--- LIFT (with_phase - baseline) ---")
    for k, v in deltas.items():
        print(f"  {k:30s}  {v:+.4f}")


if __name__ == "__main__":
    main()
