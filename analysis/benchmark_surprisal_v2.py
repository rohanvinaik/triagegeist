#!/usr/bin/env python3
"""Supplementary regime-split benchmark: baseline + surprisal-basis features.

Runs a single variant (`hard_bucket_v2`) using the same fold seed and regime
gate as the main regime-split benchmark (analysis/benchmark_phase_features_foldsafe.py)
so results are directly comparable. Per-fold (easy, hard) QWK lines use the
same log format, so analysis/regime_stats.py can parse them alongside
baseline/phase/full_stack from the main log.

Feature set: baseline (no phase, no Tier-B) + 87 surprisal-basis features
from src/surprisal_features.py. Fold-safe: surprisal baseline fit on train
partition only.

Output: analysis/surprisal_benchmark.json
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
from src.surprisal_features import (
    build_surprisal_features, fit_surprisal_baseline,
)

DATA = PROJECT_ROOT / "data" / "extracted"
OUT = PROJECT_ROOT / "analysis" / "surprisal_benchmark.json"
HARD_FRACTION = 0.10


def strip_phase(X: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in X.columns if
            c.endswith("_dev") or c.startswith("bank_r_") or
            c == "bank_psi" or c == "bank_coherence_spread"]
    return X.drop(columns=drop)


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
    if len(y_true) == 0:
        return float("nan")
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def _f1(y_true, y_pred):
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro"))


def main():
    print("Loading data...")
    train = pd.read_csv(DATA / "train.csv")
    complaints = pd.read_csv(DATA / "chief_complaints.csv")
    history = pd.read_csv(DATA / "patient_history.csv")

    print("Decomposing banks once on full train (12s)...")
    merged = train.merge(history, on="patient_id", how="left",
                         suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    cc = classify_complaints_batch(
        complaints[complaints.patient_id.isin(train.patient_id)])
    decomps = decompose_dataframe(merged, cc)
    pid_to_decomp = {d.patient_id: d for d in decomps}

    print("Computing r_total gate...")
    r_map = compute_r_total(decomps)
    r_total = np.array([r_map[pid] for pid in train.patient_id.values])
    r_threshold = float(np.quantile(r_total, HARD_FRACTION))
    hard_mask = r_total <= r_threshold
    print(f"  r threshold: {r_threshold:.3f}  (hard={int(hard_mask.sum())}, "
          f"easy={int((~hard_mask).sum())})")

    y = train.triage_acuity.values
    n = len(train)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_proba = np.zeros((n, 5))
    oof_preds = np.zeros(n, dtype=int)

    label = "hard_bucket_v2"
    print(f"\n{'=' * 60}")
    print(f"{label} (baseline + surprisal-basis features, no phase, no Tier-B)")
    print("=" * 60)

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        print(f"[{label}] fold {fold}/5  preparing features ...")

        tr_df = train.iloc[tr_idx].reset_index(drop=True)
        va_df = train.iloc[va_idx].reset_index(drop=True)
        tr_decomps = [pid_to_decomp[p] for p in tr_df.patient_id]
        va_decomps = [pid_to_decomp[p] for p in va_df.patient_id]

        # Build baseline features (no phase)
        tr_feats = build_features(
            tr_df.drop(columns=["triage_acuity"], errors="ignore"),
            complaints, history, tr_decomps,
        )
        va_feats = build_features(
            va_df.drop(columns=["triage_acuity"], errors="ignore"),
            complaints, history, va_decomps,
        )

        # Fit surprisal baseline on fold train decomps, apply to both
        surprisal_baseline = fit_surprisal_baseline(tr_decomps)
        tr_surp = build_surprisal_features(tr_decomps, surprisal_baseline)
        va_surp = build_surprisal_features(va_decomps, surprisal_baseline)
        # Align on patient_id join → reindex against tr_feats index
        tr_surp = tr_surp.reindex(tr_df.patient_id).reset_index(drop=True)
        va_surp = va_surp.reindex(va_df.patient_id).reset_index(drop=True)
        tr_surp.index = tr_feats.index
        va_surp.index = va_feats.index
        tr_feats = pd.concat([tr_feats, tr_surp], axis=1)
        va_feats = pd.concat([va_feats, va_surp], axis=1)

        X_tr, _, cat_cols = prepare_xy(tr_feats)
        X_va, _, _ = prepare_xy(va_feats)
        # Strip phase (we want baseline + surprisal only)
        X_tr = strip_phase(X_tr)
        X_va = strip_phase(X_va)

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
            random_seed=42 + fold, verbose=0,
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
        print(f"[{label}] fold {fold}  "
              f"all: mF1={_f1(y[va_idx], oof_preds[va_idx]):.4f} "
              f"qwk={_qwk(y[va_idx], oof_preds[va_idx]):.4f}  |  "
              f"easy (n={len(fold_easy)}) qwk={_qwk(y[fold_easy], oof_preds[fold_easy]):.4f}  |  "
              f"hard (n={len(fold_hard)}) qwk={_qwk(y[fold_hard], oof_preds[fold_hard]):.4f}  "
              f"n_feat={X_tr.shape[1]}  elapsed={time.time()-t0:.0f}s")

    easy_mask = ~hard_mask
    res = {
        "label": label,
        "include_surprisal": True,
        "n_easy": int(easy_mask.sum()),
        "n_hard": int(hard_mask.sum()),
        "argmax_macro_f1_all": _f1(y, oof_preds),
        "argmax_qwk_all": _qwk(y, oof_preds),
        "argmax_accuracy_all": float(accuracy_score(y, oof_preds)),
        "argmax_macro_f1_easy": _f1(y[easy_mask], oof_preds[easy_mask]),
        "argmax_qwk_easy": _qwk(y[easy_mask], oof_preds[easy_mask]),
        "argmax_macro_f1_hard": _f1(y[hard_mask], oof_preds[hard_mask]),
        "argmax_qwk_hard": _qwk(y[hard_mask], oof_preds[hard_mask]),
    }
    thr, best_qwk = optimize_thresholds(oof_proba, y)
    tuned = predict_with_thresholds(oof_proba, thr)
    res.update({
        "tuned_qwk_all": float(best_qwk),
        "tuned_qwk_easy": _qwk(y[easy_mask], tuned[easy_mask]),
        "tuned_qwk_hard": _qwk(y[hard_mask], tuned[hard_mask]),
        "tuned_macro_f1_all": _f1(y, tuned),
        "thresholds": thr.tolist(),
        "elapsed_sec": round(time.time() - t0, 1),
        "hard_threshold_r_total": r_threshold,
    })

    print(f"[{label}] DONE:")
    print(f"  all:  argmax mF1={res['argmax_macro_f1_all']:.4f} "
          f"qwk={res['argmax_qwk_all']:.4f} | tuned qwk={res['tuned_qwk_all']:.4f}")
    print(f"  easy: argmax qwk={res['argmax_qwk_easy']:.4f} | "
          f"tuned qwk={res['tuned_qwk_easy']:.4f}")
    print(f"  hard: argmax qwk={res['argmax_qwk_hard']:.4f} | "
          f"tuned qwk={res['tuned_qwk_hard']:.4f}")

    OUT.write_text(json.dumps(res, indent=2, default=str))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
