#!/usr/bin/env python3
"""
Regenerate OOF predictions via full-pipeline 5-fold CV, then compute the three
evidentiary numbers for the reframed writeup:

  (1) Clinical-safety: fraction of predictions ≥2 ESI levels from label.
  (2) Error-location: confusion mass distribution across adjacency classes
      (exact / ±1 / ±2+).
  (3) LLM-band overlap: do the cases routed to the LLM (top-2 prob gap < 0.20)
      concentrate on ESI-boundary pairs where human clinicians disagree?

Uses the same ensemble config as the final submission (3 CB + 2 LGB seeds,
same hyperparams) but on a reduced iteration budget (1200 CB / 1000 LGB) for
CV tractability. Error locations are stable across ensemble strength.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

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
OUT_DIR = Path(__file__).resolve().parent

CB_PARAMS = dict(
    loss_function="MultiClass", eval_metric="TotalF1",
    iterations=1200, learning_rate=0.05, depth=8,
    l2_leaf_reg=5, min_data_in_leaf=10, random_strength=1.0,
    bootstrap_type="Bernoulli", subsample=0.85, colsample_bylevel=0.8,
    auto_class_weights="Balanced",
)
LGB_PARAMS = dict(
    objective="multiclass", num_class=5, n_estimators=1000,
    learning_rate=0.05, max_depth=8, num_leaves=127,
    min_child_samples=15, subsample=0.85, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=5.0, class_weight="balanced",
    verbose=-1, n_jobs=-1,
)

LLM_GAP_THRESHOLD = 0.20


def build_train_features():
    print("Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    complaints = pd.read_csv(DATA_DIR / "chief_complaints.csv")
    history = pd.read_csv(DATA_DIR / "patient_history.csv")

    print("Decomposing (bank + complaint)...")
    merged = train.merge(history, on="patient_id", how="left", suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    cc = classify_complaints_batch(complaints[complaints["patient_id"].isin(train["patient_id"])])
    decomps = decompose_dataframe(merged, cc)

    print("Running hard-rule scan on train for audit...")
    hard = {}
    for dec in (triage_patient(d) for d in decomps):
        if dec.method == "rules" and dec.confidence >= 0.95:
            hard[dec.patient_id] = dec.esi_prediction

    print("Building features...")
    feats = build_features(train, complaints, history, decomps)
    phase_cols = [c for c in feats.columns if c.startswith("bank_") and c.endswith("_dev")]
    feats = feats.drop(columns=phase_cols, errors="ignore")

    cohort = fit_cohort_expectations(train, complaints)
    tr_temp = build_temporal_features(train, complaints, cohort)
    feats["temporal_news2_deviation"] = tr_temp["temporal_news2_deviation"].values

    return train, feats, hard


def run_cv_oof(X: pd.DataFrame, y: np.ndarray, cat_cols: list[str],
               n_splits: int = 5, seed: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_classes = 5
    oof_proba = np.zeros((len(X), n_classes), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[Fold {fold}/{n_splits}] train={len(tr_idx)} val={len(val_idx)}")
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        fold_probas = []

        for cb_seed in [42, 123, 777]:
            print(f"  CatBoost seed={cb_seed}")
            cb = CatBoostClassifier(**CB_PARAMS, random_seed=cb_seed + fold, verbose=0)
            cb.fit(X_tr, y_tr, cat_features=cat_cols,
                   eval_set=(X_val, y_val), use_best_model=True,
                   early_stopping_rounds=80, verbose=0)
            fold_probas.append(cb.predict_proba(X_val))
            del cb; gc.collect()

        X_lgb_tr = X_tr.copy()
        X_lgb_val = X_val.copy()
        for c in cat_cols:
            X_lgb_tr[c] = X_lgb_tr[c].astype("category")
            X_lgb_val[c] = X_lgb_val[c].astype("category")

        for lgb_seed in [42, 314]:
            print(f"  LightGBM seed={lgb_seed}")
            lgb = LGBMClassifier(**LGB_PARAMS, random_state=lgb_seed + fold)
            lgb.fit(X_lgb_tr, y_tr, eval_set=[(X_lgb_val, y_val)],
                    callbacks=[])
            fold_probas.append(lgb.predict_proba(X_lgb_val))
            del lgb; gc.collect()

        weights = [0.6 / 3] * 3 + [0.4 / 2] * 2
        blend = sum(w * p for w, p in zip(weights, fold_probas))
        oof_proba[val_idx] = blend.astype(np.float32)

        val_pred_argmax = blend.argmax(axis=1) + 1
        fold_acc = (val_pred_argmax == y_val).mean()
        fold_qwk = cohen_kappa_score(y_val, val_pred_argmax, weights="quadratic")
        print(f"  fold {fold}: acc={fold_acc:.4f}  qwk={fold_qwk:.4f}")

    return oof_proba


def apply_pipeline_merge(train_df, oof_proba, hard_rules, thresholds):
    """Apply hard-rule overrides + QWK thresholds to OOF probs to produce
    the final OOF predictions that correspond to how the submission pipeline
    would have decided on training patients."""
    pids = train_df["patient_id"].values
    # QWK threshold assignment on probability centroid
    qwk_preds = predict_with_thresholds(oof_proba, thresholds)
    final = np.array(qwk_preds, dtype=int)
    method = np.array(["ensemble+qwk"] * len(pids), dtype=object)
    for i, pid in enumerate(pids):
        if pid in hard_rules:
            final[i] = hard_rules[pid]
            method[i] = "rules"
    # LLM routing: gap < 0.20
    sorted_p = np.sort(oof_proba, axis=1)
    gap = sorted_p[:, -1] - sorted_p[:, -2]
    llm_routed = (gap < LLM_GAP_THRESHOLD) & (method != "rules")
    return final, method, gap, llm_routed


def summarize(y_true, y_pred, method, gap, llm_routed):
    out = {}
    # --- (1) Clinical safety ---
    dev = np.abs(y_pred - y_true)
    n = len(y_true)
    out["n_total"] = int(n)
    out["exact_match"] = int((dev == 0).sum())
    out["off_by_1"] = int((dev == 1).sum())
    out["off_by_2"] = int((dev == 2).sum())
    out["off_by_3"] = int((dev == 3).sum())
    out["off_by_4"] = int((dev == 4).sum())
    out["pct_exact"] = float((dev == 0).mean())
    out["pct_within_1"] = float((dev <= 1).mean())
    out["pct_ge_2"] = float((dev >= 2).mean())
    out["n_ge_2"] = int((dev >= 2).sum())

    # --- Overall metrics ---
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    out["accuracy"] = float((y_pred == y_true).mean())
    out["qwk"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    out["linear_kappa"] = float(cohen_kappa_score(y_true, y_pred, weights="linear"))

    # --- (2) Error location ---
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    out["confusion_matrix"] = cm.tolist()

    # off-by-1 decomposition by boundary pair
    err_mask = y_pred != y_true
    boundaries = {}
    for lo, hi in [(1, 2), (2, 3), (3, 4), (4, 5)]:
        pair = (((y_true == lo) & (y_pred == hi)) |
                ((y_true == hi) & (y_pred == lo)))
        boundaries[f"{lo}_vs_{hi}"] = int(pair.sum())
    out["err_by_boundary"] = boundaries
    out["err_total"] = int(err_mask.sum())
    out["pct_err_on_3_4_boundary"] = (
        float(boundaries["3_vs_4"] / max(err_mask.sum(), 1))
    )
    out["pct_err_on_2_3_boundary"] = (
        float(boundaries["2_vs_3"] / max(err_mask.sum(), 1))
    )
    out["pct_err_on_high_acuity_boundary"] = float(
        (boundaries["1_vs_2"] + boundaries["2_vs_3"]) / max(err_mask.sum(), 1)
    )
    out["pct_err_on_low_acuity_boundary"] = float(
        (boundaries["3_vs_4"] + boundaries["4_vs_5"]) / max(err_mask.sum(), 1)
    )

    # --- (3) LLM-band overlap ---
    out["n_llm_routed"] = int(llm_routed.sum())
    out["pct_llm_routed"] = float(llm_routed.mean())
    if llm_routed.sum() > 0:
        llm_true = y_true[llm_routed]
        llm_pred = y_pred[llm_routed]
        llm_err = llm_pred != llm_true
        llm_boundaries = {}
        for lo, hi in [(1, 2), (2, 3), (3, 4), (4, 5)]:
            pair = (((llm_true == lo) & (llm_pred == hi)) |
                    ((llm_true == hi) & (llm_pred == lo)))
            llm_boundaries[f"{lo}_vs_{hi}"] = int(pair.sum())
        out["llm_err_by_boundary"] = llm_boundaries
        out["llm_err_total"] = int(llm_err.sum())
        out["llm_band_true_dist"] = {
            str(i): int((llm_true == i).sum()) for i in range(1, 6)
        }
    # Where are the LLM-routed cases, clinically? — by true label
    out["llm_band_by_true_label"] = {
        str(i): int((llm_routed & (y_true == i)).sum()) for i in range(1, 6)
    }

    # Method-stratified numbers
    out["by_method"] = {}
    for m in np.unique(method):
        mask = method == m
        if mask.sum() == 0:
            continue
        out["by_method"][str(m)] = {
            "n": int(mask.sum()),
            "accuracy": float((y_pred[mask] == y_true[mask]).mean()),
            "pct_ge_2": float((np.abs(y_pred[mask] - y_true[mask]) >= 2).mean()),
        }
    return out


def main():
    train, feats, hard = build_train_features()
    y = train["triage_acuity"].values

    X, _, cat_cols = prepare_xy(feats, train["triage_acuity"])
    print(f"Feature matrix: {X.shape}  cat_cols={len(cat_cols)}")

    print(f"\nStarting 5-fold CV OOF generation ...")
    oof_proba = run_cv_oof(X, y, cat_cols, n_splits=5, seed=42)

    print("\nOptimizing QWK thresholds on OOF ...")
    thresholds, oof_qwk_raw = optimize_thresholds(oof_proba, y)
    print(f"  thresholds={thresholds.tolist()}  OOF QWK={oof_qwk_raw:.4f}")

    print("\nApplying hard-rule overrides + QWK ...")
    final, method, gap, llm_routed = apply_pipeline_merge(
        train, oof_proba, hard, thresholds
    )

    print("\nSummarizing evidentiary numbers ...")
    report = summarize(y, final, method, gap, llm_routed)
    report["oof_thresholds"] = thresholds.tolist()
    report["oof_qwk_argmax"] = float(oof_qwk_raw)

    # Persist artifacts
    np.save(OUT_DIR / "oof_proba.npy", oof_proba)
    np.save(OUT_DIR / "oof_final.npy", final)
    np.save(OUT_DIR / "oof_method.npy", method)
    np.save(OUT_DIR / "oof_gap.npy", gap)
    (OUT_DIR / "oof_summary.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items()
                      if not isinstance(v, (dict, list))}, indent=2))
    print(f"\nArtifacts in: {OUT_DIR}")


if __name__ == "__main__":
    main()
