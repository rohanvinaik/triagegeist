#!/usr/bin/env python3
"""
Train the full ensemble once, save every artifact needed for inference.

Run this ONCE. All downstream consumers (the Kaggle notebook and any
deployment) load from `submission/models/` and do not retrain.

Artifacts written to submission/models/:
    cb_seed42.cbm, cb_seed123.cbm, cb_seed777.cbm   — CatBoost weights
    lgb_seed42.txt, lgb_seed314.txt                  — LightGBM weights
    feature_columns.json                             — inference column order
    cat_columns.json                                 — categorical column names
    cohort_expectations.pkl                          — fitted cohort table
    qwk_thresholds.json                              — ordinal thresholds
    test_proba.npy                                   — cached ensemble output
    test_patient_ids.npy                             — row alignment sanity
    hard_rules.json                                  — Scale 0 rule firings on test
    meta.json                                        — training metadata
"""

from __future__ import annotations

import gc
import json
import pickle
import sys
import time
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
from src.temporal_bank import build_temporal_features, fit_cohort_expectations

DATA_DIR = PROJECT_ROOT / "data" / "extracted"
MODELS_DIR = PROJECT_ROOT / "submission" / "models"

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

CB_SEEDS = [42, 123, 777]
LGB_SEEDS = [42, 314]
BLEND_WEIGHTS = [0.6 / 3] * 3 + [0.4 / 2] * 2
QWK_THRESHOLDS = [1.56376089, 2.51698442, 3.55305499, 4.58287457]


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("TriageGeist: one-shot training pass")
    print("=" * 60)

    print("\n[1/6] Loading data ...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    complaints = pd.read_csv(DATA_DIR / "chief_complaints.csv")
    history = pd.read_csv(DATA_DIR / "patient_history.csv")
    print(f"  train={len(train):,}  test={len(test):,}")

    print("\n[2/6] Decomposing + hard-rule scan ...")
    decomps = {}
    for label, df in [("train", train), ("test", test)]:
        merged = df.merge(history, on="patient_id", how="left", suffixes=("", "_dup"))
        merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
        cc = classify_complaints_batch(
            complaints[complaints["patient_id"].isin(df["patient_id"])])
        decomps[label] = decompose_dataframe(merged, cc)

    hard_rules = {}
    for d in decomps["test"]:
        dec = triage_patient(d)
        if dec.method == "rules" and dec.confidence >= 0.95:
            hard_rules[dec.patient_id] = int(dec.esi_prediction)
    print(f"  hard rules fired on {len(hard_rules):,} test patients "
          f"({100 * len(hard_rules) / len(test):.2f}%)")

    print("\n[3/6] Feature engineering ...")
    train_feats = build_features(train, complaints, history, decomps["train"])
    test_feats = build_features(test, complaints, history, decomps["test"])

    phase_cols = [c for c in train_feats.columns
                  if c.startswith("bank_") and c.endswith("_dev")]
    train_feats = train_feats.drop(columns=phase_cols, errors="ignore")
    test_feats = test_feats.drop(columns=phase_cols, errors="ignore")

    cohort = fit_cohort_expectations(train, complaints)
    tr_temp = build_temporal_features(train, complaints, cohort)
    te_temp = build_temporal_features(test, complaints, cohort)
    train_feats["temporal_news2_deviation"] = tr_temp["temporal_news2_deviation"].values
    test_feats["temporal_news2_deviation"] = te_temp["temporal_news2_deviation"].values

    X_train, y_train, cat_cols = prepare_xy(train_feats, train["triage_acuity"])
    X_test, _, _ = prepare_xy(test_feats)
    for c in set(X_train.columns) - set(X_test.columns):
        X_test[c] = 0
    X_test = X_test[X_train.columns]

    feature_columns = list(X_train.columns)
    print(f"  features={len(feature_columns)}  cat_cols={len(cat_cols)}")

    print("\n[4/6] Training 3× CatBoost ...")
    all_test_proba = []
    for seed in CB_SEEDS:
        t = time.time()
        print(f"  CatBoost seed={seed} ...", flush=True)
        cb = CatBoostClassifier(**CB_PARAMS, random_seed=seed, verbose=0)
        cb.fit(X_train, y_train, cat_features=cat_cols)
        cb.save_model(str(MODELS_DIR / f"cb_seed{seed}.cbm"))
        proba = cb.predict_proba(X_test).astype(np.float32)
        all_test_proba.append(proba)
        del cb; gc.collect()
        print(f"    saved cb_seed{seed}.cbm  ({time.time()-t:.0f}s)", flush=True)

    print("\n[5/6] Training 2× LightGBM ...")
    X_lgb = X_train.copy()
    X_test_lgb = X_test.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype("category")
        X_test_lgb[c] = X_test_lgb[c].astype("category")

    for seed in LGB_SEEDS:
        t = time.time()
        print(f"  LightGBM seed={seed} ...", flush=True)
        lgb = LGBMClassifier(**LGB_PARAMS, random_state=seed)
        lgb.fit(X_lgb, y_train)
        lgb.booster_.save_model(str(MODELS_DIR / f"lgb_seed{seed}.txt"))
        proba = lgb.predict_proba(X_test_lgb).astype(np.float32)
        all_test_proba.append(proba)
        del lgb; gc.collect()
        print(f"    saved lgb_seed{seed}.txt  ({time.time()-t:.0f}s)", flush=True)

    print("\n[6/6] Persisting inference artifacts ...")
    test_proba = sum(w * p for w, p in zip(BLEND_WEIGHTS, all_test_proba)).astype(np.float32)
    np.save(MODELS_DIR / "test_proba.npy", test_proba)
    np.save(MODELS_DIR / "test_patient_ids.npy",
            np.array(test["patient_id"].tolist(), dtype=object))

    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(feature_columns))
    (MODELS_DIR / "cat_columns.json").write_text(json.dumps(cat_cols))
    with (MODELS_DIR / "cohort_expectations.pkl").open("wb") as f:
        pickle.dump(cohort, f, protocol=pickle.HIGHEST_PROTOCOL)
    (MODELS_DIR / "qwk_thresholds.json").write_text(json.dumps(QWK_THRESHOLDS))
    (MODELS_DIR / "hard_rules.json").write_text(json.dumps(hard_rules))
    (MODELS_DIR / "meta.json").write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(feature_columns)),
        "n_cat_cols": int(len(cat_cols)),
        "n_hard_rules": int(len(hard_rules)),
        "catboost_seeds": CB_SEEDS,
        "catboost_iters": CB_PARAMS["iterations"],
        "lightgbm_seeds": LGB_SEEDS,
        "lightgbm_n_estimators": LGB_PARAMS["n_estimators"],
        "blend_weights": BLEND_WEIGHTS,
        "qwk_thresholds": QWK_THRESHOLDS,
        "training_time_min": round((time.time() - t0) / 60, 2),
    }, indent=2))

    elapsed = time.time() - t0
    print(f"\nDone. Total time: {elapsed/60:.1f} min")
    print(f"Artifacts in: {MODELS_DIR}")
    for p in sorted(MODELS_DIR.iterdir()):
        print(f"  {p.name:30s} {p.stat().st_size:>12,} B")


if __name__ == "__main__":
    main()
