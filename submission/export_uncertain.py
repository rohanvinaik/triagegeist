#!/usr/bin/env python3
"""
Export uncertain cases for LLM healing.

Runs the model ensemble, identifies cases where the probability gap
between top-2 classes is small, and exports their clinical data
for external LLM processing.
"""

from __future__ import annotations

import json
import sys
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

DATA_DIR = PROJECT_ROOT / "data" / "extracted"
SUBMISSION_DIR = Path(__file__).resolve().parent


def main():
    print("Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    complaints = pd.read_csv(DATA_DIR / "chief_complaints.csv")
    history = pd.read_csv(DATA_DIR / "patient_history.csv")

    # Bank decomposition
    print("Bank decomposition...")
    train_merged = train.merge(history, on="patient_id", how="left",
                               suffixes=("", "_dup"))
    train_merged = train_merged[[c for c in train_merged.columns
                                  if not c.endswith("_dup")]]
    train_cc = classify_complaints_batch(
        complaints[complaints["patient_id"].isin(train["patient_id"])]
    )
    train_decomps = decompose_dataframe(train_merged, train_cc)

    test_merged = test.merge(history, on="patient_id", how="left",
                              suffixes=("", "_dup"))
    test_merged = test_merged[[c for c in test_merged.columns
                                if not c.endswith("_dup")]]
    test_cc = classify_complaints_batch(
        complaints[complaints["patient_id"].isin(test["patient_id"])]
    )
    test_decomps = decompose_dataframe(test_merged, test_cc)

    # Hard rules
    test_decisions = [triage_patient(d) for d in test_decomps]
    hard_rule_pids = {dec.patient_id for dec in test_decisions
                      if dec.method == "rules" and dec.confidence >= 0.95}

    # Build features + train ensemble
    print("Building features...")
    train_features = build_features(train, complaints, history, train_decomps)
    test_features = build_features(test, complaints, history, test_decomps)

    X_train, y_train, cat_cols = prepare_xy(train_features, train["triage_acuity"])
    X_test, _, _ = prepare_xy(test_features)

    missing = set(X_train.columns) - set(X_test.columns)
    for c in missing:
        X_test[c] = 0
    X_test = X_test[X_train.columns]

    # Train ensemble
    print("Training ensemble...")
    all_proba = []

    for seed in [42, 123, 777]:
        print(f"  CatBoost (seed={seed})...")
        model = CatBoostClassifier(
            loss_function="MultiClass", eval_metric="TotalF1",
            iterations=2000, learning_rate=0.05, depth=8,
            l2_leaf_reg=5, min_data_in_leaf=10, random_strength=1.0,
            bootstrap_type="Bernoulli", subsample=0.85, colsample_bylevel=0.8,
            random_seed=seed, verbose=500, auto_class_weights="Balanced",
        )
        model.fit(X_train, y_train, cat_features=cat_cols)
        all_proba.append(model.predict_proba(X_test))

    for seed in [42, 314]:
        print(f"  LightGBM (seed={seed})...")
        X_lgb = X_train.copy()
        X_test_lgb = X_test.copy()
        for c in cat_cols:
            X_lgb[c] = X_lgb[c].astype("category")
            X_test_lgb[c] = X_test_lgb[c].astype("category")

        lgb = LGBMClassifier(
            objective="multiclass", num_class=5, n_estimators=1500,
            learning_rate=0.05, max_depth=8, num_leaves=127,
            min_child_samples=15, subsample=0.85, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=5.0, class_weight="balanced",
            random_state=seed, verbose=-1, n_jobs=-1,
        )
        lgb.fit(X_lgb, y_train)
        all_proba.append(lgb.predict_proba(X_test_lgb))

    proba_avg = np.mean(all_proba, axis=0)
    model_preds = proba_avg.argmax(axis=1) + 1

    # Complaint lookup
    complaint_lookup = {}
    for _, row in complaints.iterrows():
        complaint_lookup[row["patient_id"]] = row.get("chief_complaint_raw", "")

    # Identify uncertain cases with wider threshold
    threshold = 0.20
    uncertain_cases = []
    test_ids = test["patient_id"].tolist()

    for i in range(len(proba_avg)):
        pid = test_ids[i]
        if pid in hard_rule_pids:
            continue
        sorted_idx = proba_avg[i].argsort()[::-1]
        top1_prob = proba_avg[i, sorted_idx[0]]
        top2_prob = proba_avg[i, sorted_idx[1]]
        gap = top1_prob - top2_prob

        if gap < threshold:
            row = test.iloc[i]
            c1, c2 = sorted_idx[0] + 1, sorted_idx[1] + 1
            uncertain_cases.append({
                "patient_id": pid,
                "model_pred": int(model_preds[i]),
                "top1_class": int(c1),
                "top1_prob": round(float(top1_prob), 4),
                "top2_class": int(c2),
                "top2_prob": round(float(top2_prob), 4),
                "gap": round(float(gap), 4),
                "age": int(row.get("age", 0)),
                "sex": str(row.get("sex", "")),
                "arrival_mode": str(row.get("arrival_mode", "")),
                "transport_origin": str(row.get("transport_origin", "")),
                "mental_status": str(row.get("mental_status_triage", "")),
                "heart_rate": float(row.get("heart_rate", 0)),
                "systolic_bp": row.get("systolic_bp"),
                "diastolic_bp": row.get("diastolic_bp"),
                "respiratory_rate": row.get("respiratory_rate"),
                "temperature_c": row.get("temperature_c"),
                "spo2": float(row.get("spo2", 0)),
                "gcs_total": int(row.get("gcs_total", 15)),
                "pain_score": int(row.get("pain_score", 0)),
                "news2_score": float(row.get("news2_score", 0)),
                "chief_complaint": complaint_lookup.get(pid, ""),
                "chief_complaint_system": str(row.get("chief_complaint_system", "")),
                "num_comorbidities": int(row.get("num_comorbidities", 0)),
                "num_active_medications": int(row.get("num_active_medications", 0)),
                "num_prior_ed_visits": int(row.get("num_prior_ed_visits_12m", 0)),
            })

    print(f"\nUncertain cases (gap < {threshold}): {len(uncertain_cases)}")
    print(f"Hard rules: {len(hard_rule_pids)}")
    print(f"Confident model: {len(test_ids) - len(uncertain_cases) - len(hard_rule_pids)}")

    # Save uncertain cases
    out_path = SUBMISSION_DIR / "uncertain_cases.json"
    out_path.write_text(json.dumps(uncertain_cases, indent=2, default=str))
    print(f"Saved to {out_path}")

    # Save model predictions for non-uncertain cases
    baseline = {}
    for i, pid in enumerate(test_ids):
        if pid in hard_rule_pids:
            dec = [d for d in test_decisions if d.patient_id == pid][0]
            baseline[pid] = {"pred": dec.esi_prediction, "method": "rules"}
        else:
            baseline[pid] = {"pred": int(model_preds[i]), "method": "model"}

    base_path = SUBMISSION_DIR / "baseline_preds.json"
    base_path.write_text(json.dumps(baseline, indent=2))
    print(f"Baseline predictions saved to {base_path}")


if __name__ == "__main__":
    main()
