#!/usr/bin/env python3
"""
Generate final test submission.

Multi-scale pipeline:
  Scale 0: Hard rules (GCS≤8, cardiac arrest) → deterministic ESI
  Scale 1: CatBoost ensemble (3 seeds) + LightGBM ensemble (2 seeds)
  Scale 2: LLM residual (Qwen 3 8B via Ollama) for uncertain boundary cases
"""

from __future__ import annotations

import argparse
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
from src.llm_residual import heal_uncertain_cases, identify_uncertain_cases
from src.model import CAT_COLS, prepare_xy

DATA_DIR = PROJECT_ROOT / "data" / "extracted"
SUBMISSION_DIR = Path(__file__).resolve().parent


def _train_catboost_ensemble(X_train, y_train, cat_cols):
    """Train 3 CatBoost models with different seeds."""
    models = []
    for seed in [42, 123, 777]:
        print(f"  CatBoost (seed={seed})...")
        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=2000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=5,
            min_data_in_leaf=10,
            random_strength=1.0,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            colsample_bylevel=0.8,
            random_seed=seed,
            verbose=500,
            auto_class_weights="Balanced",
        )
        model.fit(X_train, y_train, cat_features=cat_cols)
        models.append(("catboost", model))
    return models


def _train_lightgbm_ensemble(X_train, y_train, cat_cols):
    """Train 2 LightGBM models with different seeds."""
    # LightGBM needs numeric categoricals
    X_lgb = X_train.copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype("category")

    models = []
    for seed in [42, 314]:
        print(f"  LightGBM (seed={seed})...")
        model = LGBMClassifier(
            objective="multiclass",
            num_class=5,
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=127,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5.0,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_lgb, y_train)
        models.append(("lightgbm", model))
    return models


def _predict_ensemble(models, X_test, cat_cols):
    """Average predictions across CatBoost + LightGBM ensemble."""
    proba_sum = None
    n_models = 0

    for model_type, model in models:
        if model_type == "lightgbm":
            X = X_test.copy()
            for c in cat_cols:
                X[c] = X[c].astype("category")
            proba = model.predict_proba(X)
        else:
            proba = model.predict_proba(X_test)

        if proba_sum is None:
            proba_sum = proba
        else:
            proba_sum += proba
        n_models += 1

    return proba_sum / n_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM residual healing")
    parser.add_argument("--llm-threshold", type=float, default=0.15,
                        help="Uncertainty threshold for LLM healing")
    args = parser.parse_args()

    print("Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    complaints = pd.read_csv(DATA_DIR / "chief_complaints.csv")
    history = pd.read_csv(DATA_DIR / "patient_history.csv")

    # ---------------------------------------------------------------
    # Scale 0: Bank decomposition + hard rules
    # ---------------------------------------------------------------
    print("\n=== Scale 0: Bank Decomposition ===")
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

    test_decisions = [triage_patient(d) for d in test_decomps]
    hard_rule_preds = {}
    for dec in test_decisions:
        if dec.method == "rules" and dec.confidence >= 0.95:
            hard_rule_preds[dec.patient_id] = dec.esi_prediction
    print(f"  Hard rules resolved: {len(hard_rule_preds)}/{len(test)}")

    # ---------------------------------------------------------------
    # Scale 1: CatBoost + LightGBM ensemble
    # ---------------------------------------------------------------
    print("\n=== Scale 1: Model Ensemble ===")
    train_features = build_features(train, complaints, history, train_decomps)
    test_features = build_features(test, complaints, history, test_decomps)

    X_train, y_train, cat_cols = prepare_xy(train_features, train["triage_acuity"])
    X_test, _, _ = prepare_xy(test_features)

    missing_in_test = set(X_train.columns) - set(X_test.columns)
    for c in missing_in_test:
        X_test[c] = 0
    X_test = X_test[X_train.columns]

    cb_models = _train_catboost_ensemble(X_train, y_train, cat_cols)
    lgb_models = _train_lightgbm_ensemble(X_train, y_train, cat_cols)
    all_models = cb_models + lgb_models

    print(f"\n  Ensemble: {len(cb_models)} CatBoost + {len(lgb_models)} LightGBM")

    proba_avg = _predict_ensemble(all_models, X_test, cat_cols)
    model_preds = proba_avg.argmax(axis=1) + 1  # 0-indexed → 1-5

    # Merge hard rules
    test_ids = test["patient_id"].tolist()
    final_preds = []
    n_rules = 0
    for i, pid in enumerate(test_ids):
        if pid in hard_rule_preds:
            final_preds.append(hard_rule_preds[pid])
            n_rules += 1
        else:
            final_preds.append(int(model_preds[i]))

    print(f"  After rules: {n_rules} rules, {len(test_ids) - n_rules} model")

    # ---------------------------------------------------------------
    # Scale 2: LLM residual for uncertain cases
    # ---------------------------------------------------------------
    if not args.no_llm:
        print("\n=== Scale 2: LLM Residual (Qwen 3 8B) ===")
        uncertain = identify_uncertain_cases(proba_avg, args.llm_threshold)
        # Filter out cases already handled by hard rules
        uncertain = [(idx, classes, gap) for idx, classes, gap in uncertain
                     if test_ids[idx] not in hard_rule_preds]
        print(f"  Uncertain cases: {len(uncertain)}")

        if uncertain:
            test_complaints = complaints[
                complaints["patient_id"].isin(test["patient_id"])
            ]
            decisions = heal_uncertain_cases(
                uncertain, test, test_complaints, final_preds
            )
            n_changed = sum(1 for d in decisions if d.original_pred != d.llm_pred)
            print(f"  LLM changed: {n_changed}/{len(decisions)} predictions")
    else:
        print("\n=== Scale 2: LLM skipped (--no-llm) ===")

    # ---------------------------------------------------------------
    # Write submission
    # ---------------------------------------------------------------
    submission = pd.DataFrame({
        "patient_id": test_ids,
        "triage_acuity": final_preds,
    })

    output_path = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)

    print(f"\nSubmission saved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Distribution:\n{submission['triage_acuity'].value_counts().sort_index()}")

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert list(submission.columns) == list(sample.columns)
    assert len(submission) == len(sample)
    print("\nSanity check passed!")


if __name__ == "__main__":
    main()
