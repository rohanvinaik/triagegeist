"""
CatBoost model for residual ESI prediction.

Handles the 10-15% of cases where deterministic rules and coherence
scoring don't produce high-confidence predictions. Uses the
bank signal features as additional inputs (data geometry → model).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold


TARGET = "triage_acuity"
ID_COL = "patient_id"

# Categorical columns to pass natively to CatBoost
CAT_COLS = [
    "site_id", "triage_nurse_id", "arrival_mode", "arrival_day",
    "arrival_season", "shift", "age_group", "sex", "language",
    "insurance_type", "transport_origin", "pain_location",
    "mental_status_triage", "chief_complaint_system",
]


def prepare_xy(features_df: pd.DataFrame,
                target_series: pd.Series | None = None,
                ) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    """Prepare feature matrix, dropping ID and non-feature columns."""
    drop = [c for c in [ID_COL, TARGET] if c in features_df.columns]
    X = features_df.drop(columns=drop, errors="ignore").copy()

    # Identify categorical columns present
    cat_present = [c for c in CAT_COLS if c in X.columns]

    # Fill missing categoricals and force string type
    for c in cat_present:
        X[c] = X[c].fillna("_missing_").astype(str)

    # Fill missing numerics with median
    num_cols = [c for c in X.columns if c not in cat_present]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    y = None
    if target_series is not None:
        y = target_series.values

    return X, y, cat_present


def train_cv(features_df: pd.DataFrame,
             target_series: pd.Series,
             n_splits: int = 5,
             seed: int = 42,
             ) -> tuple[list[CatBoostClassifier], np.ndarray, dict]:
    """Train CatBoost with stratified k-fold CV.

    Returns:
        (models, oof_predictions, metrics_dict)
    """
    X, y, cat_cols = prepare_xy(features_df, target_series)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models = []
    oof_preds = np.zeros(len(X), dtype=int)
    oof_proba = np.zeros((len(X), y.max()))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

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
            random_seed=seed + fold,
            verbose=500,
            auto_class_weights="Balanced",
        )

        model.fit(
            X_tr, y_tr,
            cat_features=cat_cols,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=100,
        )

        val_pred = model.predict(X_val).reshape(-1).astype(int)
        oof_preds[val_idx] = val_pred

        proba = model.predict_proba(X_val)
        oof_proba[val_idx, :proba.shape[1]] = proba

        fold_f1 = f1_score(y_val, val_pred, average="macro")
        print(f"Fold {fold}: macro F1 = {fold_f1:.4f}")

        models.append(model)

    overall_f1 = f1_score(y, oof_preds, average="macro")
    overall_acc = (oof_preds == y).mean()

    print(f"\nOverall CV: macro F1 = {overall_f1:.4f}, acc = {overall_acc:.4f}")
    print(classification_report(y, oof_preds))

    metrics = {
        "macro_f1": overall_f1,
        "accuracy": overall_acc,
        "n_splits": n_splits,
    }

    return models, oof_preds, metrics


def predict_ensemble(models: list[CatBoostClassifier],
                     features_df: pd.DataFrame,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Predict using ensemble of CV models (averaged probabilities).

    Returns:
        (predictions, probabilities)
    """
    X, _, cat_cols = prepare_xy(features_df)

    proba_sum = None
    for model in models:
        proba = model.predict_proba(X)
        if proba_sum is None:
            proba_sum = proba
        else:
            proba_sum += proba

    proba_avg = proba_sum / len(models)
    predictions = proba_avg.argmax(axis=1) + 1  # CatBoost 0-indexes, ESI is 1-5

    return predictions, proba_avg


def save_models(models: list[CatBoostClassifier], output_dir: Path):
    """Save trained models to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(models):
        model.save_model(str(output_dir / f"catboost_fold_{i}.cbm"))


def load_models(model_dir: Path) -> list[CatBoostClassifier]:
    """Load trained models from disk."""
    models = []
    for path in sorted(model_dir.glob("catboost_fold_*.cbm")):
        model = CatBoostClassifier()
        model.load_model(str(path))
        models.append(model)
    return models
