"""
Multi-scale pipeline orchestrator for ESI triage prediction.

Implements the NCEMS multi-scale residual decomposition pattern:

Scale 0: Bank decomposition + deterministic rules (80-90% of signal)
         Hard ESI rules fire for clinically unambiguous cases.

Scale 1: Cross-bank coherence scoring (Kuramoto convergence)
         When banks agree, confidence boosts nonlinearly.

Scale 2: CatBoost model for residual (10-15% of signal)
         Handles cases where rules + coherence are uncertain.
         Bank signals feed as features (data geometry → model input).

Scale 3: SOTA LLM for genuinely ambiguous cases (last 5%)
         Only fires when model confidence is below threshold.
         [Not yet implemented — placeholder for API calls]

Each scale resolves what it can and passes residuals upward.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .banks import Bank, BankDecomposition, decompose_dataframe
from .coherence import TriageDecision, triage_patient
from .complaint_lexicon import classify_complaints_batch
from .feature_engine import build_features


# Confidence threshold: above this, trust coherence; below, use model
COHERENCE_CONFIDENCE_THRESHOLD = 0.70


@dataclass
class PipelineResult:
    """Full pipeline output for a dataset."""
    predictions: pd.Series          # patient_id → ESI (1-5)
    confidences: pd.Series          # patient_id → confidence (0-1)
    methods: pd.Series              # patient_id → method used
    decisions: list[TriageDecision] # full decision trail
    stats: dict                     # pipeline statistics


def _load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame,
                                         pd.DataFrame, pd.DataFrame | None]:
    """Load competition data files."""
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    complaints = pd.read_csv(data_dir / "chief_complaints.csv")
    history = pd.read_csv(data_dir / "patient_history.csv")
    return train, test, complaints, history


def run_geometric_scales(df: pd.DataFrame,
                         complaints_df: pd.DataFrame,
                         history_df: pd.DataFrame | None = None,
                         ) -> tuple[list[TriageDecision],
                                    list[BankDecomposition]]:
    """Run Scale 0 (banks) + Scale 1 (coherence) on a DataFrame.

    Returns:
        (decisions, bank_decompositions)
    """
    # Merge history into main df for bank decomposition
    merged = df.copy()
    if history_df is not None:
        merged = merged.merge(history_df, on="patient_id", how="left",
                              suffixes=("", "_hx_dup"))
        dup_cols = [c for c in merged.columns if c.endswith("_hx_dup")]
        merged = merged.drop(columns=dup_cols)

    # Scale 0a: Complaint lexicon
    patient_complaints = complaints_df[
        complaints_df["patient_id"].isin(df["patient_id"])
    ]
    complaint_signals = classify_complaints_batch(patient_complaints)

    # Scale 0b: Bank decomposition (all 11 banks)
    decompositions = decompose_dataframe(merged, complaint_signals)

    # Scale 1: Coherence scoring + hard rules
    decisions = [triage_patient(d) for d in decompositions]

    return decisions, decompositions


def run_full_pipeline(data_dir: Path,
                      mode: str = "benchmark",
                      model_dir: Path | None = None,
                      confidence_threshold: float = COHERENCE_CONFIDENCE_THRESHOLD,
                      ) -> PipelineResult:
    """Run the complete multi-scale pipeline.

    Args:
        data_dir: Path to extracted competition data
        mode: "benchmark" (train+eval) or "submit" (predict test)
        model_dir: Path to saved models (for submit mode)
        confidence_threshold: Below this, use model instead of coherence

    Returns:
        PipelineResult with predictions and diagnostics
    """
    train, test, complaints, history = _load_data(data_dir)

    target_df = test if mode == "submit" else train

    # ---------------------------------------------------------------
    # Scales 0-1: Geometric (deterministic rules + coherence)
    # ---------------------------------------------------------------
    decisions, decompositions = run_geometric_scales(
        target_df, complaints, history
    )

    # Partition: high-confidence (use coherence) vs low-confidence (use model)
    high_conf_idx = []
    low_conf_idx = []
    for i, dec in enumerate(decisions):
        if dec.confidence >= confidence_threshold:
            high_conf_idx.append(i)
        else:
            low_conf_idx.append(i)

    n_total = len(decisions)
    n_geometric = len(high_conf_idx)
    n_model = len(low_conf_idx)

    print(f"Scale 0-1: {n_geometric}/{n_total} resolved by geometry "
          f"({100*n_geometric/n_total:.1f}%)")
    print(f"Scale 2:   {n_model}/{n_total} need model "
          f"({100*n_model/n_total:.1f}%)")

    # ---------------------------------------------------------------
    # Scale 2: CatBoost model for residual
    # ---------------------------------------------------------------
    final_preds = {}
    final_confs = {}
    final_methods = {}

    # Set geometric predictions
    for i in high_conf_idx:
        dec = decisions[i]
        final_preds[dec.patient_id] = dec.esi_prediction
        final_confs[dec.patient_id] = dec.confidence
        final_methods[dec.patient_id] = dec.method

    if n_model > 0:
        # Build features for model
        features = build_features(
            target_df, complaints, history, decompositions
        )

        if mode == "benchmark":
            # Train model on the full training set and use OOF predictions
            from .model import train_cv

            y = train["triage_acuity"]
            train_features = build_features(train, complaints, history,
                                            decompositions)
            models, oof_preds, metrics = train_cv(train_features, y)

            # Build patient_id → index lookup for fast OOF access
            pid_to_idx = {pid: idx for idx, pid in
                          enumerate(train["patient_id"])}

            # For low-confidence cases, use OOF predictions
            for i in low_conf_idx:
                dec = decisions[i]
                pid = dec.patient_id
                idx = pid_to_idx.get(pid)
                if idx is not None:
                    final_preds[pid] = int(oof_preds[idx])
                else:
                    final_preds[pid] = dec.esi_prediction
                final_confs[pid] = 0.60
                final_methods[pid] = "model"

        elif mode == "submit":
            from .model import load_models, predict_ensemble

            if model_dir is None:
                raise ValueError("model_dir required for submit mode")
            models = load_models(model_dir)
            model_preds, model_proba = predict_ensemble(models, features)

            # Map model predictions to low-confidence patients
            test_pid_to_idx = {pid: idx for idx, pid in
                               enumerate(target_df["patient_id"])}
            for i in low_conf_idx:
                dec = decisions[i]
                pid = dec.patient_id
                pid_idx = test_pid_to_idx.get(pid)
                if pid_idx is not None:
                    final_preds[pid] = int(model_preds[pid_idx])
                    final_confs[pid] = float(model_proba[pid_idx].max())
                else:
                    final_preds[pid] = dec.esi_prediction
                    final_confs[pid] = 0.5
                final_methods[pid] = "model"
    else:
        metrics = {}

    # Build output Series
    patient_ids = target_df["patient_id"].tolist()
    pred_series = pd.Series(
        [final_preds.get(pid, 3) for pid in patient_ids],
        index=patient_ids, name="triage_acuity"
    )
    conf_series = pd.Series(
        [final_confs.get(pid, 0.0) for pid in patient_ids],
        index=patient_ids, name="confidence"
    )
    method_series = pd.Series(
        [final_methods.get(pid, "unknown") for pid in patient_ids],
        index=patient_ids, name="method"
    )

    stats = {
        "total": n_total,
        "geometric": n_geometric,
        "model": n_model,
        "geometric_pct": 100 * n_geometric / n_total,
        "model_pct": 100 * n_model / n_total,
        "confidence_threshold": confidence_threshold,
    }
    if metrics:
        stats["model_metrics"] = metrics

    return PipelineResult(
        predictions=pred_series,
        confidences=conf_series,
        methods=method_series,
        decisions=decisions,
        stats=stats,
    )


def evaluate_pipeline(result: PipelineResult,
                      ground_truth: pd.Series) -> dict:
    """Evaluate pipeline predictions against ground truth."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
    )

    y_true = ground_truth.values
    y_pred = result.predictions.loc[ground_truth.index].values

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"Pipeline Evaluation")
    print(f"{'='*60}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred,
                                target_names=["ESI 1", "ESI 2", "ESI 3",
                                              "ESI 4", "ESI 5"]))

    # Per-method breakdown
    methods = result.methods.loc[ground_truth.index]
    for method in methods.unique():
        mask = methods == method
        if mask.sum() > 0:
            method_f1 = f1_score(y_true[mask], y_pred[mask], average="macro")
            print(f"  {method}: n={mask.sum()}, macro F1={method_f1:.4f}")

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
    }
