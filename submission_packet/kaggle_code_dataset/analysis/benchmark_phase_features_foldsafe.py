#!/usr/bin/env python3
"""Honest-signal CV benchmark with regime-split reporting.

Two-regime architecture: the pipeline routes the easy 90-95% through the
standard QWK-optimized ensemble and the hard 5-10% "tail" through a
coherence-based process with phase features + Tier-B context. Gate is
`bank_r_total` (Kuramoto order parameter of the confident banks, computed
from bank decompositions — no target dependency).

For each variant (baseline / +phase / +full_stack) we report QWK separately
on EASY rows (top 90% by r_total) and HARD rows (bottom 10%), plus the
aggregate. A real architectural win shows up as "+phase lifts hard_qwk
without moving easy_qwk" — the exact regime-separated claim the architecture
is built around.

No complaint target-encoding (declined shortcut — see feature_engine.py).
Tier-B artifacts fit fold-locally.

Output: analysis/phase_feature_benchmark_foldsafe.json
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
OUT = PROJECT_ROOT / "analysis" / "phase_feature_benchmark_foldsafe.json"

# Fraction of rows routed to the HARD regime (lowest bank-r quantile).
HARD_FRACTION = 0.10


def compute_r_total_per_patient(decomps: list) -> dict[str, float]:
    """Compute the Kuramoto order parameter r across confident banks.

    Mirrors the logic in feature_engine._build_bank_features so that the
    regime gate is the same signal the pipeline already uses. Pure function
    of clinical bank signals — no target dependency, safe as a gate.
    """
    out = {}
    for d in decomps:
        thetas, weights = [], []
        for _, sig in d.signals.items():
            if sig.confidence > 0.05:
                theta = (sig.esi_estimate - 1.0) / 4.0 * np.pi
                thetas.append(theta)
                weights.append(sig.confidence)
        if thetas:
            t = np.asarray(thetas, dtype=float)
            w = np.asarray(weights, dtype=float)
            z = (np.exp(1j * t) * w).sum() / w.sum()
            out[d.patient_id] = float(np.abs(z))
        else:
            out[d.patient_id] = 0.0
    return out


def strip_phase(X: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in X.columns if
            c.endswith("_dev") or c.startswith("bank_r_") or
            c == "bank_psi" or c == "bank_coherence_spread"]
    return X.drop(columns=drop)


def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro"))


def run_foldsafe_cv(train: pd.DataFrame, complaints: pd.DataFrame,
                    history: pd.DataFrame, decomps_all: list,
                    r_total: np.ndarray, hard_mask: np.ndarray,
                    *, include_phase: bool, include_tier_b: bool,
                    label: str,
                    n_splits: int = 5, seed: int = 42) -> dict:
    y = train.triage_acuity.values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_proba = np.zeros((len(train), 5))
    oof_preds = np.zeros(len(train), dtype=int)
    pid_to_decomp = {d.patient_id: d for d in decomps_all}

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        print(f"[{label}] fold {fold}/{n_splits}  preparing features ...")

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

        if include_tier_b:
            artifacts = fit_tier_b_artifacts(tr_df, complaints, history)
            tier_b_tr = build_tier_b_features(tr_df.reset_index(drop=True),
                                               complaints, history, artifacts)
            tier_b_va = build_tier_b_features(va_df.reset_index(drop=True),
                                               complaints, history, artifacts)
            tier_b_tr.index = tr_feats.index
            tier_b_va.index = va_feats.index
            tr_feats = pd.concat([tr_feats, tier_b_tr], axis=1)
            va_feats = pd.concat([va_feats, tier_b_va], axis=1)

        X_tr, _, cat_cols = prepare_xy(tr_feats)
        X_va, _, _ = prepare_xy(va_feats)

        if not include_phase:
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
            random_seed=seed + fold, verbose=0,
            auto_class_weights="Balanced",
        )
        m.fit(X_tr, y[tr_idx], cat_features=cat_cols,
              eval_set=(X_va, y[va_idx]), use_best_model=True,
              early_stopping_rounds=100)

        proba = m.predict_proba(X_va)
        oof_proba[va_idx, :proba.shape[1]] = proba
        oof_preds[va_idx] = m.predict(X_va).reshape(-1).astype(int)

        # Per-fold telemetry, stratified by regime
        fold_easy = va_idx[~hard_mask[va_idx]]
        fold_hard = va_idx[hard_mask[va_idx]]
        print(f"[{label}] fold {fold}  "
              f"all: mF1={_f1(y[va_idx], oof_preds[va_idx]):.4f} "
              f"qwk={_qwk(y[va_idx], oof_preds[va_idx]):.4f}  |  "
              f"easy (n={len(fold_easy)}) qwk={_qwk(y[fold_easy], oof_preds[fold_easy]):.4f}  |  "
              f"hard (n={len(fold_hard)}) qwk={_qwk(y[fold_hard], oof_preds[fold_hard]):.4f}  "
              f"elapsed={time.time()-t0:.0f}s")

    # Aggregate — separately on easy, hard, and all
    easy_mask = ~hard_mask
    res = {
        "label": label,
        "include_phase": include_phase,
        "include_tier_b": include_tier_b,
        "n_easy": int(easy_mask.sum()),
        "n_hard": int(hard_mask.sum()),
        # Aggregate (pooled across both regimes)
        "argmax_macro_f1_all": _f1(y, oof_preds),
        "argmax_qwk_all": _qwk(y, oof_preds),
        "argmax_accuracy_all": float(accuracy_score(y, oof_preds)),
        # Regime-separated
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
    })
    print(f"[{label}] DONE:")
    print(f"  all:  argmax mF1={res['argmax_macro_f1_all']:.4f} "
          f"qwk={res['argmax_qwk_all']:.4f} | tuned qwk={res['tuned_qwk_all']:.4f}")
    print(f"  easy: argmax qwk={res['argmax_qwk_easy']:.4f} | "
          f"tuned qwk={res['tuned_qwk_easy']:.4f}")
    print(f"  hard: argmax qwk={res['argmax_qwk_hard']:.4f} | "
          f"tuned qwk={res['tuned_qwk_hard']:.4f}")
    return res


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

    print("Computing bank_r_total gate per patient...")
    r_map = compute_r_total_per_patient(decomps)
    r_total = np.array([r_map[pid] for pid in train.patient_id.values])
    r_threshold = float(np.quantile(r_total, HARD_FRACTION))
    hard_mask = r_total <= r_threshold
    print(f"  r_total range: [{r_total.min():.3f}, {r_total.max():.3f}]")
    print(f"  r_total mean:  {r_total.mean():.3f}, median: {np.median(r_total):.3f}")
    print(f"  Hard threshold ({HARD_FRACTION*100:.0f}th pctile): "
          f"r <= {r_threshold:.3f}")
    print(f"  Hard rows: {int(hard_mask.sum())}  Easy rows: {int((~hard_mask).sum())}")

    # Distribution of ESI in hard vs easy — sanity check
    y_all = train.triage_acuity.values
    print(f"  Easy ESI dist: {np.bincount(y_all[~hard_mask], minlength=6)[1:].tolist()}")
    print(f"  Hard ESI dist: {np.bincount(y_all[hard_mask], minlength=6)[1:].tolist()}")

    print("\n" + "=" * 60)
    print("BASELINE (no phase, no Tier-B, no complaint TE)")
    print("=" * 60)
    baseline = run_foldsafe_cv(train, complaints, history, decomps,
                                r_total, hard_mask,
                                include_phase=False, include_tier_b=False,
                                label="baseline")

    print("\n" + "=" * 60)
    print("PHASE (+ Kuramoto phase features)")
    print("=" * 60)
    phase = run_foldsafe_cv(train, complaints, history, decomps,
                             r_total, hard_mask,
                             include_phase=True, include_tier_b=False,
                             label="phase")

    print("\n" + "=" * 60)
    print("FULL_STACK (+ phase + Tier-B)")
    print("=" * 60)
    full = run_foldsafe_cv(train, complaints, history, decomps,
                            r_total, hard_mask,
                            include_phase=True, include_tier_b=True,
                            label="full_stack")

    def _delta_by_regime(a, b):
        """Delta b-a for each regime metric."""
        keys = [k for k in a if isinstance(a.get(k), float)
                and isinstance(b.get(k), float)
                and k != "elapsed_sec"]
        return {k: b[k] - a[k] for k in keys}

    result = {
        "hard_fraction": HARD_FRACTION,
        "hard_threshold_r_total": r_threshold,
        "n_total": int(len(train)),
        "n_easy": int((~hard_mask).sum()),
        "n_hard": int(hard_mask.sum()),
        "baseline": baseline,
        "phase": phase,
        "full_stack": full,
        "delta_phase_minus_baseline": _delta_by_regime(baseline, phase),
        "delta_full_minus_phase": _delta_by_regime(phase, full),
        "delta_full_minus_baseline": _delta_by_regime(baseline, full),
        "notes": (
            "Regime-split benchmark: bank_r_total (Kuramoto order parameter) "
            "separates the bottom 10% (HARD, banks disagree → hard residual) "
            "from the top 90% (EASY, banks agree → standard pipeline). "
            "Each variant's OOF predictions are reported separately on easy/hard "
            "to test the two-regime claim: phase and Tier-B features should "
            "lift hard_qwk without hurting easy_qwk. "
            "cc_condition_prior removed (declined shortcut)."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {OUT}")

    print(f"\n--- LIFT phase vs baseline ---")
    for k, v in result["delta_phase_minus_baseline"].items():
        print(f"  {k:30s}  {v:+.4f}")
    print(f"\n--- LIFT full_stack vs phase ---")
    for k, v in result["delta_full_minus_phase"].items():
        print(f"  {k:30s}  {v:+.4f}")
    print(f"\n--- LIFT full_stack vs baseline ---")
    for k, v in result["delta_full_minus_baseline"].items():
        print(f"  {k:30s}  {v:+.4f}")

    print(f"\n=== Summary ===")
    for name, r in [("baseline", baseline), ("phase", phase),
                     ("full_stack", full)]:
        print(f"  {name:12s}  easy qwk={r['argmax_qwk_easy']:.4f}  "
              f"hard qwk={r['argmax_qwk_hard']:.4f}  "
              f"all qwk={r['argmax_qwk_all']:.4f}")


if __name__ == "__main__":
    main()
