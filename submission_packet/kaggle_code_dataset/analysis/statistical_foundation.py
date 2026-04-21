#!/usr/bin/env python3
"""
Tier-B statistical foundation — the analysis everything else depends on.

Outputs JSON artifact `analysis/foundation_report.json` consumed by:
  - Kuramoto phase-feature design (which bank subsets are clinically coherent)
  - Compositional bank selection (which triples are truly orthogonal)
  - Clinician-variance encoding (which leakage fields tell us about raters)

Sections:
  1.  Bank signal matrix (11 banks × 80K patients; ESI estimate + confidence).
  2.  Pairwise Pearson + partial correlations across banks.
  3.  Mutual information MI(bank_i; target) and MI(composite; target).
  4.  Conditional MI: MI(bank_i; target | banks_\\i) — unique information per bank.
  5.  Compositional candidates — AgeNormalizedShock, ChronicBurdenResidual,
      ThermoImmuneProduct — tested against raw components.
  6.  Kuramoto order parameter distribution across patients; subset r values.
  7.  Clinician/site variance decomposition (triage_nurse_id, site_id).
  8.  Leakage-as-calibrator feasibility: does disposition correlate with
      our coherence confidence in the expected direction?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.banks import Bank, decompose_dataframe
from src.complaint_lexicon import classify_complaints_batch

OUT = PROJECT_ROOT / "analysis" / "foundation_report.json"
DATA = PROJECT_ROOT / "data" / "extracted"


# ---------------------------------------------------------------------------
# 1-2. Bank signal matrix + correlations
# ---------------------------------------------------------------------------
def build_bank_matrix(train: pd.DataFrame, complaints: pd.DataFrame,
                     history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (esi_matrix, conf_matrix) indexed by patient_id, cols = bank names."""
    merged = train.merge(history, on="patient_id", how="left",
                         suffixes=("", "_dup"))
    merged = merged[[c for c in merged.columns if not c.endswith("_dup")]]
    cc = classify_complaints_batch(
        complaints[complaints.patient_id.isin(train.patient_id)]
    )
    decomps = decompose_dataframe(merged, cc)

    bank_names = [b.value for b in Bank]
    rows_esi, rows_conf = [], []
    for d in decomps:
        esi = {b.value: np.nan for b in Bank}
        conf = {b.value: 0.0 for b in Bank}
        for bank, sig in d.signals.items():
            esi[bank.value] = sig.esi_estimate
            conf[bank.value] = sig.confidence
        rows_esi.append({"patient_id": d.patient_id, **esi})
        rows_conf.append({"patient_id": d.patient_id, **conf})
    esi_m = pd.DataFrame(rows_esi).set_index("patient_id")[bank_names]
    conf_m = pd.DataFrame(rows_conf).set_index("patient_id")[bank_names]
    return esi_m, conf_m


def partial_corr(X: pd.DataFrame) -> pd.DataFrame:
    """Partial correlation from precision matrix. Drops all-NaN columns."""
    X = X.dropna(axis=1, how="all").fillna(X.median())
    cov = np.cov(X.values, rowvar=False)
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        prec = np.linalg.pinv(cov)
    d = np.sqrt(np.abs(np.diag(prec)))
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    return pd.DataFrame(pcorr, index=X.columns, columns=X.columns)


# ---------------------------------------------------------------------------
# 3-4. Mutual information (with target + conditional)
# ---------------------------------------------------------------------------
def mi_with_target(X: pd.DataFrame, y: np.ndarray, n_neighbors: int = 3,
                   sample: int | None = 20000,
                   seed: int = 42) -> pd.Series:
    """MI(col; target) for each col. Subsamples for speed."""
    from sklearn.feature_selection import mutual_info_classif
    rng = np.random.default_rng(seed)
    if sample and len(X) > sample:
        idx = rng.choice(len(X), size=sample, replace=False)
        Xs, ys = X.iloc[idx], y[idx]
    else:
        Xs, ys = X, y
    Xs = Xs.fillna(Xs.median())
    mi = mutual_info_classif(Xs.values, ys, n_neighbors=n_neighbors,
                             random_state=seed)
    return pd.Series(mi, index=X.columns, name="MI")


def conditional_mi_all_but_one(X: pd.DataFrame, y: np.ndarray,
                               sample: int = 20000,
                               seed: int = 42) -> pd.Series:
    """MI(col_i; y | X_\\i) via MI(X, y) − MI(X_\\i, y) approximation."""
    from sklearn.feature_selection import mutual_info_classif
    rng = np.random.default_rng(seed)
    if len(X) > sample:
        idx = rng.choice(len(X), size=sample, replace=False)
        X, y = X.iloc[idx], y[idx]
    X = X.fillna(X.median())

    # Baseline: MI between full X and y via average of univariate (upper bound proxy).
    # For true conditional MI we'd need kNN in joint space. Use regression-residual proxy:
    # for each col, fit a simple regressor predicting col from others, then compute
    # MI of the residual with y.
    from sklearn.linear_model import Ridge
    out = {}
    for c in X.columns:
        other = [x for x in X.columns if x != c]
        if not other:
            out[c] = np.nan
            continue
        reg = Ridge(alpha=1.0).fit(X[other].values, X[c].values)
        resid = X[c].values - reg.predict(X[other].values)
        mi = mutual_info_classif(resid.reshape(-1, 1), y,
                                 n_neighbors=3, random_state=seed)[0]
        out[c] = mi
    return pd.Series(out, name="conditional_MI_residual")


# ---------------------------------------------------------------------------
# 5. Compositional candidates
# ---------------------------------------------------------------------------
def build_compositions(train: pd.DataFrame,
                       history: pd.DataFrame) -> pd.DataFrame:
    """Compute 3 candidate compositional banks for every training patient.

    All inputs handled with defensive defaults; outputs are signed/positive
    numeric scores (not ESI estimates) — the purpose here is to measure
    information content against the target, not to be directly emittable.
    """
    m = train.merge(history, on="patient_id", how="left",
                    suffixes=("", "_dup"))
    m = m[[c for c in m.columns if not c.endswith("_dup")]]

    age = m["age"].astype(float).clip(lower=1)
    gcs = m["gcs_total"].fillna(15).astype(float).clip(lower=3, upper=15)
    si = m["shock_index"].astype(float)  # HR/SBP
    news2 = m["news2_score"].astype(float)
    temp = m["temperature_c"].astype(float)
    meds = m["num_active_medications"].fillna(0).astype(float)
    n_co = m["num_comorbidities"].fillna(0).astype(float).clip(lower=0)
    immuno = (m.get("hx_immunosuppressed", pd.Series(0, index=m.index))
              .fillna(0).astype(int))

    # 1. AgeNormalizedShock = shock_index × (age/40)^0.5 × (1 + (15-gcs)/15)
    gcs_deficit = 15 - gcs
    ans = si * (age / 40.0) ** 0.5 * (1.0 + gcs_deficit / 15.0)

    # 2. ChronicBurdenResidual = NEWS2 − E[NEWS2 | comorbidity_count, age_group]
    cond_mean = m.groupby(["age_group",
                           pd.cut(n_co, bins=[-0.5, 0.5, 2.5, 4.5, 10.5],
                                  labels=["0", "1-2", "3-4", "5+"])],
                          observed=True)["news2_score"].transform("mean")
    cbr = news2 - cond_mean

    # 3. ThermoImmuneProduct = (temp − 37)² × (1 + 2·immuno) × max(0, log(1+meds−3))
    temp_sq = (temp - 37.0) ** 2
    poly = np.maximum(0, np.log1p(np.clip(meds - 3, a_min=0, a_max=None)))
    tip = temp_sq * (1.0 + 2.0 * immuno) * poly

    return pd.DataFrame({
        "age_normalized_shock": ans,
        "chronic_burden_residual": cbr,
        "thermo_immune_product": tip,
        # Also the raw components, for comparison:
        "raw_shock_index": si,
        "raw_news2": news2,
        "raw_temp_dev": (temp - 37.0).abs(),
    }, index=m.index)


# ---------------------------------------------------------------------------
# 6. Kuramoto order parameter per patient
# ---------------------------------------------------------------------------
def kuramoto_summary(esi_m: pd.DataFrame,
                     conf_m: pd.DataFrame) -> dict:
    """Compute per-patient r (order param), phase distribution stats, subset r."""
    # Map ESI [1,5] → phase θ ∈ [0, π] (half-circle is natural for ordinal).
    # Full circle would collapse 1 and 5 to same phase.
    theta = (esi_m - 1.0) / 4.0 * np.pi  # (n, 11)

    # Weight by confidence
    w = conf_m.values  # (n, 11)
    # Only banks with conf > 0.05 contribute
    mask = w > 0.05
    w_eff = np.where(mask, w, 0.0)
    e_ith = np.exp(1j * theta.values) * w_eff
    denom = w_eff.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    z = e_ith.sum(axis=1) / denom.squeeze()
    r = np.abs(z)
    psi = np.angle(z)

    # Per-bank signed phase deviation (θ_i − ψ)
    dev = theta.values - psi.reshape(-1, 1)
    # Wrap to [−π, π]
    dev = np.mod(dev + np.pi, 2 * np.pi) - np.pi
    dev = np.where(mask, dev, 0.0)

    # Subset-r for clinically meaningful groupings
    groups = {
        "physiologic_core": ["severity", "consciousness", "respiratory",
                             "cardiovascular"],
        "chronic_profile": ["history", "demographic", "utilization"],
        "complaint_context": ["complaint", "pain", "thermal"],
        "arrival_only": ["arrival"],
    }
    subset_r = {}
    bank_cols = esi_m.columns.tolist()
    for gname, members in groups.items():
        idx = [bank_cols.index(m) for m in members if m in bank_cols]
        if not idx:
            continue
        sub_theta = theta.iloc[:, idx].values
        sub_w = w_eff[:, idx]
        sub_denom = sub_w.sum(axis=1, keepdims=True)
        sub_denom = np.where(sub_denom > 0, sub_denom, 1.0)
        sub_z = (np.exp(1j * sub_theta) * sub_w).sum(axis=1) / sub_denom.squeeze()
        subset_r[gname] = np.abs(sub_z)

    return {
        "r": r,
        "psi": psi,
        "deviations": pd.DataFrame(dev, index=esi_m.index,
                                    columns=[f"dev_{c}" for c in esi_m.columns]),
        "subset_r": subset_r,
    }


# ---------------------------------------------------------------------------
# 7. Clinician/site variance decomposition
# ---------------------------------------------------------------------------
def clinician_variance(train: pd.DataFrame) -> dict:
    """Per-nurse / per-site empirical ESI distribution and deviation from pop."""
    pop_dist = train.triage_acuity.value_counts(normalize=True).sort_index()

    nurse = (train.groupby("triage_nurse_id").triage_acuity
             .value_counts(normalize=True).unstack(fill_value=0)
             .reindex(columns=[1, 2, 3, 4, 5], fill_value=0))
    nurse_counts = train.groupby("triage_nurse_id").size()

    site = (train.groupby("site_id").triage_acuity
            .value_counts(normalize=True).unstack(fill_value=0)
            .reindex(columns=[1, 2, 3, 4, 5], fill_value=0))
    site_counts = train.groupby("site_id").size()

    # L1 distance from population distribution
    nurse_dev = (nurse - pop_dist.values).abs().sum(axis=1)
    site_dev = (site - pop_dist.values).abs().sum(axis=1)

    return {
        "population": pop_dist.to_dict(),
        "n_nurses": int(len(nurse)),
        "n_sites": int(len(site)),
        "nurse_count_stats": nurse_counts.describe().to_dict(),
        "site_count_stats": site_counts.describe().to_dict(),
        "nurse_dev_stats": nurse_dev.describe().to_dict(),
        "site_dev_stats": site_dev.describe().to_dict(),
        "nurse_dev_top5": nurse_dev.nlargest(5).to_dict(),
        "site_dev_top5": site_dev.nlargest(5).to_dict(),
    }


# ---------------------------------------------------------------------------
# 8. Leakage-as-calibrator feasibility
# ---------------------------------------------------------------------------
def leakage_calibrator_check(train: pd.DataFrame,
                             bank_esi: pd.DataFrame,
                             bank_conf: pd.DataFrame) -> dict:
    """Does outcome severity track our coherence confidence as expected?"""
    weights = bank_conf.values
    # Use the highest individual bank confidence as a coherence proxy
    top_conf = weights.max(axis=1)

    aligned = train.set_index("patient_id").loc[bank_esi.index]
    disp = aligned["disposition"].fillna("unknown")
    los = aligned["ed_los_hours"].fillna(-1).values

    # Severe outcome = admitted / transferred / deceased / observation
    severe = disp.isin(["admitted", "transferred", "deceased",
                        "observation"]).values

    # Bin by top_conf quartile, compute severe-outcome rate + mean LOS
    bins = pd.qcut(pd.Series(top_conf), 4,
                   labels=["q1", "q2", "q3", "q4"], duplicates="drop")
    out = {}
    for label in ["q1", "q2", "q3", "q4"]:
        mask = (bins == label).to_numpy()
        if mask.sum() == 0:
            continue
        out[label] = {
            "n": int(mask.sum()),
            "conf_range": (float(top_conf[mask].min()),
                           float(top_conf[mask].max())),
            "severe_outcome_rate": float(severe[mask].mean()),
            "mean_los": float(los[mask][los[mask] >= 0].mean()
                              if (los[mask] >= 0).any() else np.nan),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("Loading data...")
    train = pd.read_csv(DATA / "train.csv")
    complaints = pd.read_csv(DATA / "chief_complaints.csv")
    history = pd.read_csv(DATA / "patient_history.csv")
    y = train.triage_acuity.values

    print(f"[t={time.time()-t0:.0f}s] Building bank signal matrix for {len(train)} patients...")
    esi_m, conf_m = build_bank_matrix(train, complaints, history)
    # Align training data to bank matrix index
    train_idx = train.set_index("patient_id").loc[esi_m.index].reset_index()
    y = train_idx.triage_acuity.values

    print(f"[t={time.time()-t0:.0f}s] Section 2: correlations across 11 banks...")
    pearson = esi_m.corr().round(4)
    pcorr = partial_corr(esi_m).round(4)

    print(f"[t={time.time()-t0:.0f}s] Section 3-4: mutual information...")
    mi_bank = mi_with_target(esi_m, y, sample=20000).round(4)
    mi_cond = conditional_mi_all_but_one(esi_m, y, sample=20000).round(4)

    # Bank + confidence joint features: MI of (esi, conf) pair per bank
    joint_X = pd.concat([esi_m.add_suffix("_esi"),
                         conf_m.add_suffix("_conf")], axis=1)
    mi_joint = mi_with_target(joint_X, y, sample=20000).round(4)

    print(f"[t={time.time()-t0:.0f}s] Section 5: compositional candidates...")
    comp = build_compositions(train_idx, history)
    mi_comp = mi_with_target(comp, y, sample=20000).round(4)

    # MI of composition vs MI of its best component
    comp_vs_component = {
        "age_normalized_shock": {
            "composite": float(mi_comp["age_normalized_shock"]),
            "best_component": float(mi_comp["raw_shock_index"]),
            "lift": float(mi_comp["age_normalized_shock"] - mi_comp["raw_shock_index"]),
        },
        "chronic_burden_residual": {
            "composite": float(mi_comp["chronic_burden_residual"]),
            "raw_news2": float(mi_comp["raw_news2"]),
            "lift": float(mi_comp["chronic_burden_residual"] - mi_comp["raw_news2"]),
        },
        "thermo_immune_product": {
            "composite": float(mi_comp["thermo_immune_product"]),
            "raw_temp_dev": float(mi_comp["raw_temp_dev"]),
            "lift": float(mi_comp["thermo_immune_product"] - mi_comp["raw_temp_dev"]),
        },
    }

    print(f"[t={time.time()-t0:.0f}s] Section 6: Kuramoto order parameter...")
    kura = kuramoto_summary(esi_m, conf_m)
    r_stats = pd.Series(kura["r"]).describe().to_dict()
    subset_r_stats = {k: pd.Series(v).describe().to_dict()
                      for k, v in kura["subset_r"].items()}

    # Does r correlate with correctness? Use "distance from label" on weighted ESI as proxy.
    pred_rough = ((esi_m * conf_m).sum(axis=1) / conf_m.sum(axis=1)).values
    pred_round = np.clip(np.round(pred_rough), 1, 5).astype(int)
    correct = (pred_round == y).astype(int)
    from scipy.stats import pointbiserialr
    r_corr = pointbiserialr(kura["r"], correct)
    r_correctness = {"point_biserial_r": float(r_corr.statistic),
                     "pvalue": float(r_corr.pvalue)}

    # Per-bank deviation MI with target (for Kuramoto phase feature design)
    mi_dev = mi_with_target(kura["deviations"], y, sample=20000).round(4)

    print(f"[t={time.time()-t0:.0f}s] Section 7: clinician/site variance...")
    clin_var = clinician_variance(train)

    print(f"[t={time.time()-t0:.0f}s] Section 8: leakage-as-calibrator...")
    leak = leakage_calibrator_check(train, esi_m, conf_m)

    # Assemble report
    report = {
        "meta": {
            "n_train": int(len(train)),
            "n_banks": int(esi_m.shape[1]),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "pearson_corr_bank_esi": pearson.to_dict(),
        "partial_corr_bank_esi": pcorr.to_dict(),
        "mi_bank_esi_vs_target": mi_bank.to_dict(),
        "conditional_mi_bank_esi": mi_cond.to_dict(),
        "mi_bank_esi_plus_conf_vs_target": mi_joint.to_dict(),
        "composition_vs_components": comp_vs_component,
        "kuramoto_r_distribution": r_stats,
        "kuramoto_subset_r_distribution": subset_r_stats,
        "kuramoto_r_correctness_corr": r_correctness,
        "mi_phase_deviation_vs_target": mi_dev.to_dict(),
        "clinician_variance": clin_var,
        "leakage_calibrator_quartiles": leak,
    }

    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"[t={time.time()-t0:.0f}s] Wrote {OUT}")

    # Quick summary printout
    print("\n=== Quick summary ===")
    print(f"MI(bank_esi, target) top 5: {mi_bank.nlargest(5).to_dict()}")
    print(f"Conditional MI (unique info) top 5: {mi_cond.nlargest(5).to_dict()}")
    print(f"Kuramoto r: mean={r_stats['mean']:.3f} std={r_stats['std']:.3f}")
    print(f"r vs correctness: ρ={r_correctness['point_biserial_r']:.4f}")
    print(f"Composition lifts:")
    for k, v in comp_vs_component.items():
        print(f"  {k}: composite={v['composite']:.4f}, lift={v['lift']:+.4f}")
    print(f"n_nurses={clin_var['n_nurses']}, n_sites={clin_var['n_sites']}")


if __name__ == "__main__":
    main()
