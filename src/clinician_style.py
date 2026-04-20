"""
Clinician and site style encoding + confidence calibration from outcome data.

Two classes of Tier-B signal that the foundation analysis validated empirically:

1. **Clinician variance as separate signal** (2026-04-17 finding: 50 nurses, L1
   deviation from population ESI distribution mean 3.1%, max 6.0%, std 1.3%.
   Modest size but real — and critical for the clinician-facing writeup).
   Standard ML conflates patient variance with rater variance. We separate
   them: fit each nurse's empirical ESI distribution, expose per-patient
   features that encode "what this particular nurse tends to call."

2. **Leakage-as-confidence-calibrator** (2026-04-17 finding: severe-outcome
   rate is strictly monotonic in coherence-confidence quartiles: 20%, 39%,
   51%, 84%). `disposition` and `ed_los_hours` are post-triage outcomes —
   they can't enter features at inference time because they don't exist on
   test. But at TRAINING time they validate whether our confidence is
   well-calibrated: if we said 0.8-confidence and the patient was actually
   admitted + stayed 6h, the confidence was correct. We use them as a
   meta-signal about the training set, never as a feature.

The only functions here that touch disposition/ed_los_hours are training-time
calibrators; they emit *calibration curves*, not features. Features come from
`style_features_for_patient`, which uses patient_id to look up the treating
nurse's learned style vector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


ESI_LEVELS = [1, 2, 3, 4, 5]

# Outcomes that indicate the patient was actually severely ill
SEVERE_DISPOSITIONS = {"admitted", "transferred", "deceased", "observation"}


# ---------------------------------------------------------------------------
# Fit clinician + site style vectors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StyleBank:
    """Per-rater (nurse or site) ESI-assignment style.

    A style is a 5-dim probability simplex over ESI levels, plus derived scalars
    (over/under-triage tendencies, L1 distance from population mean) that are
    useful as features.
    """
    rater_id: str
    distribution: np.ndarray         # shape (5,), sums to 1
    n_seen: int                      # patient count
    l1_dev_from_pop: float           # L1 distance from population mean
    over_triage_bias: float          # P(ESI<=2) − pop P(ESI<=2), signed
    under_triage_bias: float         # P(ESI>=4) − pop P(ESI>=4), signed


def _style_bank(rater_id: str, dist: np.ndarray, pop: np.ndarray,
                n: int) -> StyleBank:
    return StyleBank(
        rater_id=rater_id,
        distribution=dist,
        n_seen=int(n),
        l1_dev_from_pop=float(np.abs(dist - pop).sum()),
        over_triage_bias=float(dist[:2].sum() - pop[:2].sum()),    # ESI 1-2
        under_triage_bias=float(dist[3:].sum() - pop[3:].sum()),   # ESI 4-5
    )


def fit_style_banks(train_df: pd.DataFrame,
                    rater_col: str,
                    smoothing: float = 20.0,
                    target_col: str = "triage_acuity",
                    ) -> tuple[dict[str, StyleBank], np.ndarray]:
    """Fit per-rater ESI-distribution style banks with Dirichlet smoothing.

    Args:
        train_df: training data with rater_col and target_col.
        rater_col: column identifying the rater (e.g. "triage_nurse_id").
        smoothing: pseudo-count per ESI level toward population mean. Prevents
            overfitting on sparse raters. Higher = more shrinkage.
        target_col: the triage_acuity target column.

    Returns:
        dict mapping rater_id → StyleBank, and the population distribution.
    """
    target = train_df[target_col].values
    pop_counts = np.array([np.sum(target == l) for l in ESI_LEVELS], dtype=float)
    pop_dist = pop_counts / pop_counts.sum()

    banks: dict[str, StyleBank] = {}
    for rater_id, grp in train_df.groupby(rater_col):
        counts = np.array([np.sum(grp[target_col] == l) for l in ESI_LEVELS],
                          dtype=float)
        # Dirichlet-smoothed posterior: shrink toward population
        smoothed = (counts + smoothing * pop_dist) / (counts.sum() + smoothing)
        banks[str(rater_id)] = _style_bank(str(rater_id), smoothed, pop_dist,
                                            int(grp.shape[0]))

    return banks, pop_dist


# ---------------------------------------------------------------------------
# Per-patient style features (safe — no target / no leakage)
# ---------------------------------------------------------------------------

def style_features_for_patients(patient_df: pd.DataFrame,
                                nurse_banks: dict[str, StyleBank],
                                site_banks: dict[str, StyleBank],
                                pop_dist: np.ndarray,
                                ) -> pd.DataFrame:
    """Build per-patient features from a patient's treating nurse + site.

    Uses only patient_id, triage_nurse_id, site_id to look up pre-fit styles.
    Never touches target or post-outcome columns. Safe for test-time use as
    long as styles were fit on a disjoint train set (or on the same train via
    k-fold, documented below).

    Features emitted (prefix `style_`):
      - nurse_l1_dev: L1 distance of this nurse's distribution from pop
      - nurse_over_bias, nurse_under_bias: signed over/under-triage tendencies
      - site_l1_dev, site_over_bias, site_under_bias: same for site
      - nurse_pop_entropy: entropy of nurse distribution (uncertainty of this rater)
      - combined_bias: nurse_over_bias + site_over_bias
      - nurse_expected_esi: E[ESI | nurse] (the expected call under population-mean patient)
    """
    n = len(patient_df)
    out = {
        "style_nurse_l1_dev": np.zeros(n),
        "style_nurse_over_bias": np.zeros(n),
        "style_nurse_under_bias": np.zeros(n),
        "style_nurse_entropy": np.zeros(n),
        "style_nurse_expected_esi": np.zeros(n),
        "style_site_l1_dev": np.zeros(n),
        "style_site_over_bias": np.zeros(n),
        "style_site_under_bias": np.zeros(n),
        "style_combined_over_bias": np.zeros(n),
    }

    # Default (for unseen raters): no deviation from population.
    pop_bank = _style_bank("_pop_", pop_dist, pop_dist, 0)
    esi_arr = np.asarray(ESI_LEVELS, dtype=float)

    for i, (_, row) in enumerate(patient_df.iterrows()):
        nurse_id = str(row.get("triage_nurse_id", ""))
        site_id = str(row.get("site_id", ""))
        nb = nurse_banks.get(nurse_id, pop_bank)
        sb = site_banks.get(site_id, pop_bank)

        out["style_nurse_l1_dev"][i] = nb.l1_dev_from_pop
        out["style_nurse_over_bias"][i] = nb.over_triage_bias
        out["style_nurse_under_bias"][i] = nb.under_triage_bias
        p = nb.distribution
        out["style_nurse_entropy"][i] = float(-(p * np.log(p + 1e-12)).sum())
        out["style_nurse_expected_esi"][i] = float((p * esi_arr).sum())
        out["style_site_l1_dev"][i] = sb.l1_dev_from_pop
        out["style_site_over_bias"][i] = sb.over_triage_bias
        out["style_site_under_bias"][i] = sb.under_triage_bias
        out["style_combined_over_bias"][i] = (
            nb.over_triage_bias + sb.over_triage_bias)

    return pd.DataFrame(out, index=patient_df.index)


# ---------------------------------------------------------------------------
# Confidence calibrator: use disposition + ed_los_hours at TRAINING time
# only, to calibrate predicted confidence against real outcome severity.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfidenceCalibrator:
    """Isotonic-style mapping from raw-confidence → severe-outcome probability.

    Fit on training data where we have BOTH (a) the pipeline's predicted
    confidence and (b) the actual disposition + ed_los_hours. The calibrator
    learns "when the pipeline said confidence c, the actual severe-outcome
    rate was p(c)." At test time we can report calibrated confidence without
    ever having touched the leakage fields as features.
    """
    bin_edges: np.ndarray            # quantile boundaries of training confidence
    severe_rate: np.ndarray          # empirical severe-outcome rate per bin
    mean_los: np.ndarray             # mean ED length-of-stay per bin
    n_per_bin: np.ndarray


def fit_confidence_calibrator(train_df: pd.DataFrame,
                              confidence: np.ndarray,
                              n_bins: int = 10) -> ConfidenceCalibrator:
    """Fit isotonic calibration from pipeline confidence → outcome severity.

    Args:
        train_df: training data indexed same as `confidence`, must have
            `disposition` and `ed_los_hours` columns.
        confidence: per-patient confidence from the coherence layer.

    Returns:
        ConfidenceCalibrator with bin edges and empirical severe-outcome rates.
        Can be applied at test time via `calibrate(confidence)` to map raw
        confidence to calibrated severe-outcome probability.
    """
    disp = train_df["disposition"].fillna("unknown").values
    los = train_df["ed_los_hours"].fillna(-1).values
    severe = np.isin(disp, list(SEVERE_DISPOSITIONS)).astype(float)

    edges = np.quantile(confidence, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Use unique edges to avoid duplicate-bin issues.
    edges = np.unique(edges)
    bin_idx = np.digitize(confidence, edges[1:-1])
    n_effective = len(edges) - 1

    severe_rate = np.zeros(n_effective)
    mean_los = np.zeros(n_effective)
    counts = np.zeros(n_effective, dtype=int)
    for b in range(n_effective):
        mask = bin_idx == b
        if not mask.any():
            continue
        counts[b] = int(mask.sum())
        severe_rate[b] = float(severe[mask].mean())
        valid_los = los[mask]
        valid_los = valid_los[valid_los >= 0]
        mean_los[b] = float(valid_los.mean()) if len(valid_los) else np.nan

    # Enforce monotonicity (pool adjacent violators — isotonic regression).
    for _ in range(len(severe_rate)):
        fixed = False
        for i in range(len(severe_rate) - 1):
            if severe_rate[i] > severe_rate[i + 1] and counts[i] > 0 and counts[i + 1] > 0:
                total = counts[i] + counts[i + 1]
                pooled = (severe_rate[i] * counts[i]
                          + severe_rate[i + 1] * counts[i + 1]) / total
                severe_rate[i] = pooled
                severe_rate[i + 1] = pooled
                fixed = True
        if not fixed:
            break

    return ConfidenceCalibrator(
        bin_edges=edges,
        severe_rate=severe_rate,
        mean_los=mean_los,
        n_per_bin=counts,
    )


def calibrate(cal: ConfidenceCalibrator, confidence: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator to new confidence values."""
    bin_idx = np.digitize(confidence, cal.bin_edges[1:-1])
    bin_idx = np.clip(bin_idx, 0, len(cal.severe_rate) - 1)
    return cal.severe_rate[bin_idx]


# ---------------------------------------------------------------------------
# Training-set label-noise detection: which labels look like undertriage?
# ---------------------------------------------------------------------------

def detect_undertriage(train_df: pd.DataFrame,
                       los_threshold: float = 6.0) -> pd.Series:
    """Heuristic: ESI 4/5 labels with severe outcome = suspected undertriage.

    At training time, if a patient was labeled ESI 4 but was admitted with
    ed_los_hours > threshold, the original triage decision likely
    under-estimated acuity. Flag such rows for downweighting during model
    training. This is not feature creation — just a sample-weight adjustment.

    Returns a boolean Series aligned to train_df.index (True = suspect).
    """
    disp = train_df["disposition"].fillna("unknown")
    los = train_df["ed_los_hours"].fillna(-1)
    acuity = train_df["triage_acuity"]

    suspect = (
        acuity.isin([4, 5]) &
        disp.isin(list(SEVERE_DISPOSITIONS)) &
        (los >= los_threshold)
    )
    return suspect


def recommend_sample_weights(train_df: pd.DataFrame,
                             suspect_weight: float = 0.5) -> np.ndarray:
    """Build sample-weight vector downweighting suspected-undertriage rows.

    Returns an array of length len(train_df) with 1.0 for non-suspect rows
    and suspect_weight for suspected-undertriage rows. Feed as sample_weight
    argument to GBM fit methods.
    """
    suspect = detect_undertriage(train_df)
    w = np.ones(len(train_df), dtype=float)
    w[suspect.values] = suspect_weight
    return w


# ---------------------------------------------------------------------------
# Fold-safe style fitting for training
# ---------------------------------------------------------------------------

def fit_fold_safe_style_features(train_df: pd.DataFrame,
                                 fold_indices: list[tuple[np.ndarray, np.ndarray]],
                                 smoothing: float = 20.0) -> pd.DataFrame:
    """Fit style features across K folds without target leakage.

    For each fold, fits style banks on the TRAIN partition only and applies
    them to the VALIDATION partition. This prevents a patient's own triage
    label from influencing their nurse's style vector when used as a feature.

    Args:
        train_df: must have triage_nurse_id, site_id, triage_acuity.
        fold_indices: list of (train_idx, val_idx) pairs.
        smoothing: Dirichlet smoothing (higher = more shrinkage to pop).

    Returns:
        DataFrame of OOF style features aligned to train_df.index.
    """
    out = pd.DataFrame(index=train_df.index)
    for tr, va in fold_indices:
        fold_train = train_df.iloc[tr]
        fold_val = train_df.iloc[va]
        nurse, pop = fit_style_banks(fold_train, "triage_nurse_id",
                                     smoothing=smoothing)
        site, _ = fit_style_banks(fold_train, "site_id", smoothing=smoothing)
        val_feats = style_features_for_patients(fold_val, nurse, site, pop)
        for col in val_feats.columns:
            out.loc[val_feats.index, col] = val_feats[col].values

    return out.fillna(0.0)
