"""Surprisal-basis feature engineering for the hard-bucket regime.

Theoretical frame: bank decompositions are information channels with biased
baseline activation distributions. A usually-silent bank firing loudly carries
more Bayesian information than a usually-loud bank firing at the same amplitude
(Monty Hall / complementary-sparsity). This module computes baseline-conditional
surprisal signals that directly target the hard bucket — patients where the
standard bank consensus fails and diagnosis must be reconstructed by composing
several low-density, surprising signals (the Dr. House mechanism).

Emitted feature families:
  - per-bank confidence surprisal         `bank_<name>_surprisal`           (11 feats)
  - surprise-weighted dissent per bank    `bank_<name>_weighted_dissent`    (11 feats)
  - bank-pair dissonance                  `dissonance_<a>_<b>`              (C(11,2)=55 feats)
  - amplitude dispersion                  `bank_esi_std`, `bank_esi_iqr`    (2 feats)
  - dominant dissenter identity           `dominant_dissenter_code`         (1 categorical)
  - scalar surprise-weighted dissent      `total_surprise_weighted_dissent` (1 feat)
  - subset-r ratios                       `r_core_over_chronic`, etc.       (3 feats)

All features are fold-safe: `fit_surprisal_baseline` learns the per-bank
confidence ECDFs from the TRAIN decompositions only; `build_surprisal_features`
applies those baselines to VAL/test decompositions.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .banks import Bank, BankDecomposition


# Canonical ordering for bank iteration — keeps feature column order stable.
_BANK_ORDER: tuple[Bank, ...] = (
    Bank.SEVERITY, Bank.CONSCIOUSNESS, Bank.RESPIRATORY, Bank.CARDIOVASCULAR,
    Bank.THERMAL, Bank.PAIN, Bank.COMPLAINT, Bank.HISTORY, Bank.DEMOGRAPHIC,
    Bank.UTILIZATION, Bank.ARRIVAL,
)

# Subset groupings (match feature_engine._build_bank_features)
_SUBSETS: dict[str, frozenset[Bank]] = {
    "physiologic_core": frozenset({
        Bank.SEVERITY, Bank.CONSCIOUSNESS, Bank.RESPIRATORY, Bank.CARDIOVASCULAR,
    }),
    "chronic_profile": frozenset({
        Bank.HISTORY, Bank.DEMOGRAPHIC, Bank.UTILIZATION,
    }),
    "complaint_context": frozenset({
        Bank.COMPLAINT, Bank.PAIN, Bank.THERMAL,
    }),
}

# Integer code for dominant-dissenter categorical (0 = none / all-agree)
_BANK_TO_CODE: dict[Bank, int] = {b: i + 1 for i, b in enumerate(_BANK_ORDER)}


@dataclass(frozen=True)
class SurprisalBaseline:
    """Per-bank confidence distributions fit on a training partition.

    For each bank, stores the sorted confidence values observed across train
    decompositions. The tail probability P(conf >= c | baseline) is computed
    via ECDF on this sorted array at apply time.
    """
    bank_conf_sorted: dict[Bank, np.ndarray] = field(default_factory=dict)
    bank_count: dict[Bank, int] = field(default_factory=dict)

    def tail_prob(self, bank: Bank, observed_conf: float) -> float:
        """P(C >= observed | baseline) via right-tail ECDF."""
        sorted_arr = self.bank_conf_sorted.get(bank)
        n = self.bank_count.get(bank, 0)
        if sorted_arr is None or n == 0:
            return 1.0
        # Number of baseline samples strictly less than observed → those are
        # NOT in the right tail. (n - idx_ge) / n is the right-tail prob.
        idx_ge = np.searchsorted(sorted_arr, observed_conf, side="left")
        tail = max(1, n - int(idx_ge)) / n   # floor at 1/n so -log is finite
        return float(tail)

    def surprisal(self, bank: Bank, observed_conf: float) -> float:
        """-log P(C >= observed | baseline). Higher = more surprising activation."""
        return float(-np.log(self.tail_prob(bank, observed_conf)))


def fit_surprisal_baseline(
    train_decomps: list[BankDecomposition],
) -> SurprisalBaseline:
    """Fit per-bank confidence baselines from a training partition.

    Pure function — no side effects, no target dependency. Safe for fold-local use.
    """
    per_bank: dict[Bank, list[float]] = {b: [] for b in _BANK_ORDER}
    for d in train_decomps:
        for bank in _BANK_ORDER:
            sig = d.signals.get(bank)
            if sig is not None:
                per_bank[bank].append(float(sig.confidence))
    sorted_arrs: dict[Bank, np.ndarray] = {}
    counts: dict[Bank, int] = {}
    for bank, vals in per_bank.items():
        arr = np.sort(np.asarray(vals, dtype=float)) if vals else np.zeros(0)
        sorted_arrs[bank] = arr
        counts[bank] = int(len(arr))
    return SurprisalBaseline(bank_conf_sorted=sorted_arrs, bank_count=counts)


def _weighted_consensus_esi(decomp: BankDecomposition) -> float:
    """Confidence-weighted mean ESI across banks with conf > 0.05.

    Scalar "soft consensus" that's stable even at low r (unlike Kuramoto ψ).
    """
    ws, es = [], []
    for bank in _BANK_ORDER:
        sig = decomp.signals.get(bank)
        if sig is not None and sig.confidence > 0.05:
            ws.append(sig.confidence)
            es.append(sig.esi_estimate)
    if not ws:
        return 3.0  # population mean fallback
    w = np.asarray(ws)
    return float((np.asarray(es) * w).sum() / w.sum())


def _subset_r(decomp: BankDecomposition, members: frozenset[Bank]) -> float:
    """Kuramoto order parameter restricted to a bank subset (conf>0.05)."""
    thetas, ws = [], []
    for bank in members:
        sig = decomp.signals.get(bank)
        if sig is not None and sig.confidence > 0.05:
            thetas.append((sig.esi_estimate - 1.0) / 4.0 * np.pi)
            ws.append(sig.confidence)
    if not thetas:
        return 0.0
    t = np.asarray(thetas, dtype=float)
    w = np.asarray(ws, dtype=float)
    z = (np.exp(1j * t) * w).sum() / w.sum()
    return float(np.abs(z))


def _build_one_row(decomp: BankDecomposition,
                    baseline: SurprisalBaseline) -> dict:
    """Compute the full set of surprisal-basis features for one patient."""
    row: dict = {"patient_id": decomp.patient_id}
    consensus_esi = _weighted_consensus_esi(decomp)

    # --- Per-bank surprisal + surprise-weighted dissent
    surprisals: dict[Bank, float] = {}
    weighted_dissents: dict[Bank, float] = {}
    confs: dict[Bank, float] = {}
    esis: dict[Bank, float] = {}
    for bank in _BANK_ORDER:
        sig = decomp.signals.get(bank)
        if sig is None:
            s = 0.0
            confs[bank] = 0.0
            esis[bank] = consensus_esi
            weighted_dissents[bank] = 0.0
        else:
            s = baseline.surprisal(bank, sig.confidence)
            confs[bank] = float(sig.confidence)
            esis[bank] = float(sig.esi_estimate)
            dissent_mag = abs(float(sig.esi_estimate) - consensus_esi)
            weighted_dissents[bank] = s * dissent_mag
        surprisals[bank] = s
        row[f"bank_{bank.value}_surprisal"] = s
        row[f"bank_{bank.value}_weighted_dissent"] = weighted_dissents[bank]

    # --- Bank-pair dissonance: |ESI_i - ESI_j| × conf_i × conf_j
    for a, b in itertools.combinations(_BANK_ORDER, 2):
        row[f"dissonance_{a.value}_{b.value}"] = (
            abs(esis[a] - esis[b]) * confs[a] * confs[b]
        )

    # --- Amplitude dispersion across confident banks
    confident_esis = [esis[b] for b in _BANK_ORDER if confs[b] > 0.05]
    confident_ws = np.asarray(
        [confs[b] for b in _BANK_ORDER if confs[b] > 0.05], dtype=float
    )
    if confident_esis:
        arr = np.asarray(confident_esis, dtype=float)
        if confident_ws.sum() > 0:
            mu = (arr * confident_ws).sum() / confident_ws.sum()
            var = (confident_ws * (arr - mu) ** 2).sum() / confident_ws.sum()
            row["bank_esi_std_weighted"] = float(np.sqrt(max(var, 0.0)))
        else:
            row["bank_esi_std_weighted"] = 0.0
        if len(arr) >= 2:
            row["bank_esi_iqr"] = float(
                np.quantile(arr, 0.75) - np.quantile(arr, 0.25)
            )
            row["bank_esi_range"] = float(arr.max() - arr.min())
        else:
            row["bank_esi_iqr"] = 0.0
            row["bank_esi_range"] = 0.0
    else:
        row["bank_esi_std_weighted"] = 0.0
        row["bank_esi_iqr"] = 0.0
        row["bank_esi_range"] = 0.0

    # --- Dominant dissenter: argmax of (surprisal × |ESI - consensus|)
    # Only considers banks with confidence above 0.05.
    candidates = [
        (weighted_dissents[b], b)
        for b in _BANK_ORDER if confs[b] > 0.05
    ]
    if candidates:
        top_val, top_bank = max(candidates, key=lambda x: x[0])
        if top_val > 0.0:
            row["dominant_dissenter_code"] = _BANK_TO_CODE[top_bank]
            row["dominant_dissenter_value"] = float(top_val)
        else:
            row["dominant_dissenter_code"] = 0
            row["dominant_dissenter_value"] = 0.0
    else:
        row["dominant_dissenter_code"] = 0
        row["dominant_dissenter_value"] = 0.0

    # --- Scalar: total surprise-weighted dissent across all banks
    row["total_surprise_weighted_dissent"] = float(
        sum(weighted_dissents.values())
    )

    # --- Subset-r ratios: physiologic_core / chronic_profile, etc.
    r_core = _subset_r(decomp, _SUBSETS["physiologic_core"])
    r_chronic = _subset_r(decomp, _SUBSETS["chronic_profile"])
    r_complaint = _subset_r(decomp, _SUBSETS["complaint_context"])
    eps = 1e-3
    row["r_core_over_chronic"] = r_core / (r_chronic + eps)
    row["r_core_over_complaint"] = r_core / (r_complaint + eps)
    row["r_chronic_over_complaint"] = r_chronic / (r_complaint + eps)

    # Consensus ESI itself — useful scalar handoff
    row["soft_consensus_esi"] = consensus_esi

    return row


def build_surprisal_features(
    decomps: list[BankDecomposition],
    baseline: SurprisalBaseline,
) -> pd.DataFrame:
    """Emit surprisal-basis features for each patient, indexed by patient_id."""
    rows = [_build_one_row(d, baseline) for d in decomps]
    df = pd.DataFrame(rows).set_index("patient_id")
    return df
