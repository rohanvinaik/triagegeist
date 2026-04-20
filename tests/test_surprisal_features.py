"""Tests for src.surprisal_features — kill-targeted."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.banks import Bank, BankDecomposition, BankSignal
from src.surprisal_features import (
    SurprisalBaseline, build_surprisal_features, fit_surprisal_baseline,
)


def _sig(bank, esi, conf, evidence="x"):
    return BankSignal(bank=bank, esi_estimate=esi, confidence=conf,
                      esi_floor=0, esi_ceiling=6, evidence=evidence)


def _decomp(pid: str, entries: list[tuple[Bank, float, float]]) -> BankDecomposition:
    d = BankDecomposition(patient_id=pid)
    for bank, esi, conf in entries:
        d.add(_sig(bank, esi, conf))
    return d


# ---------- SurprisalBaseline.tail_prob / surprisal ----------

def test_tail_prob_value_empty_baseline_returns_one():
    b = SurprisalBaseline()
    assert b.tail_prob(Bank.SEVERITY, 0.5) == 1.0
    assert b.surprisal(Bank.SEVERITY, 0.5) == 0.0


def test_tail_prob_value_observed_below_all_is_one():
    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    b = SurprisalBaseline(
        bank_conf_sorted={Bank.SEVERITY: arr},
        bank_count={Bank.SEVERITY: 5},
    )
    # observed=0.05 → all 5 baseline values >= 0.05 → tail prob = 5/5 = 1.0
    assert b.tail_prob(Bank.SEVERITY, 0.05) == pytest.approx(1.0)
    assert b.surprisal(Bank.SEVERITY, 0.05) == pytest.approx(0.0)


def test_tail_prob_value_observed_above_all():
    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    b = SurprisalBaseline(
        bank_conf_sorted={Bank.SEVERITY: arr},
        bank_count={Bank.SEVERITY: 5},
    )
    # observed=0.9 → 0 baseline samples in right tail → tail floor = 1/5 = 0.2
    assert b.tail_prob(Bank.SEVERITY, 0.9) == pytest.approx(0.2)
    # Surprisal = -log(0.2) ≈ 1.609
    assert b.surprisal(Bank.SEVERITY, 0.9) == pytest.approx(-np.log(0.2))


def test_tail_prob_value_observed_at_median():
    arr = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    b = SurprisalBaseline(
        bank_conf_sorted={Bank.SEVERITY: arr},
        bank_count={Bank.SEVERITY: 5},
    )
    # observed=0.3: 3 baseline samples are >= 0.3 (0.3, 0.4, 0.5) → tail = 3/5
    assert b.tail_prob(Bank.SEVERITY, 0.3) == pytest.approx(0.6)


def test_surprisal_monotonic_in_observed_conf():
    arr = np.linspace(0.0, 1.0, 100)
    b = SurprisalBaseline(
        bank_conf_sorted={Bank.SEVERITY: arr},
        bank_count={Bank.SEVERITY: 100},
    )
    # Higher observed conf → lower tail prob → higher surprisal (monotone up)
    s_low = b.surprisal(Bank.SEVERITY, 0.1)
    s_mid = b.surprisal(Bank.SEVERITY, 0.5)
    s_high = b.surprisal(Bank.SEVERITY, 0.95)
    assert s_low <= s_mid <= s_high


# ---------- fit_surprisal_baseline ----------

def test_fit_baseline_value_one_bank():
    decomps = [
        _decomp("P0", [(Bank.SEVERITY, 3.0, 0.2)]),
        _decomp("P1", [(Bank.SEVERITY, 3.0, 0.5)]),
        _decomp("P2", [(Bank.SEVERITY, 3.0, 0.8)]),
    ]
    b = fit_surprisal_baseline(decomps)
    assert b.bank_count[Bank.SEVERITY] == 3
    np.testing.assert_allclose(
        b.bank_conf_sorted[Bank.SEVERITY], [0.2, 0.5, 0.8],
    )


def test_fit_baseline_sparse_channel_lower_count():
    """Banks absent from some decomps have lower count."""
    decomps = [
        _decomp("P0", [(Bank.SEVERITY, 3, 0.5), (Bank.ARRIVAL, 3, 0.05)]),
        _decomp("P1", [(Bank.SEVERITY, 3, 0.5)]),
        _decomp("P2", [(Bank.SEVERITY, 3, 0.5)]),
    ]
    b = fit_surprisal_baseline(decomps)
    assert b.bank_count[Bank.SEVERITY] == 3
    assert b.bank_count[Bank.ARRIVAL] == 1


# ---------- build_surprisal_features (integration) ----------

@pytest.fixture
def mini_decomps():
    """Three decompositions + a baseline fitted on them."""
    decomps = [
        _decomp("P0", [
            (Bank.SEVERITY, 2.0, 0.8),
            (Bank.CONSCIOUSNESS, 1.5, 0.9),
            (Bank.ARRIVAL, 3.0, 0.05),  # normally silent
        ]),
        _decomp("P1", [
            (Bank.SEVERITY, 3.5, 0.5),
            (Bank.CONSCIOUSNESS, 3.0, 0.3),
            (Bank.ARRIVAL, 3.0, 0.05),
        ]),
        _decomp("P2", [
            (Bank.SEVERITY, 4.0, 0.4),
            (Bank.CONSCIOUSNESS, 3.8, 0.2),
            (Bank.ARRIVAL, 5.0, 0.90),   # anomalously LOUD arrival
        ]),
    ]
    return decomps


def test_build_features_value_shape_and_columns(mini_decomps):
    baseline = fit_surprisal_baseline(mini_decomps)
    feats = build_surprisal_features(mini_decomps, baseline)
    assert len(feats) == 3
    # Must have per-bank surprisal + pair dissonance + scalar summaries
    assert "bank_severity_surprisal" in feats.columns
    assert "bank_arrival_surprisal" in feats.columns
    assert "dissonance_severity_arrival" in feats.columns
    assert "total_surprise_weighted_dissent" in feats.columns
    assert "dominant_dissenter_code" in feats.columns
    assert "soft_consensus_esi" in feats.columns


def test_build_features_logical_loud_rare_bank_has_high_surprisal(mini_decomps):
    """P2's ARRIVAL at conf=0.90 is rare given baseline (two at 0.05) →
    its surprisal should be HIGHER than ARRIVAL surprisal for P0 (conf=0.05)."""
    baseline = fit_surprisal_baseline(mini_decomps)
    feats = build_surprisal_features(mini_decomps, baseline)
    s_p0 = feats.loc["P0", "bank_arrival_surprisal"]
    s_p2 = feats.loc["P2", "bank_arrival_surprisal"]
    assert s_p2 > s_p0


def test_build_features_value_dissonance_nonneg(mini_decomps):
    baseline = fit_surprisal_baseline(mini_decomps)
    feats = build_surprisal_features(mini_decomps, baseline)
    diss_cols = [c for c in feats.columns if c.startswith("dissonance_")]
    assert all((feats[c] >= 0).all() for c in diss_cols)


def test_build_features_value_soft_consensus_in_range(mini_decomps):
    baseline = fit_surprisal_baseline(mini_decomps)
    feats = build_surprisal_features(mini_decomps, baseline)
    consensus = feats["soft_consensus_esi"]
    assert (consensus >= 1.0).all() and (consensus <= 5.0).all()


def test_build_features_value_dominant_dissenter_in_enum_range(mini_decomps):
    baseline = fit_surprisal_baseline(mini_decomps)
    feats = build_surprisal_features(mini_decomps, baseline)
    code = feats["dominant_dissenter_code"]
    # 0 = no dissenter, 1-11 = Bank enum position
    assert ((code >= 0) & (code <= 11)).all()


def test_build_features_swap_patient_order_preserves_per_patient_values():
    """Reordering input decomps shouldn't change per-patient feature values."""
    ds = [
        _decomp("P0", [(Bank.SEVERITY, 3, 0.5), (Bank.THERMAL, 2, 0.8)]),
        _decomp("P1", [(Bank.SEVERITY, 4, 0.3), (Bank.THERMAL, 5, 0.1)]),
    ]
    baseline = fit_surprisal_baseline(ds)
    feats_normal = build_surprisal_features(ds, baseline)
    feats_reversed = build_surprisal_features(ds[::-1], baseline)
    pd.testing.assert_series_equal(
        feats_normal.loc["P0"],
        feats_reversed.loc["P0"],
    )


def test_build_features_boundary_empty_decomp():
    """A decomposition with no signals should emit zero-filled features, not crash."""
    empty = BankDecomposition(patient_id="P_empty")
    baseline = fit_surprisal_baseline([])
    feats = build_surprisal_features([empty], baseline)
    assert len(feats) == 1
    assert feats.loc["P_empty", "total_surprise_weighted_dissent"] == 0.0


# ---------- subset r ratio sanity ----------

def test_build_features_value_subset_r_ratios_finite(mini_decomps):
    baseline = fit_surprisal_baseline(mini_decomps)
    feats = build_surprisal_features(mini_decomps, baseline)
    for col in ["r_core_over_chronic", "r_core_over_complaint",
                "r_chronic_over_complaint"]:
        assert feats[col].notna().all()
        assert np.all(np.isfinite(feats[col].values))
