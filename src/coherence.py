"""
Cross-bank coherence engine and deterministic ESI triage rules.

Implements the NCEMS Kuramoto coherence pattern: when orthogonal
banks converge on the same ESI level, confidence increases nonlinearly.
When they diverge, confidence drops and the case is flagged for model.

Also implements hard ESI rules that override bank consensus for
clinically unambiguous cases (e.g., GCS ≤ 8 → ESI 1 always).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .banks import Bank, BankDecomposition, BankSignal


@dataclass(frozen=True)
class TriageDecision:
    """Final triage decision with provenance."""
    patient_id: str
    esi_prediction: int          # 1-5
    confidence: float            # 0.0-1.0
    method: str                  # "rules", "coherence", "model", "llm"
    evidence: list[str]          # human-readable decision trail


# ---------------------------------------------------------------------------
# Hard triage rules (override everything)
# ---------------------------------------------------------------------------

def _apply_hard_rules(decomp: BankDecomposition) -> TriageDecision | None:
    """Apply deterministic ESI rules for clinically unambiguous cases.

    These rules have near-100% clinical certainty and override
    all other signals. Returns None if no hard rule fires.
    """
    signals = decomp.signals
    evidence = []

    # Rule 1: GCS ≤ 8 → ESI 1 (comatose = resuscitation)
    # This is clinically unambiguous: GCS ≤ 8 requires immediate airway management
    cs = signals.get(Bank.CONSCIOUSNESS)
    if cs and cs.esi_estimate <= 1.0 and cs.confidence >= 0.95:
        evidence.append(f"HARD_RULE: {cs.evidence}")
        return TriageDecision(decomp.patient_id, 1, 0.99, "rules", evidence)

    # Rule 2: Critical complaint (cardiac arrest, respiratory arrest, etc.)
    # Only truly critical complaints with very high confidence
    cc = signals.get(Bank.COMPLAINT)
    if cc and cc.esi_estimate <= 1.0 and cc.confidence >= 0.95:
        evidence.append(f"HARD_RULE: {cc.evidence}")
        return TriageDecision(decomp.patient_id, 1, 0.96, "rules", evidence)

    # NOTE: Shock index, SpO2, temperature, and arrival mode are NOT hard rules.
    # They contribute as bank signals to coherence scoring but don't override
    # because the data shows they are not deterministic (e.g., high shock index
    # can be ESI 2, not just ESI 1).

    return None


# ---------------------------------------------------------------------------
# Kuramoto-style coherence scoring
# ---------------------------------------------------------------------------

def _compute_coherence(decomp: BankDecomposition) -> tuple[float, float, list[str]]:
    """Compute cross-bank coherence score and weighted ESI estimate.

    Returns:
        (weighted_esi, coherence_confidence, evidence_list)

    The coherence pattern from NCEMS: when multiple independent banks
    converge on the same ESI range, confidence increases nonlinearly
    (Kuramoto synchronization). When they diverge, confidence drops.
    """
    signals = [s for s in decomp.signals.values() if s.confidence > 0.05]
    if not signals:
        return 3.0, 0.0, ["no signals"]

    # Confidence-weighted ESI
    total_weight = sum(s.confidence for s in signals)
    weighted_esi = sum(s.esi_estimate * s.confidence for s in signals) / total_weight

    # Compute convergence: confidence-WEIGHTED variance
    # Prevents low-confidence outlier banks from inflating disagreement
    weighted_var = sum(
        s.confidence * (s.esi_estimate - weighted_esi) ** 2 for s in signals
    ) / total_weight
    std_dev = math.sqrt(weighted_var)

    # Kuramoto-style nonlinear boost
    # Perfect agreement (std=0) → max boost
    # High disagreement (std>1.5) → penalty
    if std_dev < 0.3:
        coherence_boost = 0.25   # strong convergence
    elif std_dev < 0.6:
        coherence_boost = 0.15   # moderate convergence
    elif std_dev < 1.0:
        coherence_boost = 0.05   # weak convergence
    elif std_dev < 1.5:
        coherence_boost = -0.05  # mild divergence
    else:
        coherence_boost = -0.15  # strong divergence

    # Base confidence from highest-confidence banks
    top_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)
    base_conf = top_signals[0].confidence * 0.5
    if len(top_signals) > 1:
        base_conf += top_signals[1].confidence * 0.3
    if len(top_signals) > 2:
        base_conf += top_signals[2].confidence * 0.2

    coherence_conf = max(0.05, min(0.95, base_conf + coherence_boost))

    # Apply hard floor/ceiling constraints
    floor = max((s.esi_floor for s in signals if s.esi_floor > 0), default=0)
    ceiling = min((s.esi_ceiling for s in signals if s.esi_ceiling < 6), default=6)

    if floor > 0:
        weighted_esi = max(weighted_esi, float(floor))
    if ceiling < 6:
        weighted_esi = min(weighted_esi, float(ceiling))

    evidence = [
        f"weighted_esi={weighted_esi:.2f}",
        f"std={std_dev:.2f}",
        f"boost={coherence_boost:+.2f}",
        f"n_banks={len(signals)}",
        f"floor={floor} ceil={ceiling}",
    ]

    return weighted_esi, coherence_conf, evidence


# ---------------------------------------------------------------------------
# Coherence-based interaction rules
# ---------------------------------------------------------------------------

def _apply_interaction_rules(decomp: BankDecomposition,
                             weighted_esi: float,
                             confidence: float) -> tuple[float, float, list[str]]:
    """Apply cross-bank interaction rules that encode clinical reasoning.

    These are domain-specific coherence patterns: combinations of bank
    signals that should escalate or de-escalate triage.
    """
    signals = decomp.signals
    adjustments = []

    sev = signals.get(Bank.SEVERITY)
    cs = signals.get(Bank.CONSCIOUSNESS)
    resp = signals.get(Bank.RESPIRATORY)
    cc = signals.get(Bank.COMPLAINT)
    hx = signals.get(Bank.HISTORY)
    thermal = signals.get(Bank.THERMAL)

    # Interaction 1: High NEWS2 + altered consciousness → strong ESI 1-2
    if (sev and sev.esi_estimate <= 2.0
            and cs and cs.esi_estimate <= 2.5):
        weighted_esi = min(weighted_esi, 2.0)
        confidence = max(confidence, 0.85)
        adjustments.append("sev+consciousness convergence → ESI≤2")

    # Interaction 2: Fever + immunosuppressed → sepsis risk → escalate
    if (thermal and thermal.esi_estimate <= 3.0
            and hx and "immunosuppressed" in hx.evidence):
        weighted_esi = min(weighted_esi, 2.5)
        confidence = max(confidence, 0.70)
        adjustments.append("fever+immunosuppressed → sepsis risk")

    # Interaction 3: Chest pain complaint + cardiovascular derangement → ESI 2
    cv = signals.get(Bank.CARDIOVASCULAR)
    if (cc and "chest_pain" in cc.evidence
            and cv and cv.esi_estimate <= 3.0):
        weighted_esi = min(weighted_esi, 2.0)
        confidence = max(confidence, 0.75)
        adjustments.append("chest_pain+cv_derangement → ESI≤2")

    # Interaction 4: Dyspnea complaint + respiratory derangement → ESI 2
    if (cc and "dyspnea" in cc.evidence
            and resp and resp.esi_estimate <= 3.0):
        weighted_esi = min(weighted_esi, 2.0)
        confidence = max(confidence, 0.75)
        adjustments.append("dyspnea+resp_derangement → ESI≤2")

    # Interaction 5: Coagulopathy + trauma/bleeding complaint → escalate
    if (hx and "coagulopathy" in hx.evidence
            and cc and any(k in cc.evidence for k in
                          ["hemorrhage", "bleeding", "trauma"])):
        weighted_esi = min(weighted_esi, 2.0)
        confidence = max(confidence, 0.70)
        adjustments.append("coagulopathy+bleeding → escalate")

    # Interaction 6: All banks normal + minor complaint → ESI 5
    all_normal = all(
        s.esi_estimate >= 4.0
        for b, s in signals.items()
        if b in (Bank.SEVERITY, Bank.CONSCIOUSNESS, Bank.RESPIRATORY,
                 Bank.CARDIOVASCULAR, Bank.THERMAL)
        and s.confidence > 0.2
    )
    if (all_normal and cc
            and cc.esi_estimate >= 4.5 and cc.confidence >= 0.40):
        weighted_esi = max(weighted_esi, 4.5)
        confidence = max(confidence, 0.75)
        adjustments.append("all_normal+minor_complaint → ESI≥4.5")

    return weighted_esi, confidence, adjustments


# ---------------------------------------------------------------------------
# Main triage function
# ---------------------------------------------------------------------------

def triage_patient(decomp: BankDecomposition) -> TriageDecision:
    """Produce a triage decision from bank decomposition.

    Multi-scale decision order:
    1. Hard rules (deterministic, highest priority)
    2. Coherence scoring (weighted bank consensus)
    3. Interaction rules (cross-bank clinical patterns)
    4. Returns decision with confidence for model gating
    """
    # Scale 0: Hard rules
    hard = _apply_hard_rules(decomp)
    if hard:
        return hard

    # Scale 1: Coherence scoring
    weighted_esi, coherence_conf, coherence_evidence = _compute_coherence(decomp)

    # Scale 2: Interaction rules
    adjusted_esi, adjusted_conf, interaction_evidence = _apply_interaction_rules(
        decomp, weighted_esi, coherence_conf
    )

    # Round to nearest integer ESI (1-5)
    esi = max(1, min(5, round(adjusted_esi)))

    evidence = coherence_evidence + interaction_evidence

    return TriageDecision(
        patient_id=decomp.patient_id,
        esi_prediction=esi,
        confidence=adjusted_conf,
        method="coherence",
        evidence=evidence,
    )
