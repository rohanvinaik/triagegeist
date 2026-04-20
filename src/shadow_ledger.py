"""
Shadow Ledger for Triagegeist — pattern ported from harmonizing/src/shadow_ledger.py.

The pattern: everything extracted is stored in a shadow container that is
intentionally richer than the final output. Extraction maximizes recall;
emission (via a projector) maximizes precision. The final prediction is a
projection from the shadow ledger; different projectors can emit
differently (e.g. a conservative clinician-facing projection vs an
aggressive model-feed projection) without re-running the extraction layer.

For triagegeist, each bank emits ONE primary ESI estimate under the
current single-strategy imputation. The shadow ledger records alternative
estimates a bank might emit under different impute strategies:

  - observed: the bank's primary output (current pipeline default).
  - missingness_as_signal: what the bank would emit if missingness is
    treated as informative (e.g. `bp_missing` patients get a low-acuity
    prior from arrival + demographic).
  - population_conditional: what the bank would emit if we impute missing
    values from the (complaint_base, age_group) cohort mean.
  - trajectory_adjusted: what the bank would emit after applying the
    temporal-bank trajectory-marker adjustment.

In the current pipeline, only the primary estimate is emitted. The
ShadowLedger exists so future work can add projectors that select the
right strategy per patient subset, or feed multiple alternative estimates
as features. It's also the substrate for richer LLM-context rendering:
instead of "consciousness bank says ESI 3", the LLM can see "consciousness
bank says ESI 3 under observed, ESI 2 under missingness-as-signal, ESI 4
under trajectory-adjusted" and reason about the uncertainty.

This is a structural addition rather than a metric-lift one — at shipping
time we project through the observed strategy, identical to the current
pipeline. The payoff is in the writeup narrative and in a future pass
that trains a projector.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from .banks import Bank, BankDecomposition, BankSignal


class ImputeStrategy(str, Enum):
    """Alternative impute strategies a bank may use."""
    OBSERVED = "observed"                       # primary / current default
    MISSINGNESS_AS_SIGNAL = "missingness_as_signal"
    POPULATION_CONDITIONAL = "population_conditional"
    TRAJECTORY_ADJUSTED = "trajectory_adjusted"


class EvidenceScope(str, Enum):
    """Whether this evidence is safe to project into the final prediction."""
    EMITTABLE = "emittable"              # safe for primary projection
    CONTEXT_ONLY = "context_only"        # LLM / audit-trail only
    DIAGNOSTIC = "diagnostic"            # pipeline-state / not patient-facing


@dataclass(frozen=True)
class ShadowSignal:
    """A single (bank, strategy) signal with full provenance."""
    bank: Bank
    strategy: ImputeStrategy
    esi_estimate: float
    confidence: float
    evidence: str
    scope: EvidenceScope = EvidenceScope.EMITTABLE


@dataclass
class ShadowLedger:
    """Per-patient shadow container. Append-only within a pipeline run."""
    patient_id: str
    signals: list[ShadowSignal] = field(default_factory=list)
    _by_bank: dict[Bank, list[ShadowSignal]] = field(default_factory=lambda: defaultdict(list))

    def add(self, signal: ShadowSignal) -> None:
        self.signals.append(signal)
        self._by_bank[signal.bank].append(signal)

    def by_bank(self, bank: Bank) -> list[ShadowSignal]:
        """All strategies' signals for a given bank, in insertion order."""
        return list(self._by_bank.get(bank, []))

    def primary(self, bank: Bank) -> ShadowSignal | None:
        """The current-pipeline-default (OBSERVED) signal for a bank."""
        for s in self._by_bank.get(bank, []):
            if s.strategy == ImputeStrategy.OBSERVED:
                return s
        return None

    def to_primary_decomposition(self) -> BankDecomposition:
        """Project the shadow to a BankDecomposition using OBSERVED strategy.

        This is the identity projection — it reconstructs what the current
        pipeline would have emitted. Future projectors can select different
        strategies per bank or per patient subset.
        """
        d = BankDecomposition(patient_id=self.patient_id)
        for bank, sigs in self._by_bank.items():
            primary = next((s for s in sigs
                            if s.strategy == ImputeStrategy.OBSERVED), None)
            if primary is not None:
                d.add(BankSignal(
                    bank=bank,
                    esi_estimate=primary.esi_estimate,
                    confidence=primary.confidence,
                    esi_floor=0,
                    esi_ceiling=6,
                    evidence=primary.evidence,
                ))
        return d

    def alternative_count(self) -> int:
        """Number of alternative strategies stored beyond OBSERVED."""
        return sum(1 for s in self.signals
                   if s.strategy != ImputeStrategy.OBSERVED)


# ---------------------------------------------------------------------------
# Lift a BankDecomposition into a ShadowLedger
# ---------------------------------------------------------------------------

def lift_decomposition(decomp: BankDecomposition) -> ShadowLedger:
    """Promote a BankDecomposition into a shadow ledger with OBSERVED signals.

    No alternative strategies populated yet — call `add_*_strategy` methods
    below to enrich the ledger. This is the identity-lift that future
    projectors build on.
    """
    ledger = ShadowLedger(patient_id=decomp.patient_id)
    for bank, sig in decomp.signals.items():
        ledger.add(ShadowSignal(
            bank=bank,
            strategy=ImputeStrategy.OBSERVED,
            esi_estimate=sig.esi_estimate,
            confidence=sig.confidence,
            evidence=sig.evidence,
        ))
    return ledger


def add_missingness_strategy(ledger: ShadowLedger,
                              row: dict) -> None:
    """Add MISSINGNESS_AS_SIGNAL variants for banks where missingness matters.

    Current heuristic (kept simple): if systolic_bp is missing, the
    cardiovascular and severity banks get a less-severe alternative
    estimate (empirically these patients are ESI 4-5, per the
    foundation-analysis missingness pattern).
    """
    sbp = row.get("systolic_bp")
    if sbp is None or (isinstance(sbp, float) and sbp != sbp):
        # Missing BP → low-acuity prior
        primary_cv = ledger.primary(Bank.CARDIOVASCULAR)
        if primary_cv is not None:
            alt_esi = max(primary_cv.esi_estimate, 4.0)
            ledger.add(ShadowSignal(
                bank=Bank.CARDIOVASCULAR,
                strategy=ImputeStrategy.MISSINGNESS_AS_SIGNAL,
                esi_estimate=alt_esi,
                confidence=0.30,
                evidence="bp_missing suggests low-acuity triage context",
                scope=EvidenceScope.CONTEXT_ONLY,
            ))


def add_population_conditional_strategy(
    ledger: ShadowLedger,  # noqa: ARG001
    row: dict,             # noqa: ARG001
    cohort_expectations=None,
) -> None:
    """Add POPULATION_CONDITIONAL variants — imputes own-value from cohort mean.

    Requires a cohort_expectations table keyed by (complaint_base, age_group).
    Currently stubbed — wire up when a fold-safe cohort table is available.
    """
    # Placeholder: real implementation would look up cohort_expectations
    # and emit an alt estimate. Kept as a structural hook for now.
    _ = cohort_expectations
    return None
