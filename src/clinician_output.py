"""
Clinician-facing output layer — the 2nd-order variance on every prediction.

This is the thing a triage nurse or emergency physician actually reads. Not a
single ESI integer but a structured record that answers:

  1. What did the system predict? (ESI + confidence)
  2. How confident should the clinician ACTUALLY be? (calibrated severe-
     outcome probability from the leakage-calibrator)
  3. Which clinical banks AGREED and which DISAGREED? (signed phase
     deviations — the Kuramoto "Dr. House" signature)
  4. Was this patient flagged by any of the atypical-pattern detectors?
     (temporal paradox, chronic marker, extreme bank disagreement)
  5. Would a different nurse at a different site likely have predicted
     differently? (clinician style offset)
  6. Can the clinician re-trace the reasoning? (full bank evidence)

The goal is that every prediction is a transparent, clinically-interpretable
artifact. A clinician should be able to read the output and decide whether to
accept the recommendation, escalate, or override — based on the disagreement
structure, not just a confidence scalar.

This module is the READER for the audit trail, not the generator. It
consumes the bank decomposition, model prediction, calibrated confidence,
and style features, and produces a structured ClinicianReport per patient.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .banks import BankDecomposition


ESI_LEVEL_MEANINGS = {
    1: "Resuscitation — immediate life-threat",
    2: "Emergent — high risk, severe pain, dangerous vitals",
    3: "Urgent — expects 2+ resources",
    4: "Less urgent — expects 1 resource",
    5: "Non-urgent — expects 0 resources",
}


@dataclass
class BankReading:
    """A single bank's verdict on the patient, human-readable."""
    bank: str                  # e.g. "severity"
    esi_estimate: float        # the bank's continuous ESI estimate
    confidence: float          # 0.0-1.0
    signed_deviation: float    # phase deviation from consensus (radians)
    agrees: bool               # |deviation| < π/8 ≈ same ESI bucket
    evidence: str              # human-readable rationale from the bank


@dataclass
class ClinicianReport:
    """Full 2nd-order report for one patient."""
    patient_id: str

    # First-order prediction
    esi_prediction: int
    esi_meaning: str
    model_confidence: float              # raw model top-class probability
    calibrated_severe_probability: float  # from leakage calibrator (if avail)

    # Banks — primary evidence
    readings: list[BankReading]

    # Disagreement signature
    order_parameter_r: float             # Kuramoto r across confident banks
    dissenting_banks: list[str]          # banks where |signed_deviation| > π/8
    dissent_direction: str               # "toward_severe" | "toward_mild" | "mixed" | "agreed"

    # Flags from atypical-pattern detectors
    flags: list[str] = field(default_factory=list)

    # Clinician-style context (optional)
    style_offset: dict | None = None     # {pop_mean_nurse_pred, this_nurse_over_bias, ...}

    # Decision method used by the pipeline
    method: str = "ensemble"             # "rules" | "coherence" | "ensemble" | "llm"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_llm_context(self) -> str:
        """Render report as a structured prompt fragment for Scale 3 LLM.

        The Tier-B signals (phase deviations, temporal flags, style offsets,
        calibrated severity) are not individually discriminative on synthetic
        data — their value is as *context enrichment* for the LLM when it
        disambiguates boundary cases. This method formats them in a way the
        LLM can reason over alongside the raw patient presentation.

        Designed to be concatenated into the existing boundary-disambiguation
        prompt (src/llm_residual.py), adding the 2nd-order picture without
        replacing the 1st-order clinical data.
        """
        lines = []
        lines.append(f"SYSTEM CONFIDENCE LANDSCAPE:")
        lines.append(
            f"  Bank-agreement r = {self.order_parameter_r:.2f} "
            f"(1.0 = all clinical axes concur, <0.7 = contested)"
        )
        lines.append(
            f"  Model top-class probability = {self.model_confidence:.2f}"
        )
        lines.append(
            f"  Historical severe-outcome rate for patients at this "
            f"confidence level = {self.calibrated_severe_probability:.1%}"
        )

        if self.dissenting_banks:
            lines.append(f"  Dissenting clinical axes: "
                         f"{', '.join(self.dissenting_banks)}")
            lines.append(f"  Direction of dissent: {self.dissent_direction}")
            # Emit magnitudes for top dissenting banks — validated as the
            # highest conditional-MI signal on the hard residual
            # (cheat_sheet_probe_v2: dev_consciousness gains +0.23 MI,
            # dev_pain +0.15, dev_respiratory +0.15 on the D_gap_ge_1 subset).
            dissent_readings = [
                r for r in self.readings if not r.agrees and r.confidence > 0.2
            ]
            dissent_readings.sort(
                key=lambda r: abs(r.signed_deviation), reverse=True,
            )
            if dissent_readings:
                lines.append(f"  Largest dissents (signed phase deviation, radians):")
                for r in dissent_readings[:5]:
                    direction = ("pulls toward more-severe" if r.signed_deviation < 0
                                 else "pulls toward less-severe")
                    lines.append(
                        f"    {r.bank}: dev={r.signed_deviation:+.2f} "
                        f"({direction}, its ESI call is {r.esi_estimate:.1f})"
                    )
        else:
            lines.append(f"  All clinical axes agree.")

        if self.flags:
            lines.append(f"\nATYPICAL PATTERNS DETECTED:")
            for f in self.flags:
                lines.append(f"  - {f}")

        if self.style_offset:
            s = self.style_offset
            over = s.get("nurse_over_bias", 0.0) or 0.0
            under = s.get("nurse_under_bias", 0.0) or 0.0
            if abs(over) > 0.01 or abs(under) > 0.01:
                lines.append(f"\nCLINICIAN STYLE CONTEXT:")
                if over > 0.01:
                    lines.append(
                        f"  Treating nurse historically assigns high-acuity "
                        f"(ESI 1-2) at {over:+.1%} above population average"
                    )
                if under > 0.01:
                    lines.append(
                        f"  Treating nurse historically assigns low-acuity "
                        f"(ESI 4-5) at {under:+.1%} above population average"
                    )

        lines.append(f"\nPER-BANK READINGS:")
        for r in self.readings:
            agree_marker = "+" if r.agrees else "!"
            lines.append(
                f"  [{agree_marker}] {r.bank:15s} ESI={r.esi_estimate:.1f} "
                f"conf={r.confidence:.2f} — {r.evidence}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _phase_from_esi(esi: float) -> float:
    """Map ESI ∈ [1,5] to phase ∈ [0, π] (half-circle avoids wrap-around)."""
    return (esi - 1.0) / 4.0 * np.pi


def _compute_r_and_psi(decomp: BankDecomposition) -> tuple[float, float]:
    """Kuramoto order parameter r and consensus phase ψ across confident banks."""
    thetas, weights = [], []
    for _, sig in decomp.signals.items():
        if sig.confidence > 0.05:
            thetas.append(_phase_from_esi(sig.esi_estimate))
            weights.append(sig.confidence)
    if not thetas:
        return 0.0, 0.0
    t = np.asarray(thetas)
    w = np.asarray(weights)
    z = (np.exp(1j * t) * w).sum() / w.sum()
    return float(np.abs(z)), float(np.angle(z))


def build_report(
    decomp: BankDecomposition,
    esi_prediction: int,
    model_confidence: float,
    method: str = "ensemble",
    *,
    calibrated_severe_prob: float | None = None,
    style_info: dict | None = None,
    temporal_features: dict | None = None,
    deviation_threshold: float = np.pi / 8,  # ≈ 22.5° = ≈ 1/4 of an ESI level
) -> ClinicianReport:
    """Assemble the full clinician-facing report for one patient.

    Arguments beyond the bank decomposition and prediction are optional —
    the report renders whatever is supplied. `style_info` comes from
    clinician_style.style_features_for_patients; `temporal_features` comes
    from temporal_bank.build_temporal_features; `calibrated_severe_prob`
    comes from clinician_style.calibrate().
    """
    r, psi = _compute_r_and_psi(decomp)

    readings: list[BankReading] = []
    dissenters: list[tuple[str, float]] = []
    for bank, sig in decomp.signals.items():
        theta = _phase_from_esi(sig.esi_estimate)
        dev = (theta - psi + np.pi) % (2 * np.pi) - np.pi if r > 0 else 0.0
        agrees = abs(dev) < deviation_threshold
        readings.append(BankReading(
            bank=bank.value,
            esi_estimate=round(sig.esi_estimate, 2),
            confidence=round(sig.confidence, 3),
            signed_deviation=round(dev, 3),
            agrees=bool(agrees),
            evidence=sig.evidence,
        ))
        if sig.confidence > 0.2 and not agrees:
            dissenters.append((bank.value, dev))

    # Classify dissent direction
    if not dissenters:
        dissent_dir = "agreed"
    else:
        signs = [np.sign(d) for _, d in dissenters]
        # Negative deviation = lower ESI = more severe (since low ESI = high acuity)
        # Positive deviation = higher ESI = less severe
        neg = sum(1 for s in signs if s < 0)
        pos = sum(1 for s in signs if s > 0)
        if neg and not pos:
            dissent_dir = "toward_severe"
        elif pos and not neg:
            dissent_dir = "toward_mild"
        else:
            dissent_dir = "mixed"

    flags: list[str] = []
    if temporal_features:
        if temporal_features.get("temporal_paradox_flag"):
            flags.append("temporal_paradox: severe-complaint cohort with low own NEWS2")
        if temporal_features.get("temporal_chronic"):
            flags.append("chronic_presentation_marker")
    if r < 0.7:
        flags.append(f"low_bank_coherence (r={r:.2f})")
    if model_confidence < 0.6:
        flags.append(f"low_model_confidence ({model_confidence:.2f})")
    if style_info and abs(style_info.get("nurse_over_bias", 0.0)) > 0.02:
        flags.append(
            f"treating_nurse_over_triage_bias={style_info['nurse_over_bias']:+.3f}"
        )

    return ClinicianReport(
        patient_id=decomp.patient_id,
        esi_prediction=int(esi_prediction),
        esi_meaning=ESI_LEVEL_MEANINGS.get(int(esi_prediction), "unknown"),
        model_confidence=round(model_confidence, 3),
        calibrated_severe_probability=(
            round(calibrated_severe_prob, 3)
            if calibrated_severe_prob is not None else 0.0
        ),
        readings=readings,
        order_parameter_r=round(r, 3),
        dissenting_banks=[b for b, _ in dissenters],
        dissent_direction=dissent_dir,
        flags=flags,
        style_offset=style_info,
        method=method,
    )


# ---------------------------------------------------------------------------
# Pretty-printing for writeup / clinician consumption
# ---------------------------------------------------------------------------

def render_report(report: ClinicianReport) -> str:
    """Format a ClinicianReport as a human-readable block."""
    lines = []
    lines.append(f"Patient {report.patient_id}: ESI {report.esi_prediction} "
                 f"({report.esi_meaning})")
    lines.append(f"  method: {report.method}  model_conf: {report.model_confidence:.3f}"
                 f"  severe_outcome_prob: {report.calibrated_severe_probability:.3f}")
    lines.append(f"  bank coherence r: {report.order_parameter_r:.3f}  "
                 f"dissent: {report.dissent_direction}")

    if report.dissenting_banks:
        lines.append(f"  dissenting banks: {', '.join(report.dissenting_banks)}")

    if report.flags:
        lines.append("  FLAGS:")
        for f in report.flags:
            lines.append(f"    - {f}")

    lines.append("  bank readings:")
    for r in report.readings:
        marker = "  " if r.agrees else "← "
        lines.append(f"    {marker}{r.bank:15s}  ESI={r.esi_estimate:.2f}  "
                     f"conf={r.confidence:.2f}  dev={r.signed_deviation:+.2f}  "
                     f"{r.evidence}")

    if report.style_offset and isinstance(report.style_offset, dict):
        s = report.style_offset
        over = s.get("nurse_over_bias", 0)
        under = s.get("nurse_under_bias", 0)
        lines.append(f"  treating nurse: over_bias={over:+.3f} "
                     f"under_bias={under:+.3f}")

    return "\n".join(lines)


def dump_reports(reports: list[ClinicianReport], path: Path) -> None:
    """Write reports as JSON array for downstream consumption (notebook, dashboard)."""
    path.write_text(json.dumps([r.to_dict() for r in reports], indent=2,
                                default=str))
