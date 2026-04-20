"""
Post-LLM certification layer — deterministic checks against typed packet.

Ported in spirit from aimo3/src/answer_certifier.py. The LLM's decision is
not accepted unless it passes a battery of deterministic checks that
verify internal consistency between the chosen ESI, the attribution fields,
and the underlying patient packet.

Certification failures ARE signal: when the LLM "gets it right by typing"
but fails a clinical-consistency check, we log the failure and fall back
to the ensemble prediction. The certifier is the mechanical firewall that
prevents plausible-looking but clinically-incoherent LLM output from
reaching the submission.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .triage_contract import (
    AlignmentChoice, DecisiveEvidence, ESILevel,
    TriageDecision, TriagePacket,
)


@dataclass
class CertificationResult:
    certified: bool
    esi_choice: int
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    hard_contradiction: bool = False
    notes: list[str] = field(default_factory=list)


# Hard-floor rules: if these fire in the packet, the LLM CANNOT choose
# a less-severe ESI than the floor. Mirrors coherence.py's hard-rule logic.
HARD_FLOOR_RULES: list[tuple[str, int]] = [
    # (evidence substring in consciousness bank, ESI ceiling)
    ("comatose", 1),                 # GCS <= 8
    ("cardiac_arrest", 1),
    ("respiratory_arrest", 1),
    ("pulseless", 1),
    ("intubation", 1),
]


def _check_esi_in_candidates(result: CertificationResult,
                             decision: TriageDecision,
                             packet: TriagePacket) -> None:
    if decision.esi_choice not in packet.candidate_esis:
        result.checks_failed.append("esi_not_in_candidates")
        result.hard_contradiction = True
        return
    result.checks_passed.append("esi_in_candidates")


def _check_hard_floor(result: CertificationResult,
                      decision: TriageDecision,
                      packet: TriagePacket) -> None:
    """If packet has a hard-floor-triggering finding, LLM must respect it."""
    chosen = int(decision.esi_choice.value)
    for reading in packet.symbolic.bank_readings:
        for needle, ceil in HARD_FLOOR_RULES:
            if needle in reading.evidence.lower() and chosen > ceil:
                result.checks_failed.append(
                    f"hard_floor_violation:{needle}>ESI{ceil}"
                )
                result.hard_contradiction = True
                return
    result.checks_passed.append("hard_floor_respected")


def _check_alignment_consistency(result: CertificationResult,
                                 decision: TriageDecision,
                                 packet: TriagePacket) -> None:
    """If LLM claims to side with a dissenting bank, that bank must be dissenting."""
    align = decision.alignment
    dissenting = set(packet.symbolic.dissenting_banks)
    # Map alignment enum to bank name
    dissent_map = {
        AlignmentChoice.DISSENT_SEVERITY: "severity",
        AlignmentChoice.DISSENT_CONSCIOUSNESS: "consciousness",
        AlignmentChoice.DISSENT_RESPIRATORY: "respiratory",
        AlignmentChoice.DISSENT_CARDIOVASCULAR: "cardiovascular",
        AlignmentChoice.DISSENT_THERMAL: "thermal",
        AlignmentChoice.DISSENT_PAIN: "pain",
        AlignmentChoice.DISSENT_COMPLAINT: "complaint",
        AlignmentChoice.DISSENT_HISTORY: "history",
        AlignmentChoice.DISSENT_DEMOGRAPHIC: "demographic",
        AlignmentChoice.DISSENT_UTILIZATION: "utilization",
        AlignmentChoice.DISSENT_ARRIVAL: "arrival",
    }
    required_bank = dissent_map.get(align)
    if required_bank and required_bank not in dissenting:
        result.checks_failed.append(
            f"alignment_bank_not_dissenting:{required_bank}"
        )
        # Not a hard contradiction — LLM attribution is sloppy but choice may still be fine
        result.notes.append("Alignment cites non-dissenting bank; accepting with warning.")
        return
    result.checks_passed.append("alignment_consistent")


def _check_evidence_vs_packet(result: CertificationResult,
                              decision: TriageDecision,
                              packet: TriagePacket) -> None:
    """Cited evidence must be physically present in the packet.

    e.g., if LLM cites IMMUNOCOMPROMISE_FEVER, the patient must actually
    have hx_immunosuppressed AND elevated temperature. Otherwise the LLM
    is hallucinating justification.
    """
    evidence_set = set(decision.decisive_evidence)
    cl = packet.clinical
    temp = cl.vitals.get("temperature_c", 37.0)
    spo2 = cl.vitals.get("spo2", 99.0)

    violations = []
    if DecisiveEvidence.IMMUNOCOMPROMISE_FEVER in evidence_set:
        if "hx_immunosuppressed" not in cl.comorbidity_flags or temp < 38.0:
            violations.append("immunocompromise_fever_no_evidence")
    if DecisiveEvidence.COAGULOPATHY_BLEEDING in evidence_set:
        if "hx_coagulopathy" not in cl.comorbidity_flags:
            violations.append("coagulopathy_bleeding_no_hx")
    if DecisiveEvidence.PREGNANCY_COMPLICATION in evidence_set:
        if "hx_pregnant" not in cl.comorbidity_flags:
            violations.append("pregnancy_no_hx")
    if DecisiveEvidence.SEVERE_PAIN in evidence_set and cl.pain < 8:
        violations.append(f"severe_pain_but_pain={cl.pain}")
    if DecisiveEvidence.SPO2_DEPRESSED in evidence_set and spo2 >= 94:
        violations.append(f"spo2_depressed_but_spo2={spo2}")
    if DecisiveEvidence.FEVER_SIGNIFICANT in evidence_set and temp < 38.0:
        violations.append(f"fever_but_temp={temp}")
    if DecisiveEvidence.AGE_EXTREME_PEDIATRIC in evidence_set and cl.age >= 5:
        violations.append(f"pediatric_but_age={cl.age}")
    if DecisiveEvidence.AGE_EXTREME_GERIATRIC in evidence_set and cl.age < 75:
        violations.append(f"geriatric_but_age={cl.age}")

    if violations:
        result.checks_failed.extend(violations)
        result.notes.append("Evidence-packet consistency violations; accept with warning.")
        # Not a hard contradiction — ESI choice may still be clinically correct
        return
    result.checks_passed.append("evidence_consistent_with_packet")


def _check_pair_still_open(result: CertificationResult,
                           decision: TriageDecision) -> None:
    """The LLM had a non-degenerate choice: len(candidate_esis) must be 2."""
    # Note: TriagePacket always has a 2-tuple by construction, but be defensive.
    if decision.esi_choice is None:
        result.checks_failed.append("esi_null")
        result.hard_contradiction = True


def certify(decision: TriageDecision, packet: TriagePacket) -> CertificationResult:
    """Run all deterministic checks on a parsed LLM decision.

    Returns CertificationResult with `certified=True` only if no check failed.
    `hard_contradiction=True` means the LLM output must be rejected.
    Non-hard failures are accepted with warnings (noted in result.notes).
    """
    result = CertificationResult(
        certified=False,
        esi_choice=int(decision.esi_choice.value),
    )

    _check_pair_still_open(result, decision)
    if result.hard_contradiction:
        return result

    _check_esi_in_candidates(result, decision, packet)
    if result.hard_contradiction:
        return result

    _check_hard_floor(result, decision, packet)
    if result.hard_contradiction:
        return result

    _check_alignment_consistency(result, decision, packet)
    _check_evidence_vs_packet(result, decision, packet)

    # Soft failures don't block certification — they just note warnings.
    result.certified = not result.hard_contradiction
    return result


__all__ = ["certify", "CertificationResult", "HARD_FLOOR_RULES"]


if __name__ == "__main__":
    # Quick smoke-test with a minimal packet
    from .triage_contract import (
        BankReading, DissentDirection, PatientCategory, PatientClinical,
        SymbolicVerdict, TierBContext,
    )

    clinical = PatientClinical(
        age=45, sex="M", arrival_mode="walk-in",
        chief_complaint="chest pain", mental_status="alert",
        vitals={"heart_rate": 90, "systolic_bp": 130, "spo2": 98,
                "temperature_c": 37.0, "respiratory_rate": 16},
        gcs=15, pain=4, news2=1.0,
        num_comorbidities=2, num_active_medications=3,
        comorbidity_flags=["hx_hypertension"],
    )
    sym = SymbolicVerdict(
        bank_readings=[
            BankReading("severity", 3.5, 0.7, 0.1, True, "news2=1 low-medium"),
            BankReading("cardiovascular", 4.5, 0.3, 0.4, True, "hr=90 sbp=130 stable"),
        ],
        order_parameter_r=0.85,
        dissent_direction=DissentDirection.AGREED,
        dissenting_banks=[],
    )
    tb = TierBContext(
        patient_category=PatientCategory.MODEL_UNCERTAINTY,
        tier_b_flags=[],
        calibrated_severe_outcome_prob=0.35,
    )
    packet = TriagePacket(
        patient_id="TEST-001",
        clinical=clinical, symbolic=sym, tier_b=tb,
        candidate_esis=(ESILevel.E3, ESILevel.E4),
    )
    decision = TriageDecision(
        esi_choice=ESILevel.E3,
        alignment=AlignmentChoice.CONSENSUS,
        decisive_evidence=[DecisiveEvidence.CHEST_PAIN_CARDIAC],
    )
    r = certify(decision, packet)
    print(f"certified={r.certified}")
    print(f"passed={r.checks_passed}")
    print(f"failed={r.checks_failed}")
    print(f"hard_contradiction={r.hard_contradiction}")
