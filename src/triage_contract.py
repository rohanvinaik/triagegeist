"""
Typed contract for the Scale-3 LLM residual — AST-style constrained I/O.

Design principles (ported from aimo3/src/contracts.py):
  1. Closed enums everywhere. No free-text dispatch on either input or
     output side. Unknown values fail certification rather than silently
     propagating.
  2. Separate provenance. The LLM's reasoning text is stored as an opaque
     string, never parsed. All pipeline-consumable decisions come from the
     typed fields.
  3. The LLM is narrowed to TWO TASKS ONLY:
     (a) Resolve a specified uncertainty (pick ESI from a closed pair) given
         the full Tier-A/Tier-B evidence package.
     (b) Attribute its choice to a closed vocabulary of clinical evidence
         categories (so the human auditor can verify the reasoning).
     The LLM does NOT produce free-form text that enters the submission.

This module defines:
  - TriagePacket: the typed input the LLM receives (rendered to prose by a
    deterministic template, like aimo3's cheat_sheet_renderer).
  - TriageDecision: the typed output the LLM must emit as JSON.
  - LLM_DECISION_JSON_SCHEMA: JSON schema enforcing the output shape.
  - The closed enums underpinning both.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Closed enums — the full vocabulary the LLM can use
# ---------------------------------------------------------------------------

class ESILevel(str, Enum):
    """ESI 1-5. Stored as str for clean JSON serialization."""
    E1 = "1"
    E2 = "2"
    E3 = "3"
    E4 = "4"
    E5 = "5"


class DissentDirection(str, Enum):
    """How bank dissent relates to the consensus ESI estimate."""
    AGREED = "agreed"
    TOWARD_SEVERE = "toward_severe"
    TOWARD_MILD = "toward_mild"
    MIXED = "mixed"


class PatientCategory(str, Enum):
    """Why this patient was routed to the LLM residual."""
    HARD_RULE_DETERMINISTIC = "hard_rule_deterministic"   # should not reach LLM
    HIGH_COHERENCE_ROUTINE = "high_coherence_routine"     # should not reach LLM
    BANK_DISSENT = "bank_dissent"                         # banks disagree
    TEMPORAL_PARADOX = "temporal_paradox"                 # severe-cohort + low NEWS2
    LOW_INFORMATION = "low_information"                   # banks fire weakly overall
    MODEL_UNCERTAINTY = "model_uncertainty"               # top-2 prob gap small
    CHRONIC_PRESENTATION = "chronic_presentation"


class DecisiveEvidence(str, Enum):
    """Closed vocabulary of clinical evidence the LLM may cite."""
    VITAL_CRITICAL = "vital_critical"                     # any critical vital
    NEWS2_ELEVATED = "news2_elevated"
    SHOCK_INDEX_HIGH = "shock_index_high"
    SPO2_DEPRESSED = "spo2_depressed"
    ALTERED_CONSCIOUSNESS = "altered_consciousness"       # GCS < 15 or non-alert
    MENTAL_STATUS_RISK = "mental_status_risk"             # agitated/confused/drowsy
    HEMODYNAMIC_INSTABILITY = "hemodynamic_instability"
    SEVERE_PAIN = "severe_pain"                           # pain ≥ 8
    PAIN_LOCATION_HIGH_RISK = "pain_location_high_risk"   # chest/head/abdomen
    CHEST_PAIN_CARDIAC = "chest_pain_cardiac"
    DYSPNEA_RESPIRATORY = "dyspnea_respiratory"
    NEURO_DEFICIT = "neuro_deficit"                       # focal or stroke-pattern
    FEVER_SIGNIFICANT = "fever_significant"
    HYPOTHERMIA = "hypothermia"
    IMMUNOCOMPROMISE_FEVER = "immunocompromise_fever"
    COAGULOPATHY_BLEEDING = "coagulopathy_bleeding"
    AGE_EXTREME_PEDIATRIC = "age_extreme_pediatric"
    AGE_EXTREME_GERIATRIC = "age_extreme_geriatric"
    PREGNANCY_COMPLICATION = "pregnancy_complication"
    TRAJECTORY_PARADOX = "trajectory_paradox"             # severe cohort, low vitals
    TRAJECTORY_CHRONIC = "trajectory_chronic"
    MEDICATION_MASKING = "medication_masking"             # polypharmacy masking fever
    LOW_COMPLEXITY_PRESENTATION = "low_complexity_presentation"
    PROPHYLACTIC_OR_ADMIN = "prophylactic_or_admin"       # refill, suture removal
    NO_DECISIVE_EVIDENCE = "no_decisive_evidence"         # LLM had to guess


class AlignmentChoice(str, Enum):
    """What the LLM aligned with."""
    CONSENSUS = "consensus"                               # bank-weighted consensus ESI
    DISSENT_SEVERITY = "dissent_severity"
    DISSENT_CONSCIOUSNESS = "dissent_consciousness"
    DISSENT_RESPIRATORY = "dissent_respiratory"
    DISSENT_CARDIOVASCULAR = "dissent_cardiovascular"
    DISSENT_THERMAL = "dissent_thermal"
    DISSENT_PAIN = "dissent_pain"
    DISSENT_COMPLAINT = "dissent_complaint"
    DISSENT_HISTORY = "dissent_history"
    DISSENT_DEMOGRAPHIC = "dissent_demographic"
    DISSENT_UTILIZATION = "dissent_utilization"
    DISSENT_ARRIVAL = "dissent_arrival"
    TIER_B_FLAG = "tier_b_flag"                           # e.g. paradox fired
    INDEPENDENT_JUDGMENT = "independent_judgment"         # LLM overrode both


# ---------------------------------------------------------------------------
# Typed input — what the LLM receives (as rendered prose, via template)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BankReading:
    bank: str                       # bank name (closed via Bank enum elsewhere)
    esi_estimate: float
    confidence: float
    signed_deviation: float
    agrees: bool
    evidence: str                   # short closed-vocab phrase from bank itself


@dataclass(frozen=True)
class PatientClinical:
    """Raw clinical data from the dataset."""
    age: int
    sex: str
    arrival_mode: str
    chief_complaint: str
    mental_status: str
    vitals: dict[str, float]            # hr, bp, rr, temp, spo2
    gcs: int
    pain: int
    news2: float
    num_comorbidities: int
    num_active_medications: int
    comorbidity_flags: list[str]        # list of active hx_* names


@dataclass(frozen=True)
class SymbolicVerdict:
    """Pipeline's bank-layer read on the patient."""
    bank_readings: list[BankReading]
    order_parameter_r: float
    dissent_direction: DissentDirection
    dissenting_banks: list[str]


@dataclass(frozen=True)
class TierBContext:
    """Cheat-sheet context derived from Tier-B signals."""
    patient_category: PatientCategory
    tier_b_flags: list[str]             # e.g. ["temporal_paradox", "chronic"]
    calibrated_severe_outcome_prob: float


@dataclass(frozen=True)
class TriagePacket:
    """Typed clinical presentation + Tier-B context the LLM reasons over."""
    patient_id: str
    clinical: PatientClinical
    symbolic: SymbolicVerdict
    tier_b: TierBContext
    candidate_esis: tuple[ESILevel, ESILevel]   # disambiguation task

    def esi_choices(self) -> tuple[int, int]:
        return int(self.candidate_esis[0].value), int(self.candidate_esis[1].value)


# ---------------------------------------------------------------------------
# Typed output — what the LLM must emit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriageDecision:
    """LLM's disambiguation output. All fields closed-vocab except reasoning_summary.

    The reasoning_summary is retained for audit trail but NEVER parsed. All
    pipeline-consumable decisions come from the typed fields above it.
    """
    esi_choice: ESILevel
    alignment: AlignmentChoice
    decisive_evidence: list[DecisiveEvidence]       # len 1-3
    reasoning_summary: str = ""                     # free text, audit only


# JSON schema that forces the LLM into this shape.
LLM_DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["esi_choice", "alignment", "decisive_evidence"],
    "properties": {
        "esi_choice": {
            "type": "string",
            "enum": [e.value for e in ESILevel],
            "description": "The chosen ESI level. Must match one of the two candidates.",
        },
        "alignment": {
            "type": "string",
            "enum": [a.value for a in AlignmentChoice],
            "description": "Which side of the symbolic verdict you align with.",
        },
        "decisive_evidence": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "string",
                "enum": [v.value for v in DecisiveEvidence],
            },
            "description": "Up to 3 closed-vocabulary evidence categories.",
        },
        "reasoning_summary": {
            "type": "string",
            "maxLength": 240,
            "description": "One-sentence rationale. Audit-only; never parsed.",
        },
    },
}


# ---------------------------------------------------------------------------
# Deterministic prompt renderer (ELIZA-style template, no LLM in the loop)
# ---------------------------------------------------------------------------

_ESI_DEFS: dict[int, str] = {
    1: "ESI 1 — Resuscitation. Immediate life threat (intubation, CPR, active hemorrhage).",
    2: "ESI 2 — Emergent. High risk, confused/lethargic, severe pain, or dangerous vitals.",
    3: "ESI 3 — Urgent. Stable but expects 2+ resources (labs AND imaging, IV meds, consult).",
    4: "ESI 4 — Less urgent. Stable vitals, expects 1 resource.",
    5: "ESI 5 — Non-urgent. Stable vitals, expects 0 resources (exam, reassurance, prescription).",
}


def render_prompt(packet: TriagePacket) -> str:
    """Render a TriagePacket into the deterministic prompt text for the LLM.

    No string interpolation of patient free text beyond chief_complaint. All
    other fields come from typed packet values. Output is byte-for-byte
    reproducible given the same packet.
    """
    c1, c2 = packet.esi_choices()
    esi_defs = "\n".join(f"  {_ESI_DEFS[c]}" for c in sorted({c1, c2}))
    cl = packet.clinical
    sym = packet.symbolic
    tb = packet.tier_b

    # Render bank readings with phase-deviation magnitudes (the cheat-sheet
    # winner per cheat_sheet_probe_v2). Dissenters first, with direction.
    dissent_lines, agree_lines = [], []
    for r in sym.bank_readings:
        line = (f"    {r.bank:16s} ESI={r.esi_estimate:.1f} conf={r.confidence:.2f} "
                f"dev={r.signed_deviation:+.2f} — {r.evidence}")
        (agree_lines if r.agrees else dissent_lines).append(line)

    flags_block = ""
    if tb.tier_b_flags:
        flags_block = "\nATYPICAL PATTERNS:\n  - " + "\n  - ".join(tb.tier_b_flags)

    vocab_hint = (
        "\n\nRespond as JSON conforming to the provided schema:\n"
        f"  esi_choice in {{{c1}, {c2}}}\n"
        f"  alignment in {sorted(a.value for a in AlignmentChoice)}\n"
        f"  decisive_evidence: up to 3 from {sorted(v.value for v in DecisiveEvidence)}\n"
        f"  reasoning_summary: <=240 chars (audit-only)"
    )

    hr = cl.vitals.get("heart_rate", "?")
    sbp = cl.vitals.get("systolic_bp", "?")
    dbp = cl.vitals.get("diastolic_bp", "?")
    rr = cl.vitals.get("respiratory_rate", "?")
    temp = cl.vitals.get("temperature_c", "?")
    spo2 = cl.vitals.get("spo2", "?")
    dissent_block = (
        "\n".join(dissent_lines) if dissent_lines else "  (none dissenting)"
    )
    agree_block = "\n".join(agree_lines) if agree_lines else ""
    return f"""You are an emergency-department triage decision auditor. A multi-scale
symbolic pipeline has flagged this case as uncertain between ESI {c1} and ESI {c2}.
Choose the correct ESI and attribute your decision to the closed vocabulary of
clinical evidence categories below.

PATIENT:
  Age {cl.age}, Sex {cl.sex}, Arrival {cl.arrival_mode}
  Chief complaint: {cl.chief_complaint}
  Mental status: {cl.mental_status}   GCS: {cl.gcs}   Pain: {cl.pain}/10
  Vitals: HR {hr}, BP {sbp}/{dbp}, RR {rr}, Temp {temp}C, SpO2 {spo2}%
  NEWS2: {cl.news2}   Comorbidities: {cl.num_comorbidities}   Meds: {cl.num_active_medications}
  Active history flags: {', '.join(cl.comorbidity_flags) or 'none'}

SYMBOLIC CONSENSUS:
  bank_coherence r = {sym.order_parameter_r:.2f}
  dissent_direction = {sym.dissent_direction.value}
  dissenting_banks = {sym.dissenting_banks or ['none']}
  patient_category = {tb.patient_category.value}
  calibrated_severe_outcome_prob = {tb.calibrated_severe_outcome_prob:.2f}
{flags_block}

BANK READINGS (dissenters first; dev = signed phase deviation from consensus):
{dissent_block}
{agree_block}

ESI DEFINITIONS:
{esi_defs}
{vocab_hint}
"""


# ---------------------------------------------------------------------------
# Parser for typed LLM output
# ---------------------------------------------------------------------------

def parse_decision(raw: str | dict,
                   packet: TriagePacket) -> TriageDecision | None:
    """Parse LLM output (JSON string or dict) into a TriageDecision.

    Returns None on any parse or enum violation. Downstream callers check
    for None and fall back to the ensemble prediction.
    """
    import json as _json
    if isinstance(raw, str):
        try:
            obj = _json.loads(raw)
        except _json.JSONDecodeError:
            return None
    elif isinstance(raw, dict):
        obj = raw
    else:
        return None

    try:
        esi = ESILevel(str(obj["esi_choice"]))
        alignment = AlignmentChoice(obj["alignment"])
        evidence_raw = obj.get("decisive_evidence", [])
        if not isinstance(evidence_raw, list) or not 1 <= len(evidence_raw) <= 3:
            return None
        evidence = [DecisiveEvidence(v) for v in evidence_raw]
        reasoning = str(obj.get("reasoning_summary", ""))[:240]
    except (KeyError, ValueError, TypeError):
        return None

    # Must be one of the candidate ESIs
    if esi not in packet.candidate_esis:
        return None

    return TriageDecision(
        esi_choice=esi,
        alignment=alignment,
        decisive_evidence=evidence,
        reasoning_summary=reasoning,
    )
