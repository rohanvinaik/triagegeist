"""
Chief complaint lexicon for ESI triage prediction.

Maps chief_complaint_raw text to clinical severity categories.
This is Scale 0 text processing — deterministic lexicon lookup
with pattern matching for modifiers.

The lexicon is built from emergency medicine domain knowledge,
NOT from training data (NCEMS anti-overfitting principle).
"""

from __future__ import annotations

import re

from .banks import Bank, BankSignal


# ---------------------------------------------------------------------------
# High-acuity complaint patterns (ESI 1-2)
# ---------------------------------------------------------------------------

_CRITICAL_PATTERNS: list[tuple[re.Pattern, float, float, str]] = [
    # (pattern, esi_estimate, confidence, label)
    (re.compile(r"\bcardiac arrest\b", re.I), 1.0, 0.98, "cardiac_arrest"),
    (re.compile(r"\brespiratory arrest\b", re.I), 1.0, 0.98, "respiratory_arrest"),
    (re.compile(r"\bstroke\b", re.I), 1.5, 0.85, "stroke"),
    (re.compile(r"\bseizure\b.*\bstatus\b|\bstatus epilepticus\b", re.I), 1.2, 0.90, "status_epilepticus"),
    (re.compile(r"\banaphylax", re.I), 1.2, 0.92, "anaphylaxis"),
    (re.compile(r"\bintubat", re.I), 1.0, 0.95, "intubation"),
    (re.compile(r"\bpulseless\b", re.I), 1.0, 0.98, "pulseless"),
    (re.compile(r"\bactive\s+bleed|\bmassive\s+(hemorrhage|bleed)", re.I), 1.2, 0.90, "massive_hemorrhage"),
    (re.compile(r"\borbital compartment\b", re.I), 1.5, 0.85, "orbital_compartment"),
    (re.compile(r"\bcompartment syndrome\b", re.I), 1.5, 0.85, "compartment_syndrome"),
]

_EMERGENT_PATTERNS: list[tuple[re.Pattern, float, float, str]] = [
    (re.compile(r"\bchest pain\b", re.I), 2.2, 0.70, "chest_pain"),
    (re.compile(r"\bshortness of breath\b|\bdyspnea\b|\bdifficulty breathing\b", re.I), 2.3, 0.65, "dyspnea"),
    (re.compile(r"\baltered mental status\b|\bams\b", re.I), 2.0, 0.80, "altered_mental_status"),
    (re.compile(r"\bsuicidal\b", re.I), 2.0, 0.75, "suicidal"),
    (re.compile(r"\boverdose\b|\bingestion\b.*\btoxic\b", re.I), 2.0, 0.75, "overdose"),
    (re.compile(r"\bsepsis\b|\bseptic\b", re.I), 2.0, 0.80, "sepsis"),
    (re.compile(r"\bstab\b|\bgunshot\b|\bgsw\b|\bpenetrating\b", re.I), 1.5, 0.85, "penetrating_trauma"),
    (re.compile(r"\bhead injury\b|\bhead trauma\b|\btbi\b", re.I), 2.0, 0.75, "head_injury"),
    (re.compile(r"\babdominal pain\b.*\bsevere\b|\bsevere\b.*\babdominal\b", re.I), 2.5, 0.60, "severe_abd_pain"),
    (re.compile(r"\bthunderclap headache\b", re.I), 1.8, 0.85, "thunderclap_headache"),
    (re.compile(r"\bworst headache\b", re.I), 2.0, 0.75, "worst_headache"),
    (re.compile(r"\bhematemesis\b|\bvomiting blood\b", re.I), 2.2, 0.70, "hematemesis"),
    (re.compile(r"\bmeningitis\b|\bmeningeal\b", re.I), 2.0, 0.75, "meningitis"),
    (re.compile(r"\bpulmonary embolism\b|\bpe\b.*\bsuspect\b", re.I), 2.0, 0.75, "pe_suspect"),
    (re.compile(r"\baortic\b.*\bdissection\b", re.I), 1.5, 0.88, "aortic_dissection"),
    (re.compile(r"\btesticular torsion\b", re.I), 2.0, 0.75, "testicular_torsion"),
    (re.compile(r"\bectopic\b.*\bpregnancy\b", re.I), 2.0, 0.75, "ectopic_pregnancy"),
    (re.compile(r"\bdiaphoresis\b", re.I), 2.3, 0.55, "diaphoresis"),
]

# ---------------------------------------------------------------------------
# Low-acuity complaint patterns (ESI 4-5)
# ---------------------------------------------------------------------------

_MINOR_PATTERNS: list[tuple[re.Pattern, float, float, str]] = [
    (re.compile(r"\bsore throat\b|\bpharyngitis\b", re.I), 4.5, 0.55, "sore_throat"),
    (re.compile(r"\bcold symptoms\b|\burti\b|\bupper respiratory\b", re.I), 4.5, 0.55, "uri"),
    (re.compile(r"\bear pain\b|\botalgia\b", re.I), 4.3, 0.45, "ear_pain"),
    (re.compile(r"\bminor\s+(cut|laceration|wound)\b", re.I), 4.5, 0.50, "minor_laceration"),
    (re.compile(r"\binsect bite\b|\bbug bite\b", re.I), 4.8, 0.55, "insect_bite"),
    (re.compile(r"\brash\b(?!.*\bpetechial\b)", re.I), 4.3, 0.40, "rash"),
    (re.compile(r"\bsprain\b|\bstrain\b", re.I), 4.2, 0.45, "sprain"),
    (re.compile(r"\bprescription\b.*\brefill\b|\bmedication refill\b", re.I), 5.0, 0.70, "rx_refill"),
    (re.compile(r"\bsuture removal\b|\bstaple removal\b", re.I), 5.0, 0.70, "suture_removal"),
    (re.compile(r"\bcongestion\b|\brunny nose\b", re.I), 4.8, 0.50, "congestion"),
    (re.compile(r"\bconjunctivitis\b|\bpink eye\b", re.I), 4.5, 0.50, "conjunctivitis"),
    (re.compile(r"\bcontraception\b|\bcontraceptive\b", re.I), 5.0, 0.60, "contraception"),
    (re.compile(r"\bconstipation\b", re.I), 4.5, 0.45, "constipation"),
    (re.compile(r"\bsleep\s+hygiene\b", re.I), 5.0, 0.60, "sleep_hygiene"),
]

# ---------------------------------------------------------------------------
# Severity modifiers (escalate or de-escalate)
# ---------------------------------------------------------------------------

_ESCALATION_MODIFIERS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bacute\b", re.I), -0.3),
    (re.compile(r"\bsevere\b", re.I), -0.5),
    (re.compile(r"\bworsening\b", re.I), -0.3),
    (re.compile(r"\bsudden\s+onset\b", re.I), -0.4),
    (re.compile(r"\brapid\b", re.I), -0.2),
    (re.compile(r"\buncontrolled\b", re.I), -0.3),
    (re.compile(r"\bwith diaphoresis\b", re.I), -0.5),
    (re.compile(r"\bwith rigors\b", re.I), -0.3),
    (re.compile(r"\bwith syncope\b", re.I), -0.4),
    (re.compile(r"\bradiating\b", re.I), -0.3),
]

_DEESCALATION_MODIFIERS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bmild\b", re.I), 0.4),
    (re.compile(r"\bchronic\b", re.I), 0.3),
    (re.compile(r"\bintermittent\b", re.I), 0.2),
    (re.compile(r"\bfor \d+ days\b", re.I), 0.2),
    (re.compile(r"\bfollow.?up\b", re.I), 0.5),
    (re.compile(r"\badvice\b", re.I), 0.5),
    (re.compile(r"\bstable\b", re.I), 0.3),
    (re.compile(r"\bminor\b", re.I), 0.4),
    (re.compile(r"\brecurrent\b", re.I), 0.2),
]

# ---------------------------------------------------------------------------
# Complaint system → base ESI mapping
# ---------------------------------------------------------------------------

_SYSTEM_BASE_ESI: dict[str, float] = {
    "cardiovascular": 2.5,
    "neurological": 2.8,
    "respiratory": 2.8,
    "trauma": 3.0,
    "psychiatric": 3.0,
    "gastrointestinal": 3.5,
    "genitourinary": 3.5,
    "infectious": 3.2,
    "endocrine": 3.5,
    "musculoskeletal": 3.8,
    "ophthalmic": 3.8,
    "ENT": 4.0,
    "dermatological": 4.0,
    "other": 3.5,
}


def classify_complaint(complaint_raw: str | None,
                       complaint_system: str | None) -> BankSignal:
    """Classify a chief complaint into an ESI estimate with confidence.

    Applies pattern matching in priority order:
    1. Critical patterns (ESI 1-2, high confidence)
    2. Emergent patterns (ESI 2-3, medium confidence)
    3. Minor patterns (ESI 4-5, medium confidence)
    4. System-based fallback (ESI 2.5-4.0, low confidence)
    5. Modifier adjustment (escalation/de-escalation)
    """
    if not complaint_raw:
        system_esi = _SYSTEM_BASE_ESI.get(str(complaint_system), 3.5)
        return BankSignal(Bank.COMPLAINT, system_esi, 0.15, 0, 6,
                          f"system={complaint_system} no_text")

    text = str(complaint_raw).lower().strip()

    # Phase 1: Check critical patterns
    for pattern, esi, conf, label in _CRITICAL_PATTERNS:
        if pattern.search(text):
            return BankSignal(Bank.COMPLAINT, esi, conf, 1, 2,
                              f"critical:{label}")

    # Phase 2: Check emergent patterns
    best_emergent = None
    for pattern, esi, conf, label in _EMERGENT_PATTERNS:
        if pattern.search(text):
            if best_emergent is None or conf > best_emergent[1]:
                best_emergent = (esi, conf, label)

    # Phase 3: Check minor patterns
    best_minor = None
    for pattern, esi, conf, label in _MINOR_PATTERNS:
        if pattern.search(text):
            if best_minor is None or conf > best_minor[1]:
                best_minor = (esi, conf, label)

    # Phase 4: Apply modifiers
    modifier_delta = 0.0
    for pattern, delta in _ESCALATION_MODIFIERS:
        if pattern.search(text):
            modifier_delta += delta
    for pattern, delta in _DEESCALATION_MODIFIERS:
        if pattern.search(text):
            modifier_delta += delta

    # Phase 5: Select best match
    # Clinical priority: emergent ALWAYS wins over minor when both match
    if best_emergent and best_minor:
        esi = max(1.0, min(5.0, best_emergent[0] + modifier_delta))
        return BankSignal(Bank.COMPLAINT, esi, best_emergent[1], 1, 4,
                          f"emergent:{best_emergent[2]} mod={modifier_delta:+.1f}")

    if best_emergent:
        esi = max(1.0, min(5.0, best_emergent[0] + modifier_delta))
        return BankSignal(Bank.COMPLAINT, esi, best_emergent[1], 1, 4,
                          f"emergent:{best_emergent[2]} mod={modifier_delta:+.1f}")

    if best_minor:
        esi = max(1.0, min(5.0, best_minor[0] + modifier_delta))
        return BankSignal(Bank.COMPLAINT, esi, best_minor[1], 2, 5,
                          f"minor:{best_minor[2]} mod={modifier_delta:+.1f}")

    # Phase 6: Fall back to system-based ESI
    system_esi = _SYSTEM_BASE_ESI.get(str(complaint_system), 3.5)
    esi = max(1.0, min(5.0, system_esi + modifier_delta))
    conf = 0.25 + min(0.15, abs(modifier_delta) * 0.3)
    return BankSignal(Bank.COMPLAINT, esi, conf, 0, 6,
                      f"system:{complaint_system} mod={modifier_delta:+.1f}")


def classify_complaints_batch(df_complaints) -> dict[str, BankSignal]:
    """Classify all complaints in a DataFrame.

    Args:
        df_complaints: DataFrame with patient_id, chief_complaint_raw,
                       chief_complaint_system columns.

    Returns:
        dict mapping patient_id → BankSignal
    """
    results = {}
    for _, row in df_complaints.iterrows():
        pid = str(row["patient_id"])
        signal = classify_complaint(
            row.get("chief_complaint_raw"),
            row.get("chief_complaint_system"),
        )
        results[pid] = signal
    return results
