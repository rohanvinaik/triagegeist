"""
LLM residual layer (Scale 3) for ESI boundary disambiguation.

Only fires for cases where the model ensemble is uncertain —
specifically where the top-2 predicted classes have close
probabilities. Uses Qwen 3 8B via Ollama for structured
clinical reasoning.

This is the NCEMS Phase D healing pattern: the LLM picks
from candidates (ESI 4 vs 5), it doesn't generate freely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"

# Only invoke LLM when the probability gap between top-2 classes
# is smaller than this threshold
UNCERTAINTY_THRESHOLD = 0.15


@dataclass
class LLMDecision:
    patient_id: str
    original_pred: int
    llm_pred: int
    confidence_gap: float
    reasoning: str


def _build_prompt(row: dict, top2_classes: tuple[int, int],
                  complaint_raw: str | None,
                  tier_b_context: str | None = None) -> str:
    """Build a structured clinical prompt for ESI disambiguation.

    When `tier_b_context` is supplied (from ClinicianReport.to_llm_context),
    the LLM receives the 2nd-order context: bank-agreement landscape,
    atypical-pattern flags, treating-nurse style offsets, calibrated
    severe-outcome probability. These signals are individually weak on the
    synthetic training data but carry real information for boundary cases —
    they enrich the LLM's deconvolution without replacing 1st-order clinical
    data.
    """
    c1, c2 = top2_classes
    complaint = complaint_raw or "not recorded"
    context_block = f"\n\n{tier_b_context}\n" if tier_b_context else ""

    return f"""You are an experienced emergency department triage nurse using the Emergency Severity Index (ESI) 5-level system. Based on the following patient presentation, determine whether this patient is ESI {c1} or ESI {c2}.

PATIENT PRESENTATION:
- Age: {row.get('age', 'unknown')}, Sex: {row.get('sex', 'unknown')}
- Arrival: {row.get('arrival_mode', 'unknown')} from {row.get('transport_origin', 'unknown')}
- Chief complaint: {complaint}
- Mental status: {row.get('mental_status_triage', 'unknown')}
- Vitals: HR {row.get('heart_rate', '?')}, BP {row.get('systolic_bp', '?')}/{row.get('diastolic_bp', '?')}, RR {row.get('respiratory_rate', '?')}, Temp {row.get('temperature_c', '?')}°C, SpO2 {row.get('spo2', '?')}%
- GCS: {row.get('gcs_total', '?')}, Pain: {row.get('pain_score', '?')}/10
- NEWS2 score: {row.get('news2_score', '?')}
- Comorbidities: {row.get('num_comorbidities', '?')}, Active medications: {row.get('num_active_medications', '?')}
{context_block}
ESI DEFINITIONS:
{_esi_definitions(c1, c2)}

Answer with ONLY the number {c1} or {c2}. No explanation."""


_ESI_DEFS = {
    1: "ESI 1: Immediate. Life-threatening, requires immediate intervention (intubation, CPR, hemorrhage control).",
    2: "ESI 2: Emergent. High risk, confused/lethargic, severe pain/distress, or dangerous vitals.",
    3: "ESI 3: Urgent. Stable but expects 2+ resources (labs AND imaging, IV meds, specialist consult).",
    4: "ESI 4: Less urgent. Stable vitals, expects 1 resource (labs OR imaging OR procedure).",
    5: "ESI 5: Non-urgent. Stable vitals, expects 0 resources (exam only, reassurance, prescription).",
}


def _esi_definitions(c1: int, c2: int) -> str:
    """Return ESI definitions for the specific classes being disambiguated."""
    return "\n".join(f"- {_ESI_DEFS[c]}" for c in sorted({c1, c2}))


def _call_ollama(prompt: str, structured: bool = False) -> str | None:
    """Call Ollama API and extract response.

    When `structured=True`, requests a JSON-object response shape (Ollama
    format=json). Used in the typed-contract path where the LLM must emit
    a TriageDecision JSON; the caller then parses with triage_contract.
    """
    try:
        payload: dict = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 300 if structured else 50,
            },
        }
        if structured:
            payload["format"] = "json"
        resp = requests.post(OLLAMA_URL, json=payload, timeout=45)
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("response", "").strip()
            if not text:
                text = data.get("thinking", "").strip()
            return text
    except (requests.ConnectionError, requests.Timeout):
        pass
    return None


def heal_with_typed_contract(packets: list,
                              current_preds_by_pid: dict[str, int]
                              ) -> list:
    """Typed-contract disambiguation path — AST-style I/O, certified.

    Takes a list of TriagePackets (one per uncertain patient), renders each
    to a deterministic prompt via triage_contract.render_prompt, calls the
    LLM with JSON-format constraint, parses the output via
    triage_contract.parse_decision, and runs the answer certifier before
    accepting. Rejected outputs leave current_preds_by_pid unchanged.

    Returns list of CertificationResults for audit. Modifies
    current_preds_by_pid in place only for certified changes.
    """
    from .answer_certifier import certify
    from .triage_contract import parse_decision, render_prompt

    results = []
    for packet in packets:
        prompt = render_prompt(packet)
        raw = _call_ollama(prompt, structured=True)
        decision = parse_decision(raw, packet) if raw else None
        if decision is None:
            results.append(("parse_failure", packet.patient_id))
            continue
        cert = certify(decision, packet)
        if cert.certified and not cert.hard_contradiction:
            current_preds_by_pid[packet.patient_id] = cert.esi_choice
        results.append((cert, packet.patient_id, decision))
    return results


def _parse_esi(response: str | None, valid_classes: tuple[int, int]) -> int | None:
    """Parse LLM response to extract ESI prediction."""
    if not response:
        return None
    # Look for a digit that matches one of the valid classes
    for char in response:
        if char.isdigit() and int(char) in valid_classes:
            return int(char)
    return None


def identify_uncertain_cases(proba: np.ndarray,
                             threshold: float = UNCERTAINTY_THRESHOLD,
                             ) -> list[tuple[int, tuple[int, int], float]]:
    """Find cases where model is uncertain between top-2 classes.

    Returns list of (row_index, (class1, class2), gap) tuples.
    Classes are 1-indexed (ESI 1-5).
    """
    uncertain = []
    for i in range(len(proba)):
        sorted_idx = proba[i].argsort()[::-1]
        top1_prob = proba[i, sorted_idx[0]]
        top2_prob = proba[i, sorted_idx[1]]
        gap = top1_prob - top2_prob

        if gap < threshold:
            c1 = sorted_idx[0] + 1  # 0-indexed → 1-5
            c2 = sorted_idx[1] + 1
            uncertain.append((i, (c1, c2), gap))

    return uncertain


def heal_uncertain_cases(uncertain_cases: list[tuple[int, tuple[int, int], float]],
                         test_df: pd.DataFrame,
                         complaints_df: pd.DataFrame,
                         current_preds: list[int],
                         tier_b_contexts: dict[str, str] | None = None,
                         ) -> list[LLMDecision]:
    """Call LLM for each uncertain case to disambiguate.

    Modifies current_preds in place.
    Returns list of LLM decisions for audit trail.

    When `tier_b_contexts` is supplied (dict patient_id → ClinicianReport
    LLM-context string), the LLM sees the 2nd-order picture in addition to
    the raw clinical data: bank-agreement landscape, atypical-pattern flags,
    calibrated severe-outcome probability, clinician style. These signals
    don't need to be individually discriminative — they enrich the
    deconvolution context.
    """
    if not uncertain_cases:
        return []

    # Build complaint lookup
    complaint_lookup = {}
    for _, row in complaints_df.iterrows():
        complaint_lookup[row["patient_id"]] = row.get("chief_complaint_raw", "")

    decisions = []
    healed = 0
    failed = 0

    for row_idx, (c1, c2), gap in uncertain_cases:
        patient_row = test_df.iloc[row_idx]
        pid = patient_row["patient_id"]
        complaint = complaint_lookup.get(pid, "")
        tier_b = tier_b_contexts.get(pid) if tier_b_contexts else None

        prompt = _build_prompt(patient_row.to_dict(), (c1, c2), complaint,
                               tier_b_context=tier_b)
        response = _call_ollama(prompt)
        llm_esi = _parse_esi(response, (c1, c2))

        original = current_preds[row_idx]

        if llm_esi is not None:
            current_preds[row_idx] = llm_esi
            healed += 1
            decisions.append(LLMDecision(
                patient_id=pid,
                original_pred=original,
                llm_pred=llm_esi,
                confidence_gap=gap,
                reasoning=response or "",
            ))
        else:
            failed += 1

    print(f"  LLM healing: {healed} healed, {failed} failed, "
          f"{len(uncertain_cases)} total uncertain")

    return decisions
