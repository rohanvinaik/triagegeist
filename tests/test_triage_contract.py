"""Tests for src.triage_contract — kill-targeted per LintGate prescriptions."""
from __future__ import annotations

import json

import pytest

from src.triage_contract import (
    AlignmentChoice, BankReading, DecisiveEvidence, DissentDirection,
    ESILevel, LLM_DECISION_JSON_SCHEMA, PatientCategory, PatientClinical,
    SymbolicVerdict, TierBContext, TriageDecision, TriagePacket,
    parse_decision, render_prompt,
)


# ---------- TriagePacket.esi_choices ----------

def test_esi_choices_value_returns_int_pair(make_packet):
    packet = make_packet(candidates=(ESILevel.E3, ESILevel.E4))
    assert packet.esi_choices() == (3, 4)


def test_esi_choices_value_preserves_order(make_packet):
    packet = make_packet(candidates=(ESILevel.E4, ESILevel.E3))
    assert packet.esi_choices() == (4, 3)


def test_esi_choices_value_returns_ints_not_strings(make_packet):
    packet = make_packet(candidates=(ESILevel.E1, ESILevel.E2))
    a, b = packet.esi_choices()
    assert isinstance(a, int) and isinstance(b, int)
    assert (a, b) == (1, 2)


# ---------- render_prompt ----------

def test_render_prompt_value_contains_patient_fields(make_packet):
    packet = make_packet()  # default age=45, sex=M, complaint="chest pain"
    out = render_prompt(packet)
    assert "Age 45" in out
    assert "Sex M" in out
    assert "chest pain" in out
    assert "HR 90" in out
    assert "BP 130" in out


def test_render_prompt_logical_dissent_first(make_packet):
    """Dissenting banks must appear in the 'dissenting' block before agreers."""
    readings = [
        BankReading("severity", 3.5, 0.7, 0.0, True, "low"),  # agrees
        BankReading("consciousness", 2.0, 0.85, -0.5, False,
                    "GCS<=8 comatose"),  # dissents
    ]
    packet = make_packet(symbolic_kw={
        "bank_readings": readings,
        "dissenting_banks": ["consciousness"],
    })
    out = render_prompt(packet)
    conscious_idx = out.index("consciousness")
    severity_idx = out.index("severity")
    # With readings [severity_first, consciousness_second], in render the
    # dissenters come first; consciousness dissents, so it must appear first.
    assert conscious_idx < severity_idx


def test_render_prompt_arithmetic_esi_defs_included(make_packet):
    """ESI definitions for BOTH candidates are rendered."""
    packet = make_packet(candidates=(ESILevel.E2, ESILevel.E3))
    out = render_prompt(packet)
    assert "ESI 2" in out
    assert "ESI 3" in out
    # The non-candidate ESIs should NOT appear in the definitions block
    # (they may appear in enum dumps though). Check the def lines only.
    esi_block = out.split("ESI DEFINITIONS")[1]
    assert "ESI 1 — Resuscitation" not in esi_block
    assert "ESI 2 — Emergent" in esi_block
    assert "ESI 3 — Urgent" in esi_block


def test_render_prompt_logical_flags_block_conditional(make_packet):
    """ATYPICAL PATTERNS block appears only if tier_b_flags is non-empty."""
    p_no_flags = make_packet(tier_b_kw={"tier_b_flags": []})
    p_with_flags = make_packet(tier_b_kw={
        "tier_b_flags": ["temporal_paradox"],
    })
    assert "ATYPICAL PATTERNS" not in render_prompt(p_no_flags)
    assert "ATYPICAL PATTERNS" in render_prompt(p_with_flags)
    assert "temporal_paradox" in render_prompt(p_with_flags)


# ---------- parse_decision ----------

def _valid_json_payload():
    return {
        "esi_choice": "3",
        "alignment": "consensus",
        "decisive_evidence": ["chest_pain_cardiac"],
        "reasoning_summary": "stable vitals, cardiac risk factors",
    }


def test_parse_decision_value_valid_json_string(make_packet):
    packet = make_packet(candidates=(ESILevel.E3, ESILevel.E4))
    raw = json.dumps(_valid_json_payload())
    d = parse_decision(raw, packet)
    assert d is not None
    assert d.esi_choice == ESILevel.E3
    assert d.alignment == AlignmentChoice.CONSENSUS
    assert d.decisive_evidence == [DecisiveEvidence.CHEST_PAIN_CARDIAC]


def test_parse_decision_value_valid_dict(make_packet):
    packet = make_packet(candidates=(ESILevel.E3, ESILevel.E4))
    d = parse_decision(_valid_json_payload(), packet)
    assert d is not None
    assert d.esi_choice == ESILevel.E3


def test_parse_decision_boundary_evidence_count_low(make_packet):
    """Evidence list of length 0 → reject."""
    packet = make_packet(candidates=(ESILevel.E3, ESILevel.E4))
    payload = _valid_json_payload()
    payload["decisive_evidence"] = []  # below min
    assert parse_decision(payload, packet) is None


def test_parse_decision_boundary_evidence_count_high(make_packet):
    """Evidence list of length 4 → reject (max 3)."""
    packet = make_packet(candidates=(ESILevel.E3, ESILevel.E4))
    payload = _valid_json_payload()
    payload["decisive_evidence"] = ["chest_pain_cardiac"] * 4
    assert parse_decision(payload, packet) is None


def test_parse_decision_logical_esi_not_in_candidates(make_packet):
    """ESI not in the candidate pair → reject."""
    packet = make_packet(candidates=(ESILevel.E3, ESILevel.E4))
    payload = _valid_json_payload()
    payload["esi_choice"] = "1"  # not in (3, 4)
    assert parse_decision(payload, packet) is None


def test_parse_decision_logical_enum_violation(make_packet):
    """Unknown alignment enum value → reject."""
    packet = make_packet(candidates=(ESILevel.E3, ESILevel.E4))
    payload = _valid_json_payload()
    payload["alignment"] = "not_a_real_alignment"
    assert parse_decision(payload, packet) is None


def test_parse_decision_type_wrong_type(make_packet):
    """Non-str/dict input → None, no exception."""
    packet = make_packet()
    assert parse_decision(None, packet) is None
    assert parse_decision(42, packet) is None
    assert parse_decision([1, 2, 3], packet) is None


def test_parse_decision_type_malformed_json(make_packet):
    """Invalid JSON string → None."""
    packet = make_packet()
    assert parse_decision("{not json", packet) is None
    assert parse_decision("", packet) is None


def test_parse_decision_swap_returns_different_on_different_packets(
        make_packet):
    """Same raw JSON yields different results if packet's candidate set differs.
    (packet is a filter, so swapping packet changes acceptance.)"""
    raw = _valid_json_payload()  # esi_choice="3"
    p_has3 = make_packet(candidates=(ESILevel.E3, ESILevel.E4))
    p_no3 = make_packet(candidates=(ESILevel.E1, ESILevel.E2))
    assert parse_decision(raw, p_has3) is not None
    assert parse_decision(raw, p_no3) is None


# ---------- JSON schema ----------

def test_schema_all_required_fields_listed():
    required = set(LLM_DECISION_JSON_SCHEMA["required"])
    assert required == {"esi_choice", "alignment", "decisive_evidence"}


def test_schema_additional_properties_false():
    assert LLM_DECISION_JSON_SCHEMA["additionalProperties"] is False
