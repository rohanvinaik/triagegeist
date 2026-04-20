"""Tests for src.answer_certifier — kill-targeted per LintGate prescriptions.

Coverage map (12 prescriptions from mutation profile):
  _check_esi_in_candidates   VALUE
  _check_hard_floor          LOGICAL, VALUE
  _check_alignment_consistency LOGICAL, VALUE
  _check_evidence_vs_packet  LOGICAL, SWAP, VALUE
  _check_pair_still_open     VALUE
  certify                    LOGICAL, SWAP, VALUE
"""
from __future__ import annotations

import pytest

from src.answer_certifier import (
    CertificationResult, _check_alignment_consistency,
    _check_esi_in_candidates, _check_evidence_vs_packet, _check_hard_floor,
    _check_pair_still_open, certify,
)
from src.triage_contract import (
    AlignmentChoice, BankReading, DecisiveEvidence, ESILevel,
)


def _fresh(esi: int = 3) -> CertificationResult:
    return CertificationResult(certified=False, esi_choice=esi)


# ---------- _check_esi_in_candidates ----------

def test_check_esi_in_candidates_value_match(make_packet, make_decision):
    packet = make_packet(candidates=(ESILevel.E3, ESILevel.E4))
    decision = make_decision(esi=ESILevel.E3)
    r = _fresh()
    _check_esi_in_candidates(r, decision, packet)
    assert r.checks_passed == ["esi_in_candidates"]
    assert r.checks_failed == []
    assert r.hard_contradiction is False


def test_check_esi_in_candidates_value_mismatch(make_packet, make_decision):
    packet = make_packet(candidates=(ESILevel.E3, ESILevel.E4))
    decision = make_decision(esi=ESILevel.E1)
    r = _fresh()
    _check_esi_in_candidates(r, decision, packet)
    assert r.checks_failed == ["esi_not_in_candidates"]
    assert r.hard_contradiction is True


# ---------- _check_hard_floor ----------

def test_check_hard_floor_logical_branch_no_violation(make_packet,
                                                       make_decision):
    """Packet has no hard-floor evidence → hard_floor_respected."""
    packet = make_packet()  # default: no 'comatose'/'cardiac_arrest' etc
    decision = make_decision(esi=ESILevel.E3)
    r = _fresh()
    _check_hard_floor(r, decision, packet)
    assert "hard_floor_respected" in r.checks_passed
    assert r.hard_contradiction is False


def test_check_hard_floor_logical_branch_violation(make_packet,
                                                    make_decision):
    """Packet flags comatose, LLM chose ESI 3 → hard contradiction."""
    readings = [
        BankReading("consciousness", 1.0, 0.95, 0.0, False,
                    "GCS<=8 comatose"),
    ]
    packet = make_packet(symbolic_kw={"bank_readings": readings,
                                       "dissenting_banks": ["consciousness"]},
                          candidates=(ESILevel.E1, ESILevel.E3))
    decision = make_decision(esi=ESILevel.E3)
    r = _fresh(3)
    _check_hard_floor(r, decision, packet)
    assert any(c.startswith("hard_floor_violation:comatose")
               for c in r.checks_failed)
    assert r.hard_contradiction is True


def test_check_hard_floor_value_cardiac_arrest(make_packet, make_decision):
    readings = [
        BankReading("complaint", 1.0, 0.98, 0.0, False, "cardiac_arrest"),
    ]
    packet = make_packet(symbolic_kw={"bank_readings": readings},
                          candidates=(ESILevel.E1, ESILevel.E2))
    decision = make_decision(esi=ESILevel.E2)
    r = _fresh(2)
    _check_hard_floor(r, decision, packet)
    assert any("cardiac_arrest" in c for c in r.checks_failed)
    assert r.hard_contradiction is True


# ---------- _check_alignment_consistency ----------

def test_check_alignment_consistency_logical_match(make_packet,
                                                    make_decision):
    """Alignment cites a bank that IS dissenting → passes."""
    packet = make_packet(symbolic_kw={
        "dissenting_banks": ["consciousness"],
    })
    decision = make_decision(alignment=AlignmentChoice.DISSENT_CONSCIOUSNESS)
    r = _fresh()
    _check_alignment_consistency(r, decision, packet)
    assert "alignment_consistent" in r.checks_passed
    assert r.checks_failed == []


def test_check_alignment_consistency_logical_mismatch(make_packet,
                                                      make_decision):
    """Alignment cites a bank that is NOT dissenting → soft failure."""
    packet = make_packet(symbolic_kw={
        "dissenting_banks": ["severity"],  # consciousness is NOT dissenting
    })
    decision = make_decision(alignment=AlignmentChoice.DISSENT_CONSCIOUSNESS)
    r = _fresh()
    _check_alignment_consistency(r, decision, packet)
    assert any("alignment_bank_not_dissenting:consciousness" in c
               for c in r.checks_failed)
    assert r.hard_contradiction is False  # soft failure
    assert r.notes  # has warning note


def test_check_alignment_consistency_value_consensus(make_packet,
                                                      make_decision):
    """CONSENSUS alignment always passes (no required_bank)."""
    packet = make_packet(symbolic_kw={"dissenting_banks": []})
    decision = make_decision(alignment=AlignmentChoice.CONSENSUS)
    r = _fresh()
    _check_alignment_consistency(r, decision, packet)
    assert "alignment_consistent" in r.checks_passed


# ---------- _check_evidence_vs_packet ----------

def test_check_evidence_vs_packet_logical_match(make_packet, make_decision):
    """Immunocompromise_fever cited AND hx_immunosuppressed+fever present."""
    packet = make_packet(clinical_kw={
        "comorbidity_flags": ["hx_immunosuppressed"],
        "vitals": {"heart_rate": 90, "systolic_bp": 130, "spo2": 98,
                   "temperature_c": 38.6, "respiratory_rate": 16},
    })
    decision = make_decision(evidence=[DecisiveEvidence.IMMUNOCOMPROMISE_FEVER])
    r = _fresh()
    _check_evidence_vs_packet(r, decision, packet)
    assert "evidence_consistent_with_packet" in r.checks_passed
    assert r.checks_failed == []


def test_check_evidence_vs_packet_logical_mismatch(make_packet, make_decision):
    """Immunocompromise_fever cited, but patient has no hx → violation."""
    packet = make_packet(clinical_kw={
        "comorbidity_flags": ["hx_hypertension"],
        "vitals": {"heart_rate": 90, "systolic_bp": 130, "spo2": 98,
                   "temperature_c": 37.0, "respiratory_rate": 16},
    })
    decision = make_decision(evidence=[DecisiveEvidence.IMMUNOCOMPROMISE_FEVER])
    r = _fresh()
    _check_evidence_vs_packet(r, decision, packet)
    assert "immunocompromise_fever_no_evidence" in r.checks_failed
    assert r.hard_contradiction is False  # soft failure


def test_check_evidence_vs_packet_swap_order_invariant(make_packet,
                                                        make_decision):
    """Evidence list order doesn't change check outcome."""
    packet = make_packet(clinical_kw={"pain": 9})
    d1 = make_decision(evidence=[DecisiveEvidence.SEVERE_PAIN,
                                  DecisiveEvidence.CHEST_PAIN_CARDIAC])
    d2 = make_decision(evidence=[DecisiveEvidence.CHEST_PAIN_CARDIAC,
                                  DecisiveEvidence.SEVERE_PAIN])
    r1, r2 = _fresh(), _fresh()
    _check_evidence_vs_packet(r1, d1, packet)
    _check_evidence_vs_packet(r2, d2, packet)
    assert r1.checks_passed == r2.checks_passed
    assert r1.checks_failed == r2.checks_failed


@pytest.mark.parametrize("evidence,clin_kw,expect_violation", [
    (DecisiveEvidence.SEVERE_PAIN, {"pain": 3}, "severe_pain_but"),
    (DecisiveEvidence.SPO2_DEPRESSED,
     {"vitals": {"heart_rate": 90, "systolic_bp": 130, "spo2": 98,
                 "temperature_c": 37.0, "respiratory_rate": 16}},
     "spo2_depressed_but"),
    (DecisiveEvidence.FEVER_SIGNIFICANT,
     {"vitals": {"heart_rate": 90, "systolic_bp": 130, "spo2": 98,
                 "temperature_c": 37.2, "respiratory_rate": 16}},
     "fever_but"),
    (DecisiveEvidence.AGE_EXTREME_PEDIATRIC, {"age": 30}, "pediatric_but"),
    (DecisiveEvidence.AGE_EXTREME_GERIATRIC, {"age": 40}, "geriatric_but"),
    (DecisiveEvidence.COAGULOPATHY_BLEEDING,
     {"comorbidity_flags": ["hx_hypertension"]},
     "coagulopathy_bleeding_no_hx"),
    (DecisiveEvidence.PREGNANCY_COMPLICATION,
     {"comorbidity_flags": []},
     "pregnancy_no_hx"),
])
def test_check_evidence_vs_packet_value_violations(
        make_packet, make_decision, evidence, clin_kw, expect_violation):
    """Each evidence category fails when its precondition isn't met."""
    packet = make_packet(clinical_kw=clin_kw)
    decision = make_decision(evidence=[evidence])
    r = _fresh()
    _check_evidence_vs_packet(r, decision, packet)
    assert any(expect_violation in c for c in r.checks_failed)


# ---------- _check_pair_still_open ----------

def test_check_pair_still_open_value_ok(make_decision):
    decision = make_decision(esi=ESILevel.E3)
    r = _fresh()
    _check_pair_still_open(r, decision)
    assert r.checks_failed == []
    assert r.hard_contradiction is False


# ---------- certify (top-level) ----------

def test_certify_logical_happy_path(make_packet, make_decision):
    """All checks pass → certified=True."""
    packet = make_packet()
    decision = make_decision(esi=ESILevel.E3,
                              alignment=AlignmentChoice.CONSENSUS)
    r = certify(decision, packet)
    assert r.certified is True
    assert r.esi_choice == 3
    assert r.hard_contradiction is False
    assert "esi_in_candidates" in r.checks_passed
    assert "hard_floor_respected" in r.checks_passed


def test_certify_logical_hard_contradiction_short_circuits(make_packet,
                                                           make_decision):
    """ESI not in candidates → hard contradiction, skip later checks."""
    packet = make_packet(candidates=(ESILevel.E2, ESILevel.E3))
    decision = make_decision(esi=ESILevel.E5)  # not in (2, 3)
    r = certify(decision, packet)
    assert r.certified is False
    assert r.hard_contradiction is True
    assert "esi_not_in_candidates" in r.checks_failed
    # short-circuited — later checks didn't execute
    assert "hard_floor_respected" not in r.checks_passed


def test_certify_swap_soft_failures_dont_block(make_packet, make_decision):
    """Alignment mismatch is SOFT — certified stays True."""
    packet = make_packet(symbolic_kw={"dissenting_banks": ["severity"]})
    decision = make_decision(esi=ESILevel.E3,
                              alignment=AlignmentChoice.DISSENT_CONSCIOUSNESS)
    r = certify(decision, packet)
    assert r.certified is True  # soft failure doesn't block
    assert any("alignment_bank_not_dissenting" in c
               for c in r.checks_failed)


def test_certify_value_esi_choice_field(make_packet, make_decision):
    """esi_choice field on result matches decision's integer ESI."""
    packet = make_packet(candidates=(ESILevel.E2, ESILevel.E3))
    decision = make_decision(esi=ESILevel.E2)
    r = certify(decision, packet)
    assert r.esi_choice == 2
