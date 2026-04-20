"""Shared test fixtures for triagegeist.

Provides typed TriagePacket factories so each test declares only the fields
it actually cares about.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.triage_contract import (
    AlignmentChoice, BankReading, DecisiveEvidence, DissentDirection,
    ESILevel, PatientCategory, PatientClinical, SymbolicVerdict,
    TierBContext, TriageDecision, TriagePacket,
)


@pytest.fixture
def make_clinical():
    def _make(**overrides):
        defaults = dict(
            age=45, sex="M", arrival_mode="walk-in",
            chief_complaint="chest pain", mental_status="alert",
            vitals={"heart_rate": 90, "systolic_bp": 130, "spo2": 98,
                    "temperature_c": 37.0, "respiratory_rate": 16},
            gcs=15, pain=4, news2=1.0,
            num_comorbidities=2, num_active_medications=3,
            comorbidity_flags=["hx_hypertension"],
        )
        defaults.update(overrides)
        return PatientClinical(**defaults)
    return _make


@pytest.fixture
def make_symbolic():
    def _make(**overrides):
        defaults = dict(
            bank_readings=[
                BankReading("severity", 3.5, 0.7, 0.1, True, "news2=1 low-medium"),
                BankReading("cardiovascular", 4.5, 0.3, 0.4, True, "hr=90 sbp=130 stable"),
            ],
            order_parameter_r=0.85,
            dissent_direction=DissentDirection.AGREED,
            dissenting_banks=[],
        )
        defaults.update(overrides)
        return SymbolicVerdict(**defaults)
    return _make


@pytest.fixture
def make_tier_b():
    def _make(**overrides):
        defaults = dict(
            patient_category=PatientCategory.MODEL_UNCERTAINTY,
            tier_b_flags=[],
            calibrated_severe_outcome_prob=0.35,
        )
        defaults.update(overrides)
        return TierBContext(**defaults)
    return _make


@pytest.fixture
def make_packet(make_clinical, make_symbolic, make_tier_b):
    def _make(*, clinical_kw=None, symbolic_kw=None, tier_b_kw=None,
              candidates=(ESILevel.E3, ESILevel.E4),
              patient_id="TEST-001"):
        return TriagePacket(
            patient_id=patient_id,
            clinical=make_clinical(**(clinical_kw or {})),
            symbolic=make_symbolic(**(symbolic_kw or {})),
            tier_b=make_tier_b(**(tier_b_kw or {})),
            candidate_esis=candidates,
        )
    return _make


@pytest.fixture
def make_decision():
    def _make(*, esi=ESILevel.E3,
              alignment=AlignmentChoice.CONSENSUS,
              evidence=None,
              reasoning=""):
        if evidence is None:
            evidence = [DecisiveEvidence.CHEST_PAIN_CARDIAC]
        return TriageDecision(
            esi_choice=esi, alignment=alignment,
            decisive_evidence=evidence,
            reasoning_summary=reasoning,
        )
    return _make
