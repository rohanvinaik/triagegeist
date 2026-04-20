# triagegeist Context

## VERIFIED FROM PRIMARY SOURCES

**Competition format: PANEL-JUDGED, NOT leaderboard-scored.** No public or private
leaderboard. Submissions are a public Kaggle Notebook + Writeup (≤2,000 words) +
cover image + project link, judged by ≥3-panel on a 100-point rubric:
Technical Quality 30 / Clinical Relevance 25 / Docs 20 / Insight 15 / Novelty 10.
Acting on this: prioritize writeup/notebook polish and clinical framing over raw
metric chasing. 72 teams as of 2026-04-17. Deadline 2026-04-21 22:00 UTC.

**No official evaluation metric.** The Kaggle API has no `evaluation_metric` field;
the competition is rubric-scored. Competitor-notebook consensus for self-reporting
is **quadratic weighted Kappa (QWK)** — used by kimseunghwan9823, dhruvjain35,
and implied by several others. jasonkarpeles uses linear-weighted kappa;
udaken10 reports accuracy. Our writeup reports QWK, accuracy, macro F1, and
linear kappa for triangulation.

**Submission format:** CSV with `patient_id` (string) and `triage_acuity` (integer 1-5).
Sample submission is the constant-3 baseline. Row order must match test.csv —
verified in our final submission.

**Data files on disk (what's actually provided):**
- `train.csv` — 80K rows × 40 cols (includes leakage: `disposition`, `ed_los_hours`)
- `test.csv` — 20K rows × 37 cols (no target, no leakage cols)
- `chief_complaints.csv` — 100K rows × 3 cols (`patient_id`, `chief_complaint_raw`,
  `chief_complaint_system`). The `chief_complaint_system` column is NOT documented
  on the competition's data-description page but is present in the data.
- `patient_history.csv` — 100K rows × 26 cols (25 `hx_*` binary comorbidity flags).
  This file is NOT documented on the data-description page either.
- `sample_submission.csv` — 20K rows, constant ESI=3.

**Current submission (as of 2026-04-18, option-2 regenerate, verified):** 575
hard-rules (2.9%) + 19,396 ensemble+QWK (97.0%) + 29 Claude residual (0.1%).
Distribution 3.65/17.16/36.49/29.11/13.58% — close to train (4.0/16.8/36.2/28.8/14.2%).
124 ensemble features (phase deviations + style + calibrator + 4 of 5 temporal
features dropped per `project_tier_b_forensic_verdict.md`; `temporal_news2_deviation`
retained). Pre-computed Claude decisions (662) were applied to the new
ensemble but only 29 patients still fall in the < 0.20 top-2 gap band under
the new ensemble config (vs 538 previously); consider re-running the LLM
residual on the new uncertain set if rubric signal from Scale 3 matters.

**NEWS2-only baseline:** QWK ≈ 0.78, accuracy 0.55. Our pipeline's +0.19 QWK lift
over this baseline is what justifies the multi-scale complexity.

See `../../.claude/CLAUDE.md` for the Cardinal Rule on primary source verification.

<!-- LINTGATE:BEGIN compass_state v1 -->
## True North

Predict Emergency Severity Index (ESI) triage acuity levels 1-5 for emergency department patients. This is a Kaggle panel-judged community competition (Triagegeist, $10K prize, 72 teams, deadline 2026-04-21 22:00 UTC). The input is structured tabular data: 80K train / 20K test patients with vitals (BP, HR, RR, temp, SpO2, GCS, pain score, NEWS2 score), demographics (age, sex, language, insurance), arrival info (mode, time, origin), chief complaint text, and 25 binary medical history flags. Train includes 2 leakage columns (disposition, ed_los_hours) that must be excluded. Target is ordinal 1-5 ESI scale with class imbalance (ESI 1=4%, ESI 3=36%).

## Architecture Philosophy

**Two-tier multi-scale with AST-style typed LLM I/O.**

**Tier A (QWK-optimal, symbolic + ensemble, no LLM):**
1. Deterministic clinical rules (GCS ≤ 8 → ESI 1, cardiac arrest → ESI 1).
2. 11-bank decomposition with confidence-weighted Kuramoto coherence.
   Phase-deviation features (`bank_*_dev`) were benchmarked and REMOVED
   from the ensemble — 5-fold regime-split CV showed easy Δ=+0.0000 (p=1.00),
   hard Δ=−0.0025 (Cohen's d=−0.68). Phase magnitudes still render into the
   Scale-3 LLM audit context via `ClinicianReport`.
3. CatBoost + LightGBM ensemble (3+2 seeds, 60/40 probability blend).
4. QWK threshold optimization on OOF probabilities.

**Tier B — one feature kept, the rest dropped from the ensemble**
(see `memory/project_tier_b_forensic_verdict.md` for the 12-variant ablation):
5. **Kept in ensemble**: `temporal_news2_deviation` =
   `own_NEWS2 − E[NEWS2 | complaint_base, age_group]`. Carries 91% of the
   +0.0168 Tier-B QWK lift. Laundering mechanism (cohort-mean NEWS2
   correlates with cohort-mean ESI on this synthetic dataset) disclosed in
   the writeup; feature has real clinical grounding (silent MI / medication
   masking) and is kept under option 2.
6. **Dropped from ensemble, retained as LLM audit context only**: the
   other 4 temporal features, 9 clinician style features, the isotonic
   severe-outcome calibrator, 87 surprisal-basis features, and 11 phase
   deviations. All benchmarked, all inert or negative on 5-fold CV.
7. `ClinicianReport.to_llm_context()` still renders the full inventory
   (phase magnitudes, style deviations, cohort paradox) into the Scale-3
   prompt — they're clinically interpretable even when they don't lift the
   ensemble metric.

**Scale 3 — constrained LLM (AST-style):**
- Prompt rendered deterministically from a frozen `TriagePacket` dataclass
  (`src/triage_contract.py`). No free-text interpolation except chief
  complaint.
- LLM must emit JSON conforming to a closed-enum schema (`TriageDecision`:
  `esi_choice`, `alignment`, `decisive_evidence` from closed vocab).
- `AnswerCertifier` (`src/answer_certifier.py`) runs 5 deterministic checks
  (hard floors, evidence-packet consistency, alignment-bank consistency,
  etc.) and rejects violators.
- LLM's free-text `reasoning_summary` is stored for audit, NEVER parsed.

This beats generic ML because clinical domain knowledge encodes ESI decision
rules directly AND the LLM is architecturally constrained to make
auditable, machine-verifiable choices.

## Cardinal rule on the LLM

The LLM exists to resolve uncertainties in conflicting or noisy symbolic
signals, nothing else. Do not widen its job. Do not let free-form text
from the LLM enter the submission pipeline under any path. The certifier is
the firewall; changes to the LLM prompt or output schema MUST also update
the certifier checks so consistency is enforced.

## Machine Rules
# LINTGATE_FORBID: proposed_rules
# LINTGATE_FORBID: existing_rule_count
# LINTGATE_FORBID: directives_analyzed
# LINTGATE_FORBID: already_covered

<!-- LINTGATE:END compass_state -->

## Implementation Notes

Uses src/ layout