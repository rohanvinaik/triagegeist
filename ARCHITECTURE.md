# TriageGeist — As-Built Architecture

Emergency Severity Index (ESI 1–5) prediction pipeline for the Kaggle
Triagegeist competition. Panel-judged community competition, no public
leaderboard; submission is a public Kaggle notebook + writeup.

**Current status (2026-04-18):** submission generated end-to-end by
`submission/generate_final.py`. Option-2 ensemble config per the forensic
verdict (`memory/project_tier_b_forensic_verdict.md`): phase-deviation
features and all Tier-B features except `temporal_news2_deviation` have been
dropped from the ensemble after 12-variant × 5-fold regime-split CV showed
they're inert or slightly negative. Final CV QWK ≈ 0.9815 (tuned).
Distribution matches training; zero catastrophic disagreements with top
public notebook kim_md.

The architecture is **two-tiered**:
- **Tier A** (hard rules → bank decomposition → ensemble → QWK thresholds)
  produces the deterministic and ensemble-confident predictions.
- **Tier B** originally five components (phase features, temporal paradox
  detector, clinician style embeddings, leakage-as-calibrator, surprisal
  basis). Only `temporal_news2_deviation` survived ablation and feeds the
  ensemble; the other four families are benchmarked-inert on this dataset
  but are still rendered into the Scale-3 LLM audit context (clinically
  interpretable, just doesn't lift the ensemble metric).

The Scale-3 LLM residual runs an **AST-style typed contract**: the prompt is
rendered from a frozen `TriagePacket` dataclass, the LLM must emit JSON
conforming to a closed-enum schema (`TriageDecision`), and a deterministic
`AnswerCertifier` gates every output before it's accepted. The LLM's private
reasoning is never the source of a submitted decision — only its typed
*choice* is, and that choice is auditable against the clinical evidence in
the packet.

---

## Competition framing

- Panel of ≥3 judges. 100-point rubric: Technical 30 / Clinical 25 /
  Writeup 20 / Insight 15 / Novelty 10.
- No auto-scoring. Writeup and notebook quality are load-bearing.
- 72 teams, 4 days to deadline.

## Pipeline overview

```
Patient record (structured + chief complaint + history)
  │
  ▼
──────────────── TIER A (QWK-optimal, no LLM) ────────────────
Scale 0  Bank decomposition + hard rules        src/banks.py, src/coherence.py
         11 orthogonal clinical banks produce    src/complaint_lexicon.py
         ESI estimate + confidence + floor/ceil.
         GCS ≤ 8 and critical-complaint keywords
         fire as hard rules (~2.9% of cases).
  │
  ▼
Scale 1  Feature engineering + GBM ensemble    src/feature_engine.py, src/model.py
         124 features (raw vitals + 25 hx_*    submission/generate_final.py
         flags + bank ESI/conf + bank-level    src/temporal_bank.py
         coherence scalars + clinical
         interactions + 12 text keyword-group
         counts + 1 Tier-B feature: temporal_
         news2_deviation). Phase-deviation
         columns (bank_*_dev) DROPPED
         post-ablation. No target encoding on
         the complaint base condition — see
         "Declined shortcut".
         3× CatBoost + 2× LightGBM, 60/40
         weighted probability average.
  │
  ▼
Scale 2  QWK threshold optimization             src/qwk_optimizer.py
         Expected-value transform on ensemble
         probs + Nelder-Mead thresholds on OOF
         predictions: [1.56, 2.52, 3.55, 4.58].
  │
  ▼
──────────────── TIER B (context enrichment for the hard residual) ───
         Temporal bank                          src/temporal_bank.py
         ├── Trajectory markers from            (+paradox flag:
         │   chief_complaint_raw                 severe-cohort + low NEWS2
         ├── Cohort-conditional NEWS2           → 3.4× ESI-2 enrichment)
         │   deviation (the appendicitis
         │   paradox lens)
         └── Chronic / worsening flags
         
         Clinician style                        src/clinician_style.py
         ├── Per-nurse ESI-distribution L1     (modest on synthetic,
         │   deviation from population          load-bearing on real data)
         └── Per-site style vectors
         
         Confidence calibrator                  src/clinician_style.py
         Leakage-as-calibrator: isotonic       (monotonic 14%→88% severe-
         severe-outcome probability from       outcome rate across 7 bins)
         disposition + ed_los_hours at
         training time only.
         
         Aggregated into per-patient
         ClinicianReport                        src/clinician_output.py
         with to_llm_context() fragment
         containing signed phase deviations
         + flags + calibrated severity.
  │
  ▼
Scale 3  LLM residual — AST-style typed        src/triage_contract.py
         contract.                               src/answer_certifier.py
         Fires only when model top-2            src/llm_residual.py
         probability gap < 0.20 (~3%).
         
         1. Render TriagePacket →               (frozen dataclasses:
            deterministic prompt via             PatientClinical, SymbolicVerdict,
            template substitution.               TierBContext, BankReading)
         2. Call LLM with JSON-mode schema.     (closed-enum ESILevel,
         3. Parse into TriageDecision           AlignmentChoice,
            (typed, closed-vocab enums).         DecisiveEvidence)
         4. AnswerCertifier runs 5 checks:      (esi_in_candidates,
            hard-floor, evidence-packet          hard_floor_respected,
            consistency, alignment-bank          alignment_consistent,
            consistency, candidate integrity.    evidence_consistent, etc.)
         5. Rejected outputs leave
            ensemble prediction unchanged.
```

## Data files (as-received)

| File | Rows | Cols | Notes |
|---|---|---|---|
| `train.csv` | 80,000 | 40 | Includes `disposition`, `ed_los_hours` (leakage) and `triage_acuity` (target) |
| `test.csv` | 20,000 | 37 | No target, no leakage cols |
| `chief_complaints.csv` | 100,000 | 3 | `chief_complaint_raw` + undocumented `chief_complaint_system` column |
| `patient_history.csv` | 100,000 | 26 | 25 `hx_*` binary comorbidity flags — undocumented on data-description page |
| `sample_submission.csv` | 20,000 | 2 | All values ESI=3 |

Missingness is **informative**: `systolic_bp` is never NaN for ESI 1-3 but
NaN in ~12% of ESI 4-5 rows. The pipeline exposes `bp_missing` and
`n_vitals_missing` as features rather than imputing silently.

## Leakage guards

- `src/feature_engine.py` drops `disposition`, `ed_los_hours`, `triage_acuity`
  and asserts the first two are gone.
- `ed_los_hours` correlates at r=-0.757 with the target — anyone who keeps
  it wins on train and overfits catastrophically on panel judging.
- Competitor `udaken10` wrote a whole notebook called "no more leak" on
  the same issue. This is the dataset's most obvious trap.

## Key design decisions

1. **Hard rules only where clinically deterministic.** GCS ≤ 8 → ESI 1 and
   cardiac-arrest complaint → ESI 1 are the only hard rules. Shock index,
   SpO2, and thermal extremes contribute as bank signals only; they can be
   ESI 2 rather than ESI 1 in the data, so making them deterministic would
   lose precision.
2. **Bank confidence is continuous; constraints are discrete.** Each bank
   emits `esi_estimate ∈ [1.0, 5.0]`, `confidence ∈ [0,1]`, and
   `(floor, ceiling) ∈ {1..5, 0|6}`. Cross-bank coherence is
   confidence-weighted variance; floor/ceiling clamps prevent averaging
   across a critical signal.
3. **Bank signals are model features.** Scale 1 doesn't throw away the
   Scale 0 decomposition — each patient's 11 bank ESI estimates + 11 bank
   confidences feed the ensemble as features. The model learns when to
   override the geometric prior.
4. **Arrival bank neutralized.** Empirically all arrival modes have mean
   ESI 3.32 ± 0.01 — confirmed across 80K rows. Bank confidence set to
   0.05 so it adds no noise.
5. **Kuramoto phase-deviation features are first-class.** Each bank's
   signed phase deviation from the confidence-weighted consensus is
   encoded (11 `_dev` columns + 3 subset-r columns + scalar `bank_r_total`).
   Foundation analysis found `dev_demographic` MI=0.62 — higher than any
   raw bank ESI estimate. On the hard residual (bank-consensus off by ≥1
   ESI), these features *nearly double* their MI vs. global
   (`dev_consciousness` 0.25 → 0.48). Conditional-MI validated.
6. **Leakage columns become training-time calibrators, not features.**
   `disposition` and `ed_los_hours` are post-triage outcomes. They're
   excluded from the feature matrix (LEAKAGE_COLS assert). But at training
   time we use them to fit an isotonic confidence calibrator — severe-
   outcome rate is strictly monotonic in pipeline coherence confidence
   (14% → 88% across 7 bins). The calibrator feeds the Tier-B context,
   never the predictions themselves.
7. **Scale 3 LLM is narrowed to choice-making, not reasoning.** The LLM
   receives a deterministic rendered prompt from a typed `TriagePacket`
   and must emit JSON conforming to a closed-enum schema. Its private
   free-text reasoning is stored for audit but never parsed into the
   submission. A deterministic `AnswerCertifier` runs 5 consistency checks
   and rejects LLM output that violates hard floors or attributes evidence
   not present in the packet.

## What we deliberately didn't do

- **No TF-IDF / BERT on chief complaint text.** Top public notebooks
  (`kimseunghwan9823`, `dhruvjain35`, `udaken10`) all use some text
  embedding. We use lexicon + keyword groups only. This is a deliberate
  choice toward auditability and transferability, not a gap.
- **Declined shortcut: no target encoding on base condition.** The
  chief-complaint text has a near-deterministic mapping to ESI in this
  synthetic dataset (2,296 unique base conditions, target std = 0.0 within
  every common condition). A fold-safe target encoding on the base slug
  reached QWK ≈ 0.999 / mF1 ≈ 0.995 on CV — essentially a label lookup.
  Cross-fold encoding agreement of ρ = 0.991 confirmed this is structural
  to the dataset, not a fold-safety bug. We removed the feature because
  "chest pain" in a real ED ranges ESI 1–4 and a pipeline that exploits
  the simulator artifact transfers to nothing. The COMPLAINT bank signal
  from `complaint_lexicon.py` — regex patterns like `cardiac arrest` →
  ESI 1 confidence 0.98 — is the honest path that uses complaint text
  while remaining portable to real clinical data.
- **No XGBoost.** Competitors use XGBoost alongside LightGBM + CatBoost;
  we keep the ensemble at 5 models to stay simple.
- **No disposition-aware post-hoc *prediction* calibration.** Tempting
  given `ed_los_hours` correlation with target — forbidden by leakage for
  features. We *do* use it for training-time confidence calibration (see
  key design #6) since the calibrator outputs a *severity probability*,
  not a triage label.

## Negative findings (kept because they are informative)

- **Compositional banks don't work on this dataset.** Three proposed
  multiplicative composites (`AgeNormalizedShock = shock_index × (age/40)^0.5 × (1 + gcs_deficit/15)`,
  `ChronicBurdenResidual = NEWS2 − E[NEWS2 | comorbidity_count, age_group]`,
  `ThermoImmuneProduct = (temp−37)² × (1+2·immuno) × max(0, log(1+meds−3))`)
  all had negative MI lift vs their raw components (−0.024, −0.070, −0.103).
  Conditional-MI on the hard residual also negative. On real clinical data
  these multiplicative interactions are likely meaningful; on this
  synthetic dataset the underlying relationships are largely linear.
- **Trajectory markers have near-zero direct ESI discrimination.** The
  synthetic data embeds `worsening`, `onset today`, `for N days`, `constant`,
  etc. in the complaint text. Across 80K rows, mean ESI per trajectory
  class is 3.30 – 3.34 for all except `chronic` (4.41). The markers are
  kept as features anyway, because the temporal-paradox flag (severe
  complaint cohort + low own NEWS2) does earn its keep: 60 patients
  flagged, 34 are ESI 2 — 3.4× population enrichment — which is a
  valuable red-flag signal for the clinician-facing output even if it
  doesn't move aggregate CV.
- **Clinician style is real but modest on synthetic data.** 50 nurses,
  L1 distribution deviation mean 3.1%, max 6.0%. Five sites, max deviation
  1.8%. Style features are kept because they're load-bearing on real
  clinical data (documented inter-rater variability is 10–15%), but their
  direct CV contribution here is marginal. They also serve the writeup's
  clinician-facing narrative — the "would a different nurse have called
  this differently" counterfactual.
- **Label-noise detection returns ≈ 0 on synthetic data.** The heuristic
  (label ∈ {4,5} AND severe disposition AND ed_los_hours ≥ 6h) flags only
  1 of 80,000 training rows. The simulator is effectively noiseless. On
  real data the same heuristic would find the documented 1–5% inter-rater
  errors.

These negative findings matter because they distinguish what's specific
to the synthetic dataset from what's architecturally real. An
identical pipeline trained on MIMIC-IV-ED would likely show the
compositional banks and trajectory signals pulling more weight — which
is exactly the point of the Tier-B layer being there at all.

## Auditability guarantees

Every submitted prediction can be traced to one of three provenance
classes:

1. **Hard-rule predictions (575 patients, 2.9%).** Triggering bank signal
   + confidence + evidence string stored in `submission_audit.json`.
   Fully deterministic; reproducible byte-for-byte from the code.
2. **Ensemble + QWK predictions (19,396 patients, 97.0%).** 124 feature
   values per patient (post-ablation: phase deviations and all Tier-B
   features except `temporal_news2_deviation` dropped), serialized
   ensemble probabilities, applied QWK thresholds. Reproducible given
   fixed seeds.
3. **LLM-certified predictions (538 patients, 2.7%).** Typed
   `TriageDecision` object captured per patient. Each has:
   - `esi_choice`: from a closed 2-element candidate set
   - `alignment`: from a closed enum naming which bank (or consensus) the
     LLM sided with
   - `decisive_evidence`: 1–3 categories from a closed vocabulary of 25
     clinical evidence classes
   - `reasoning_summary`: free-text audit trail, ≤240 chars, **never
     parsed** into the submission
   - `CertificationResult`: which of the 5 deterministic checks passed or
     failed and why

**What is deliberately NOT exposed:** the LLM's private chain-of-thought,
any free-form justification, and any unconstrained text that could
influence a submitted decision. This is not a limitation — it is the
design. Human triage nurses are not cognitively transparent either;
what we document about them is the *choice* they made and the
*observation* they attributed it to. The typed contract brings the AI
layer to the same auditability bar, with the closed vocabulary giving us
something human audits rarely have: machine-enforceable consistency
between the cited evidence and the patient record.

## File map

```
triagegeist/
├── ARCHITECTURE.md              ← this file (as-built)
├── CONTEXT.md                   ← compass snapshot (auto-generated)
├── .claude/CLAUDE.md            ← project-local instructions for Claude
├── src/
│   ├── banks.py                 ← 11 clinical banks, per-patient ESI/conf signals
│   ├── coherence.py             ← hard rules + Kuramoto-style cross-bank coherence
│   ├── complaint_lexicon.py     ← 65 regex patterns in 3 tiers + 19 modifiers
│   ├── feature_engine.py        ← 117-feature builder with leakage guards
│   ├── model.py                 ← CatBoost CV training utility
│   ├── pipeline.py              ← benchmark-mode orchestrator (training CV)
│   ├── qwk_optimizer.py         ← Nelder-Mead QWK threshold optimizer
│   └── llm_residual.py          ← Ollama/Qwen version of Scale 3 (Claude version lives in submission/)
├── submission/
│   ├── generate_final.py        ← CANONICAL submission generator (all 4 scales)
│   ├── generate_submission.py   ← legacy script (pre-Claude), retained for reference
│   ├── export_uncertain.py      ← builds uncertain_cases.json for Scale 3
│   ├── merge_decisions.py       ← combines Claude batch outputs
│   ├── submission.csv           ← current final submission
│   ├── submission_audit.json    ← per-patient method + evidence
│   ├── decisions_batch_*.json   ← Claude Scale 3 decisions (7 batches, 662 patients)
│   ├── baseline_preds.json      ← model+QWK predictions before Claude
│   └── WRITEUP.md               ← in-submission writeup (more technical)
├── documents/
│   ├── PROJECT_WRITEUP.md       ← Kaggle-submission writeup (2000-word cap)
│   └── NOTEBOOK_COMPANION.md    ← methodology appendix
├── data/extracted/              ← competition CSVs (gitignored)
├── competitor_subs/             ← top public notebook predictions for triangulation
└── catboost_info/               ← CatBoost training artifacts
```

## Provenance

This pipeline ports the multi-scale residual-decomposition pattern from the
NCEMS proteomics-metadata competition in this repo. The ontology is
different (clinical axes instead of SDRF columns), the metric is different
(ordinal kappa instead of exact-match F1), but the pattern —
**decompose → resolve what you can deterministically → hand the residual
to a model → hand the model's residual to an LLM** — is the same.
