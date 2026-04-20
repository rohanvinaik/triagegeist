# Notebook Companion: Technical Methodology

This document accompanies the TriageGeist submission notebook and provides technical detail on the multi-scale pipeline architecture.

## Pipeline Overview

```
Patient Data
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Scale 0: Bank Decomposition + Hard Rules       │
│  11 orthogonal clinical banks → ESI estimates   │
│  Hard rules: GCS ≤ 8 → ESI 1 (deterministic)   │
│  Output: 575/20000 resolved (2.9%)              │
└────────────────────┬────────────────────────────┘
                     │ residual (97.1%)
                     ▼
┌─────────────────────────────────────────────────┐
│  Scale 1: CatBoost + LightGBM Ensemble          │
│  124 features: 22 bank ESI+conf, bank coherence │
│  scalars (r_total, psi, 3 subset r,             │
│  coherence_spread), raw vitals + demographics,  │
│  25 hx_* flags, 12 text keyword flags, 18       │
│  clinical interactions, 1 Tier-B feature        │
│  (temporal_news2_deviation). Phase-deviation    │
│  features dropped per forensic ablation;        │
│  rendered into LLM audit context instead.       │
│  3 CatBoost (2000 iter) + 2 LightGBM (1500 est) │
│  60/40 weighted average                          │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Scale 2: QWK Threshold Optimization             │
│  Ordinal thresholds: [1.56, 2.52, 3.55, 4.58]  │
│  Output: ~18,900 resolved (94.4%)               │
└────────────────────┬────────────────────────────┘
                     │ uncertain (prob gap < 0.20)
                     ▼
┌─────────────────────────────────────────────────┐
│  Scale 3: LLM Residual (Claude Opus 4.6)         │
│  662 cases reviewed, ~300 predictions changed    │
│  Output: ~538 resolved (2.7%)                    │
└─────────────────────────────────────────────────┘
```

## Bank Decomposition Detail

Each bank is an independent clinical axis producing:
- `esi_estimate`: continuous 1.0-5.0 (calibrated from 80K training examples)
- `confidence`: 0.0-1.0 (how informative this bank is for this patient)
- `esi_floor` / `esi_ceiling`: hard constraints (e.g., GCS ≤ 8 → ceiling = 1)
- `evidence`: human-readable provenance string

### Calibration Table (Clinical Prior → Empirical)

| Signal | Clinical Prior | Empirical Mean | Calibrated |
|--------|---------------|---------------|------------|
| NEWS2 = 0 | ESI 5.0 | 4.23 | 4.2 |
| NEWS2 = 1 | ESI 4.0 | 3.78 | 3.8 |
| NEWS2 = 3-4 | ESI 3.0 | 3.09 | 3.1 |
| NEWS2 = 7-8 | ESI 1.5 | 2.19 | 2.2 |
| NEWS2 = 13+ | ESI 1.0 | 1.52 | 1.5 |
| GCS 15, alert | ESI 5.0 | 3.87 | 3.8 |
| GCS 15, drowsy | ESI 3.0 | 3.22 | 3.0 |
| GCS 15, unresponsive | ESI 2.0 | 2.67 | 1.5 |
| All arrival modes | Variable | 3.32 (±0.01) | 3.3 (neutralized) |

The arrival-mode finding is notable: helicopter, ambulance, walk-in, police, transfer, and family transport all have mean ESI within ±0.02 of 3.32 in this dataset (n ≥ 2,400 per mode). The ARRIVAL bank is neutralized with confidence = 0.05 so it contributes no signal. "GCS 15 unresponsive" is a contradictory state (treated as likely-unreliable data) and the bank deliberately under-calibrates toward safer ESI 1.5.

## Coherence Scoring

Cross-bank coherence uses confidence-weighted variance:

```
weighted_esi = Σ(bank_i.esi × bank_i.confidence) / Σ(bank_i.confidence)
weighted_var = Σ(bank_i.confidence × (bank_i.esi - weighted_esi)²) / Σ(bank_i.confidence)
std_dev = √(weighted_var)
```

Convergence boost (Kuramoto-style):
- std < 0.3: +0.25 confidence (strong agreement)
- std < 0.6: +0.15 (moderate)
- std > 1.5: -0.15 (strong disagreement)

## Constraint Clamps (COEC)

Six cross-bank interaction rules that encode clinical safety logic:

1. **Severity + Consciousness → ESI ≤ 2**: High NEWS2 combined with altered consciousness clamps to emergent
2. **Fever + Immunosuppressed → Sepsis risk**: Escalates regardless of other signals
3. **Chest pain + CV derangement → ESI ≤ 2**: Cardiac rule-out protocol
4. **Dyspnea + Respiratory derangement → ESI ≤ 2**: Respiratory emergency
5. **Coagulopathy + Bleeding → Escalate**: Hemorrhage risk in anticoagulated patients
6. **All normal + Minor complaint → ESI ≥ 4.5**: De-escalation for trivially well patients

## Feature Engineering (124 features)

**Raw structured (37):** Vitals, demographics, arrival info, complaint system

**Text features (12):** Complaint length, word count, 10 keyword group flags (pain, respiratory, cardiac, neuro, gi, psych, trauma, infection, severity_high, severity_low)

**Clinical interactions (18):**
- HR × RR product and ratio (respiratory distress)
- SpO2 - RR gap (compensatory breathing)
- Age × NEWS2 (elderly severity amplification)
- GCS deficit (15 - GCS)
- Temperature deviation from 37°C
- Vital instability count (number of deranged vitals)
- Immunocompromised × fever, cardiac risk × tachycardia, pain × altered mental status
- Missing vitals count, BP missing flag, pain not recorded flag
- Mental status severity (ordinal: alert=0 → unresponsive=4)
- Resource proxy, low acuity score, NEWS2 squared

**Bank signals (22):** 11 banks × (ESI estimate + confidence). Phase-deviation
columns (`bank_*_dev`) and per-subset coherence were emitted by the pipeline
but dropped from the ensemble after fold-safe CV showed zero or negative lift.
Bank-level coherence scalars (`bank_r_total`, `bank_psi`, 3 subset r's) are kept
because they are read by the regime gate and by the Scale-3 LLM audit context.

**Tier-B (1 feature):** `temporal_news2_deviation` = own_NEWS2 −
E[NEWS2 | complaint_base, age_group]. Ablated from every competing feature
family; carries 91% of the +0.017 QWK Tier-B lift. Kept under option 2
(real clinical signal, synthetic amplification disclosed in writeup).

**History aggregates (2):** Total comorbidities, high-risk comorbidity count

## Audit Trail Example

For patient TG-MFR36MWG8 (24M, orbital cellulitis, NEWS2=10, GCS=13, SpO2=99, BP=65/51):

```
BANK SIGNALS:
  severity        : ESI=1.9  conf=0.92  news2=10 very-high
  cardiovascular  : ESI=2.0  conf=0.80  shock_index=1.30 elevated
  pain            : ESI=2.0  conf=0.75  pain=9 abdomen severe
  consciousness   : ESI=2.5  conf=0.75  gcs=13 mildly impaired
  thermal         : ESI=3.0  conf=0.50  temp=39.2 fever
  history         : ESI=2.5  conf=0.45  5 comorbidities, 3 high-risk
  respiratory     : ESI=3.2  conf=0.55  spo2=99 rr=23 mild
  complaint       : ESI=3.8  conf=0.25  system:ophthalmic

COHERENCE:
  Weighted ESI: 2.3 (4 banks converge around ESI 2.0)
  Weighted std: 0.48 (moderate convergence → +0.15 boost)

CONSTRAINT CLAMP:
  severity(1.9) + consciousness(2.5) → ESI ≤ 2 FIRES

PREDICTION: ESI 2 (confidence 0.950)
ACTUAL: ESI 2 ✓

REASONING CHAIN:
  "This patient's NEWS2 of 10 (very high), shock index of 1.30
   (elevated), severe pain, and mildly impaired GCS all converge
   on emergent acuity. The severity + consciousness constraint
   clamp confirms ESI ≤ 2. The ophthalmic complaint system alone
   would suggest lower acuity, but it has low confidence (0.25)
   and is correctly overridden by the physiological signals."
```

This trace is generated automatically for every patient. The full audit trail for all 20,000 test predictions is in `submission_audit.json`.

## Computational Cost

| Component | Hardware | Time (20K patients) | Cost |
|-----------|----------|---------------------|------|
| Bank decomposition | CPU | ~3 seconds | ~$0.00 |
| Complaint lexicon | CPU | ~1 second | ~$0.00 |
| Coherence scoring | CPU | ~2 seconds | ~$0.00 |
| Feature engineering | CPU | ~5 seconds | ~$0.00 |
| CatBoost inference (3 models) | CPU | ~2 seconds | ~$0.00 |
| LightGBM inference (2 models) | CPU | ~1 second | ~$0.00 |
| **Subtotal (97% of patients)** | **CPU** | **~14 seconds** | **~$0.00** |
| LLM residual (~660 patients) | GPU/API | ~20 minutes | ~$0.50 |
| **Total** | | **~20 minutes** | **~$0.50** |

The symbolic pipeline processes 19,400 patients in 14 seconds on a laptop CPU. The LLM handles the remaining 600 genuinely ambiguous cases. This cost structure makes the system viable for deployment in resource-constrained clinical environments.

## Files

| File | Description |
|------|-------------|
| `submission/submission.csv` | Final predictions (20,000 patients) |
| `submission/submission_audit.json` | Full audit trail |
| `submission/WRITEUP.md` | Technical writeup |
| `submission/generate_final.py` | Reproducible pipeline |
| `submission/decisions_batch_*.json` | LLM clinical reasoning |
| `documents/PROJECT_WRITEUP.md` | Competition writeup |
| `documents/NOTEBOOK_COMPANION.md` | This document |
| `src/banks.py` | 11-bank decomposition |
| `src/coherence.py` | Coherence scoring + constraint clamps |
| `src/complaint_lexicon.py` | 65-pattern complaint classifier |
| `src/feature_engine.py` | 117-feature engineering pipeline |
| `src/model.py` | CatBoost + LightGBM ensemble |
| `src/qwk_optimizer.py` | QWK threshold optimization |
| `src/llm_residual.py` | LLM boundary disambiguation |
