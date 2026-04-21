# triagegeist-code

Companion code + artifacts Dataset for the **TriageGeist** Kaggle competition notebook.

## Contents

- **`src/`** — pipeline modules imported by the notebook at inference time
  (11-bank decomposition, coherence scoring, complaint lexicon, feature engine,
  ensemble wrapper, QWK threshold optimizer, LLM residual, AnswerCertifier,
  TriagePacket schema)
- **`models/`** — pre-trained ensemble weights and fitted preprocessors:
  - `cb_seed{42,123,777}.cbm` — 3 CatBoost models (2000 iters each)
  - `lgb_seed{42,314}.txt` — 2 LightGBM models (1500 estimators each)
  - `feature_columns.json`, `cat_columns.json` — feature schema (column order + categorical columns)
  - `cohort_expectations.pkl` — fitted cohort table for `temporal_news2_deviation`
  - `qwk_thresholds.json` — pre-optimized ordinal thresholds
  - `hard_rules.json` — cached Scale 0 rule firings
  - `meta.json` — training metadata (seeds, blend weights, timing, feature count)
  - `test_proba.npy`, `test_patient_ids.npy` — cached ensemble probabilities for verification
- **`train_and_save.py`** — the one-shot training script that produced `models/`
- **`analysis/`** — the benchmark + OOF analysis scripts that underpin the writeup's
  evidence (including the 12-variant Tier-B forensic ablation)
- **`decisions_batch_*.json`** — 7 cached LLM decision files covering the 662
  patients consulted by the Scale 3 LLM (29 final LLM-determined ESI assignments)
- **`submission.csv`, `submission_audit.json`** — reference outputs for verification

## Using this dataset

The companion notebook attaches `/kaggle/input/triagegeist-code/` and runs the
full multi-scale pipeline end-to-end in under a minute, loading the pre-trained
ensemble from `models/` (no retraining).

To retrain from scratch (not required for inference): run `python3 train_and_save.py`
locally against the competition data.

License: same as the competition (Non-Commercial Research, Laitinen-Fredriksson Foundation).
GitHub source: see the writeup's Public Project Link.
