# triagegeist-code

Companion code Dataset for the **TriageGeist** Kaggle competition notebook.

Contains:

- `src/` — the full TriageGeist pipeline (11-bank decomposition, coherence scoring, complaint lexicon, feature engineering, ensemble, QWK threshold optimizer, LLM residual, AnswerCertifier, TriagePacket schema)
- `models/` — pre-trained ensemble weights and inference artifacts:
  - `cb_seed{42,123,777}.cbm` — 3 CatBoost models (2000 iters each)
  - `lgb_seed{42,314}.txt` — 2 LightGBM models (1500 estimators each)
  - `feature_columns.json`, `cat_columns.json` — feature schema
  - `cohort_expectations.pkl` — fitted cohort table for `temporal_news2_deviation`
  - `qwk_thresholds.json` — pre-optimized ordinal thresholds
  - `hard_rules.json` — cached Scale 0 rule firings
  - `meta.json` — training metadata (seeds, blend weights, timing)
- `decisions_batch_*.json` — 7 cached LLM decision files containing the 29 final LLM-determined ESI assignments
- `submission.csv`, `submission_audit.json` — reference outputs for verification

The companion notebook attaches both this Dataset and the `triagegeist` competition data, then runs the full multi-scale pipeline end-to-end.

License: same as the competition (Non-Commercial Research, Laitinen-Fredriksson Foundation).

GitHub source: see the writeup's "Public Project Link" field.
