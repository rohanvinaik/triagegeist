# triagegeist Context

Predict Emergency Severity Index (ESI) triage acuity levels 1-5 for emergency department patients. This is a Kaggle competition (Triagegeist, $10K prize, 27 teams, deadline April 21 2026). The input is structured tabular data: 80K train / 20K test patients with vitals (BP, HR, RR, temp, SpO2, GCS, pain score, NEWS2 score), demographics (age, sex, language, insurance), arrival info (mode, time, origin), chief complaint text, and 25 binary medical history flags. Train includes 2 leakage columns (disposition, ed_los_hours) that must be excluded. Target is ordinal 1-5 ESI scale with class imbalance (ESI 1=4%, ESI 3=36%).

Architecture: Multi-scale approach ported from NCEMS: (1) Deterministic clinical rules first (NEWS2 thresholds, vital sign cutoffs for ESI classification), (2) Feature engineering with domain knowledge (shock index, clinical flags, complaint×history interactions), (3) Gradient boosting ensemble (CatBoost primary) on structured features, (4) Chief complaint text processing (lexicon mapping + embeddings), (5) Optional LLM residual for ambiguous cases. This beats generic ML because clinical domain knowledge encodes ESI decision rules directly.

Forbidden:
- proposed_rules
- existing_rule_count
- directives_analyzed
- already_covered
