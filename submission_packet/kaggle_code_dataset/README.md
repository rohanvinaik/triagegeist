# triagegeist-code

Companion code Dataset for the **TriageGeist** Kaggle competition notebook.

Contains:

- `src/` — the full TriageGeist pipeline (11-bank decomposition, coherence scoring, complaint lexicon, feature engineering, ensemble, QWK threshold optimizer, LLM residual, AnswerCertifier, TriagePacket schema)
- `decisions_batch_*.json` — 7 cached LLM decision files containing the 29 final LLM-determined ESI assignments

The companion notebook attaches both this Dataset and the `triagegeist` competition data, then runs the full multi-scale pipeline end-to-end.

License: same as the competition (Non-Commercial Research, Laitinen-Fredriksson Foundation).

GitHub source: see the writeup's "Public Project Link" field.
