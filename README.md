# TriageGeist

Auditable Emergency Severity Index (ESI) triage decision-making for the Triagegeist Kaggle competition (Laitinen-Fredriksson Foundation, 2026).

The thesis is in [`documents/PROJECT_WRITEUP.md`](documents/PROJECT_WRITEUP.md). The runnable end-to-end pipeline is in [`submission/triagegeist_notebook.ipynb`](submission/triagegeist_notebook.ipynb).

## Architecture in one paragraph

ESI triage decomposes into three epistemic regimes — protocol execution, common pattern, genuine clinical judgment — each served by inference machinery whose audit cost matches its character. **2.88%** of test patients resolve via a hard-rule decision tree. **96.98%** resolve via a deterministic CatBoost + LightGBM ensemble with QWK threshold optimization. **0.14%** route to an LLM under a typed contract — closed-vocabulary JSON output, independent certifier, free-text reasoning logged but never parsed into the prediction. **99.86% of decisions involve zero LLM reasoning.**

## Repository layout

```
src/                     11-bank decomposition, coherence scoring, complaint
                         lexicon, feature engine, ensemble, QWK optimizer,
                         LLM residual, AnswerCertifier, TriagePacket schema
documents/
  PROJECT_WRITEUP.md     The 2,000-word competition writeup
  NOTEBOOK_COMPANION.md  Implementation-detail companion
analysis/
  oof_evidentiary.py     Regenerates 5-fold OOF + computes the three
                         evidentiary numbers (clinical safety, error
                         topology, routing calibration)
  integrate_oof.py       Renders fig 2/3 and substitutes OOF placeholders
                         in the writeup
  benchmark_*.py         Forensic ablations (12-variant Tier-B, temporal,
                         phase, surprisal, calibrator)
submission/
  triagegeist_notebook.ipynb   The Kaggle public notebook
  generate_final.py            Reproduces submission.csv from scratch
  submission.csv               Final submission (20,000 rows, format-validated)
  submission_audit.json        Per-prediction provenance trail
  decisions_batch_*.json       Cached LLM decisions for the 29 routed cases
figures/
  style.py                     Editorial chassis — palette, rcParams,
                               header/footer, name normalization
  fig1_auditability_triangle.py   Hero — three regimes
  fig2_disagreement_topology.py   OOF confusion matrix + safety zones
  fig3_routing_calibration.py     Architecture self-diagnosis
  fig4_ablation_tornado.py        Where the +0.0168 Tier-B lift lives
  cover.py                       560×280 cover image
  preview_dualbg.py              Composite each PNG against Kaggle
                                 light + dark page backgrounds
SUBMISSION_CHECKLIST.md     What the user has to do before submitting
```

## Reproducing

The data files are not in this repo; download from Kaggle via:
```bash
kaggle competitions download -c triagegeist -p data/
unzip data/triagegeist.zip -d data/extracted/
```

End-to-end submission regeneration:
```bash
python3 submission/generate_final.py
```

OOF analysis (used in the writeup):
```bash
python3 analysis/oof_evidentiary.py
python3 analysis/integrate_oof.py
```

Figure suite:
```bash
cd figures/
for f in fig*.py cover.py; do python3 $f; done
python3 preview_dualbg.py     # visual QA against both Kaggle backgrounds
```

## License

This project is for the Triagegeist competition. The competition data is under the Laitinen-Fredriksson Foundation's Non-Commercial Research License — see the competition page for details.
