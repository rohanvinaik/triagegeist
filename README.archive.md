# TriageGeist

Auditable multi-scale Emergency Severity Index (ESI) triage decision-making for the Triagegeist Kaggle competition (Laitinen-Fredriksson Foundation, 2026).

The thesis and all competition deliverables are in [`submission_packet/`](submission_packet/). The writeup is [`submission_packet/WRITEUP.md`](submission_packet/WRITEUP.md). The runnable end-to-end notebook is [`submission_packet/notebook/triagegeist.ipynb`](submission_packet/notebook/triagegeist.ipynb).

## Architecture in one paragraph

ESI triage decomposes into three epistemic regimes — protocol execution, common pattern, genuine clinical judgment — each served by inference machinery whose audit cost matches its character. **2.88%** of test patients resolve via a hard-rule decision tree. **96.98%** resolve via a deterministic CatBoost + LightGBM ensemble with QWK threshold optimization. **0.14%** route to an LLM under a typed contract — closed-vocabulary JSON output, independent certifier, free-text reasoning logged but never parsed into the prediction. **99.86% of decisions involve zero LLM reasoning.**

## Repository layout

```
src/                       pipeline modules (11-bank decomposition, coherence
                           scoring, complaint lexicon, feature engine, ensemble,
                           QWK optimizer, LLM residual, AnswerCertifier,
                           TriagePacket schema)
analysis/                  benchmarks + OOF evidence generator
  oof_evidentiary.py       regenerates 5-fold OOF + the three evidentiary
                           numbers used in the writeup
  integrate_oof.py         substitutes OOF placeholders + renders fig 2 / 3
  benchmark_*.py           12-variant Tier-B forensic ablations
figures/                   editorial chassis + 4 figure scripts + cover + dual-mode previewer
submission/
  train_and_save.py        one-shot training entrypoint — runs once, saves
                           every inference artifact to submission/models/
  models/                  trained ensemble weights + feature schema + cohort
                           table + QWK thresholds + metadata
                           (regenerable; written by train_and_save.py)
submission_packet/         canonical shipping folder (everything the Kaggle
                           submission needs)
  WRITEUP.md               2,000-word writeup (rubric-ready)
  cover.png                560×280 cover image
  submission.csv           final predictions (20,000 rows, format-validated)
  submission_audit.json    per-prediction provenance trail
  figures/                 the four figures inlined into the writeup
  notebook/
    triagegeist.ipynb      public Kaggle notebook — load-and-infer (<1 min)
  kaggle_code_dataset/     staging folder uploaded as the `triagegeist-code`
                           Kaggle Dataset (src/ + models/ + decisions + README)
  README.md                packet-level overview
  KAGGLE_DEPLOYMENT.md     step-by-step Kaggle submission guide
tests/                     unit tests for the pipeline modules
```

## Reproducing

Competition data is not in the repo; download from Kaggle:
```bash
kaggle competitions download -c triagegeist -p data/
unzip data/triagegeist.zip -d data/extracted/
```

Train once. Writes every inference artifact into `submission/models/`:
```bash
python3 submission/train_and_save.py
```

OOF analysis for the writeup:
```bash
python3 analysis/oof_evidentiary.py
python3 analysis/integrate_oof.py
```

Figure suite:
```bash
cd figures/
for f in fig*.py cover.py; do python3 $f; done
python3 preview_dualbg.py
```

## License

For the Triagegeist competition. Data is under the Laitinen-Fredriksson Foundation Non-Commercial Research License — see the competition page.
