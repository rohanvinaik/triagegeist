# TriageGeist — Submission Packet

This folder contains everything needed to submit to the **Triagegeist** Kaggle competition (Laitinen-Fredriksson Foundation, 2026). Deadline: **2026-04-21 22:00 UTC**.

## What's in this folder

```
submission_packet/
├── README.md                      ← you are here
├── KAGGLE_DEPLOYMENT.md           ← exact step-by-step submission instructions
├── WRITEUP.md                     ← the 1,831-word competition writeup
├── cover.png                      ← 560×280 cover image (verified dimensions)
├── submission.csv                 ← final submission (20,000 rows, format-validated)
├── submission_audit.json          ← per-prediction provenance trail
├── figures/
│   ├── fig1_auditability_triangle.png      hero — three regimes of decision-making
│   ├── fig2_disagreement_topology.png      OOF confusion matrix + clinical-safety zones
│   ├── fig3_routing_calibration.png        architecture self-diagnoses uncertainty
│   └── fig4_ablation_tornado.png           where the +0.0168 Tier-B QWK lift lives
├── notebook/
│   └── triagegeist.ipynb          ← public Kaggle notebook, end-to-end runnable
└── kaggle_code_dataset/           ← upload as a Kaggle Dataset named "triagegeist-code"
    ├── README.md
    ├── src/                       ← all pipeline modules (4,084 LOC)
    └── decisions_batch_*.json     ← 7 cached LLM decisions files (29 final decisions)
```

## Required Kaggle deliverables (per the Evaluation page)

| Required | Where to find it | Status |
|---|---|---|
| Public Notebook | `notebook/triagegeist.ipynb` | ✅ self-contained, runs end-to-end |
| Project Writeup | `WRITEUP.md` | ✅ 1,831 / 2,000 word limit |
| Cover Image (560×280) | `cover.png` | ✅ size verified |
| Public Project Link | (your GitHub repo URL — see `KAGGLE_DEPLOYMENT.md`) | ⚠️ user action |
| Submission CSV | `submission.csv` | ✅ row order matches `test.csv`, integers 1–5, no nulls |

## How to submit

See `KAGGLE_DEPLOYMENT.md`. Three steps:

1. Push the project (one directory up from this packet) to a public GitHub repo.
2. Upload `kaggle_code_dataset/` as a Kaggle Dataset named `triagegeist-code`.
3. Create the Kaggle Notebook + Writeup using the assets in this folder.

## Headline numbers from the project

- **0.011%** of OOF predictions deviate by ≥2 ESI levels (9 of 80,000)
- **0.004%** of OOF predictions are clinically-asymmetric under-triage (3 of 80,000)
- **77.4%** of OOF errors sit on the lowest-acuity ESI 4↔5 boundary; only **6.7%** involve high-acuity boundaries
- **99.86%** of test decisions involve zero LLM reasoning
- **0.14%** of test decisions are LLM-determined under a typed contract with independent certifier
- OOF macro F1 0.9756 / accuracy 0.9777 / quadratic-weighted kappa 0.9895

These numbers are computed by `analysis/oof_evidentiary.py` (in the GitHub repo); the writeup, figures, and notebook all reference the same `analysis/oof_summary.json`.
