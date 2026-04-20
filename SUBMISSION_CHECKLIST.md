# TriageGeist — Submission Checklist

Competition deadline: **2026-04-21 22:00 UTC** (~2 days from 2026-04-19).

## Required artifacts

| # | Artifact | Path | Status |
|---|---|---|---|
| 1 | Writeup (≤2000 words) | `documents/PROJECT_WRITEUP.md` | ✅ drafted, ⏳ awaiting OOF number substitution |
| 2 | Kaggle Notebook | `submission/triagegeist_notebook.ipynb` | ✅ written, runs end-to-end |
| 3 | Cover image (560×280 px) | `figures/cover.png` | ✅ size verified |
| 4 | Public GitHub repo URL | `(none yet)` | ❌ user action |
| 5 | Submission CSV | `submission/submission.csv` | ✅ 20,000 rows, format-validated |
| — | Audit trail JSON | `submission/submission_audit.json` | ✅ |
| — | Forensic ablation report | `analysis/oof_summary.json` | ⏳ regenerating (CV in progress) |

## Figure suite

All four figures use a shared editorial chassis (`figures/style.py`), transparent PNG output, mid-luminance text that survives both Kaggle light (#F7F7F8) and dark (#20232A) page modes. Verify with `python3 figures/preview_dualbg.py`.

| # | Figure | Role | Status |
|---|---|---|---|
| Hero | `figures/fig1_auditability_triangle.png` | Three regimes of decision-making (the thesis) | ✅ |
| Thesis | `figures/fig2_disagreement_topology.png` | OOF confusion matrix + clinical-safety zones | ⏳ awaiting OOF |
| Evidence | `figures/fig3_routing_calibration.png` | Architecture self-diagnoses uncertainty | ⏳ awaiting OOF |
| Decomp | `figures/fig4_ablation_tornado.png` | Where the +0.0168 Tier-B lift lives | ✅ |
| Cover | `figures/cover.png` | Stripped-down hero, 560×280 | ✅ |

## User-action steps before submitting

1. **GitHub repo**
   ```bash
   git init && git add -A && git commit -m "TriageGeist submission"
   git remote add origin <YOUR_REPO_URL>
   git push -u origin main
   ```
   Make sure the repo is **Public** in GitHub settings.

2. **Kaggle Notebook upload**
   - "Code" tab → "New Notebook" → "File" → "Upload Notebook" → pick `submission/triagegeist_notebook.ipynb`.
   - Attach the Triagegeist competition data (it auto-mounts at `/kaggle/input/triagegeist/`).
   - The notebook also imports from `src/` — either (a) attach a Kaggle Dataset that contains your `src/` directory and uncomment the `sys.path.insert(...)` line, or (b) paste the relevant `src/` modules as preceding cells. Recommend (a).
   - Click "Save Version" → "Save & Run All". Verify it runs to completion.
   - Set notebook visibility to **Public** before deadline.

3. **Writeup creation**
   - Competition page → "Writeups" tab → "New Writeup".
   - Paste contents of `documents/PROJECT_WRITEUP.md`.
   - Upload the four figures via the editor (Kaggle inlines them).
   - Replace the `![Fig N — ...](../figures/figN_*.png)` markdown links with the inlined Kaggle URLs.
   - Attach `figures/cover.png` as the Cover Image.
   - Attach the Public Notebook (link from step 2).
   - Attach the Public Project Link (URL from step 1).
   - Pick a Track (likely "Clinical Decision Support").
   - **Click Submit before the deadline.** Drafts are not judged.

## Final verification commands

After the OOF run completes:

```bash
# Substitute OOF numbers into the writeup, render fig 2/3
python3 analysis/integrate_oof.py

# Verify writeup word count is under 2000
wc -w documents/PROJECT_WRITEUP.md

# Visual QA on every figure
python3 figures/preview_dualbg.py
open figures/_preview/*.png  # macOS — eyeball each
```

## Notes on the calibrated-violation moment (Semiotics §6)

The current writeup has a deliberate sentence in §4 ("Implications for Deployment"):

> *"The customary assumption is that better systems are more expensive. In this corner of clinical AI, that assumption inverts."*

This is the place where the document spends trust capital — the only sentence in the writeup that asserts a domain-wide claim without local evidence in the same paragraph. It works only if §§1–3 have banked enough capital. If you want to swap in something more idiosyncratic, that's the spot, and that sentence is the budget.
