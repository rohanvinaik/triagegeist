# Kaggle Deployment Guide

End-to-end steps to submit this package to the Triagegeist competition. Allow ~30 minutes of human work. The notebook itself runs in under a minute on Kaggle CPU — training is done once locally and ships as pre-trained weights in the `triagegeist-code` companion Dataset.

## Step 1 — Push to GitHub (5 min)

The whole project (one level up from this packet) needs to be on a public GitHub repo so the writeup's "Public Project Link" can point to it.

```bash
cd /Users/rohanvinaik/Projects/Kaggle_Killer/competitions/triagegeist

# git init was already run by Prism setup
git add -A
git commit -m "TriageGeist competition submission"

# Create a public repo on github.com first, then:
git remote add origin https://github.com/<your-username>/triagegeist.git
git push -u origin main
```

Verify the repo is **Public** in github.com → repo settings.

## Step 2 — Upload code as a Kaggle Dataset (5 min)

The notebook depends on `src/` (the pipeline modules) and the cached LLM decisions. Standard Kaggle pattern: package both as a Dataset.

```bash
# Compress the kaggle_code_dataset folder
cd /Users/rohanvinaik/Projects/Kaggle_Killer/competitions/triagegeist/submission_packet
zip -r triagegeist-code.zip kaggle_code_dataset
```

Then on Kaggle.com:
1. **Datasets → New Dataset**
2. Upload `triagegeist-code.zip`
3. Title: `triagegeist-code`  (URL slug must be exactly this so the notebook finds it at `/kaggle/input/triagegeist-code/`)
4. Visibility: **Public**
5. Subtitle: `TriageGeist pipeline code and cached LLM decisions`
6. **Create**

## Step 3 — Create the public Kaggle Notebook (10 min)

1. Competition page (kaggle.com/competitions/triagegeist) → **Code → New Notebook**
2. **File → Upload Notebook** → pick `notebook/triagegeist.ipynb` from this packet
3. **Settings → Add Data → Competition Data**: search `triagegeist`, attach
4. **Settings → Add Data → Datasets**: search `triagegeist-code` (the one you just uploaded), attach
5. **Settings → Accelerator → CPU** (no GPU needed — pipeline is CPU-only)
6. **Save Version → Save & Run All**
7. Runs in under a minute (loads the 5 pre-trained model files from `triagegeist-code`, runs inference + QWK thresholds + cached LLM decisions, writes `submission.csv`)\n
8. Verify the final cell printed `submission.csv written.` with the expected method breakdown (575 rules / 19396 ensemble / 29 LLM-certified)
9. Once the run is green: **Share → Public**

Copy the notebook URL — you'll need it in step 4.

## Step 4 — Create the Writeup and submit (10 min)

1. Competition page → **Writeups → New Writeup**
2. **Title**: `TriageGeist: Auditable triage decision-making at the cost of compute it deserves`
3. **Track**: `Triagegeist: AI in Emergency Triage` (the only track)
4. **Body**: paste the full contents of `WRITEUP.md`
5. **Inline figures**: the writeup references four figures via relative paths (`../figures/fig1_*.png` etc). For each one:
   - Click the image-upload button in the editor
   - Upload the corresponding file from `figures/`
   - Replace the markdown image link with the inlined Kaggle URL
6. **Cover Image**: upload `cover.png` (the editor verifies the 560×280 size)
7. **Attached Public Notebook**: paste the URL from step 3
8. **Public Project Link**: paste the GitHub URL from step 1
9. Read it through one more time. Word count is 1,831 / 2,000 — the limit is enforced.
10. **Save** then **Submit**

**Drafts are not judged.** Verify the Writeups tab shows your submission as Submitted, not Draft.

## Sanity checks before clicking Submit

- [ ] Notebook is **Public** (`Share → Public`)
- [ ] Notebook ran end-to-end without errors and emitted `submission.csv`
- [ ] Code Dataset (`triagegeist-code`) is **Public**
- [ ] GitHub repo is **Public**
- [ ] All four figures inlined (not broken links) in the writeup
- [ ] Cover image attached
- [ ] Notebook URL + GitHub URL both attached to the writeup
- [ ] Track selected
- [ ] Status shows **Submitted**, not Draft

## Common gotchas

- **Notebook fails on Kaggle with "module 'src' not found"** — the `triagegeist-code` Dataset wasn't attached. Re-add via Settings → Add Data.
- **Notebook fails with "decisions_batch_X.json not found"** — same fix; the `triagegeist-code` Dataset bundles these.
- **Cover image rejected** — Kaggle is strict about the 560×280 dimensions. The `cover.png` here is verified at exactly 560×280; if Kaggle rejects, re-upload.
- **Writeup over word limit** — only the body counts. References and headers are included. Current count is 1,831; safe.
- **Image links broken in writeup** — the relative paths only work locally. You must upload each figure into the writeup editor and replace the link.
