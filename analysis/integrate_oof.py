#!/usr/bin/env python3
"""
Integration step: read analysis/oof_summary.json, render Fig 2 + Fig 3,
substitute the ⟪PLACEHOLDER⟫ markers in submission_packet/WRITEUP.md with
the formal OOF numbers, and emit a final-deliverables manifest.

Run AFTER analysis/oof_evidentiary.py has completed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = PROJECT_ROOT / "analysis"
SUBMISSION_PACKET = PROJECT_ROOT / "submission_packet"
FIGURES = PROJECT_ROOT / "figures"


def fmt_pct(p: float, decimals: int = 2) -> str:
    return f"{p * 100:.{decimals}f}"


def main():
    summary_path = ANALYSIS / "oof_summary.json"
    if not summary_path.exists():
        print(f"missing {summary_path} — run oof_evidentiary.py first")
        sys.exit(1)
    s = json.loads(summary_path.read_text())

    # --- Render fig 2 + fig 3 ---
    print("Rendering fig 2 + fig 3 ...")
    for f in ["fig2_disagreement_topology.py", "fig3_routing_calibration.py"]:
        r = subprocess.run([sys.executable, f], cwd=FIGURES,
                           capture_output=True, text=True)
        print(f"  {f}: {r.stdout.strip()}")
        if r.returncode != 0:
            print(f"  STDERR: {r.stderr.strip()}")
            sys.exit(1)

    # Re-render dual-bg previews
    subprocess.run([sys.executable, "preview_dualbg.py"], cwd=FIGURES,
                   capture_output=True, text=True)

    # --- Substitute placeholders in writeup ---
    writeup = SUBMISSION_PACKET / "WRITEUP.md"
    text = writeup.read_text()
    subs = {
        "⟪OOF_PCT_GE_2_PCT⟫": f"{fmt_pct(s['pct_ge_2'], 3)}",
        "⟪OOF_N_GE_2⟫":       f"{s['n_ge_2']:,}",
        "⟪OOF_MACRO_F1⟫":     f"{s['macro_f1']:.4f}",
        "⟪OOF_ACCURACY⟫":     f"{s['accuracy']:.4f}",
        "⟪OOF_QWK⟫":          f"{s['qwk']:.4f}",
        "⟪OOF_PCT_ERR_3_4⟫":  f"{fmt_pct(s['pct_err_on_3_4_boundary'], 1)}",
        "⟪OOF_PCT_ERR_2_3⟫":  f"{fmt_pct(s['pct_err_on_2_3_boundary'], 1)}",
    }
    n_replaced = 0
    for k, v in subs.items():
        if k in text:
            text = text.replace(k, v)
            n_replaced += 1
            print(f"  {k} -> {v}")
    writeup.write_text(text)
    print(f"writeup substitutions: {n_replaced}/{len(subs)}")

    # --- Word count check ---
    words = len(re.findall(r"\b[\w'-]+\b", text))
    print(f"writeup word count: {words}  (limit: 2,000)")

    # --- Manifest ---
    print("\n=== DELIVERABLES MANIFEST ===")
    files = [
        ("Writeup",                 SUBMISSION_PACKET / "WRITEUP.md"),
        ("Notebook",                SUBMISSION_PACKET / "notebook" / "triagegeist.ipynb"),
        ("Submission CSV",          SUBMISSION_PACKET / "submission.csv"),
        ("Audit JSON",              SUBMISSION_PACKET / "submission_audit.json"),
        ("Cover image (560x280)",   FIGURES / "cover.png"),
        ("Fig 1 (hero)",            FIGURES / "fig1_auditability_triangle.png"),
        ("Fig 2 (disagreement)",    FIGURES / "fig2_disagreement_topology.png"),
        ("Fig 3 (routing)",         FIGURES / "fig3_routing_calibration.png"),
        ("Fig 4 (ablation)",        FIGURES / "fig4_ablation_tornado.png"),
        ("OOF summary",             ANALYSIS / "oof_summary.json"),
    ]
    for name, path in files:
        ok = "✓" if path.exists() else "✗"
        size = f"{path.stat().st_size:,} B" if path.exists() else "MISSING"
        print(f"  {ok} {name:30s} {size:>12}  {path.relative_to(PROJECT_ROOT)}")

    print("\n=== KEY NUMBERS ===")
    for k in ("n_total", "exact_match", "off_by_1", "off_by_2", "off_by_3",
              "off_by_4", "pct_exact", "pct_within_1", "pct_ge_2",
              "macro_f1", "accuracy", "qwk", "linear_kappa",
              "n_llm_routed", "pct_llm_routed",
              "pct_err_on_3_4_boundary", "pct_err_on_2_3_boundary"):
        print(f"  {k:30s} {s[k]}")


if __name__ == "__main__":
    main()
