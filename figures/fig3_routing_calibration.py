"""
Fig 3: ensemble confidence vs OOF error rate, with the LLM routing band marked.

The architecture routes a case to the LLM when the ensemble's top-2 probability
gap is below 0.20. The figure shows that this routing rule corresponds to a
genuine spike in OOF error rate — the architecture diagnoses its own
uncertainty correctly. The LLM is invoked only where it is structurally
warranted.

Loads numbers from analysis/oof_*.npy and oof_summary.json. Fails informatively
if the OOF run hasn't completed yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import style
from style import P

style.apply()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = PROJECT_ROOT / "analysis"
DATA_DIR = PROJECT_ROOT / "data" / "extracted"
LLM_GAP = 0.20


def load():
    needed = [ANALYSIS / "oof_proba.npy", ANALYSIS / "oof_final.npy",
              ANALYSIS / "oof_gap.npy", ANALYSIS / "oof_summary.json"]
    for p in needed:
        if not p.exists():
            raise SystemExit(f"missing artifact: {p}. Run oof_evidentiary.py first.")
    return {
        "proba": np.load(ANALYSIS / "oof_proba.npy"),
        "final": np.load(ANALYSIS / "oof_final.npy"),
        "gap": np.load(ANALYSIS / "oof_gap.npy"),
        "summary": json.loads((ANALYSIS / "oof_summary.json").read_text()),
        "y": pd.read_csv(DATA_DIR / "train.csv")["triage_acuity"].values,
    }


def draw():
    d = load()
    gap = d["gap"]
    final = d["final"]
    y = d["y"]
    err = (final != y).astype(float)
    dev = np.abs(final - y)
    severe = (dev >= 2).astype(float)

    # Bin gap into deciles for the rate plot
    edges = np.linspace(0, gap.max() + 1e-6, 21)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bins = np.digitize(gap, edges) - 1
    bins = np.clip(bins, 0, len(edges) - 2)

    bin_n = np.zeros(len(centers), dtype=int)
    bin_err = np.zeros(len(centers))
    bin_severe = np.zeros(len(centers))
    for b in range(len(centers)):
        mask = bins == b
        bin_n[b] = mask.sum()
        if mask.sum() > 0:
            bin_err[b] = err[mask].mean()
            bin_severe[b] = severe[mask].mean()

    # --- Plot ---
    fig = plt.figure(figsize=(12.0, 6.6))
    ax = fig.add_axes((0.08, 0.20, 0.55, 0.58))
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="y", length=0)

    # Histogram of patient density across gap bins (background, ax2)
    hist_color = P.baseline
    ax2.bar(centers, bin_n, width=(centers[1] - centers[0]) * 0.95,
            color=hist_color, alpha=0.30, edgecolor="none", zorder=0,
            label="patients per gap bin")
    ax2.set_ylabel("patients per gap bin", color=P.mute, fontsize=10)
    ax2.set_ylim(0, bin_n.max() * 1.15)

    # LLM-routing band (gap < 0.20)
    ymax = max(bin_err.max() * 1.25, 0.10)
    ax.add_patch(FancyBboxPatch(
        (0, 0), LLM_GAP, ymax,
        boxstyle="round,pad=0,rounding_size=0.005",
        linewidth=0, facecolor=P.judgment, alpha=0.10, zorder=1,
    ))
    ax.text(LLM_GAP / 2, ymax * 0.94,
            "LLM-routed band\n(top-2 gap < 0.20)",
            color=P.judgment, fontsize=9.5, fontweight="bold",
            ha="center", va="top", linespacing=1.4)

    # Error rate curve (ax)
    ax.plot(centers, bin_err, color=P.harm, linewidth=2.2,
            marker="o", markersize=4.0, zorder=3, label="any disagreement")
    ax.plot(centers, bin_severe, color=P.judgment, linewidth=2.2,
            marker="s", markersize=4.0, zorder=3, label="≥2-level deviation")

    ax.set_xlim(0, gap.max() * 1.02)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Top-2 ensemble probability gap (OOF)",
                  color=P.mute, fontsize=10.5)
    ax.set_ylabel("OOF error rate within bin", color=P.mute, fontsize=10.5)
    ax.spines["bottom"].set_color(P.rule)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, alpha=0.4)

    # Legend, manual
    leg_y = ax.get_ylim()[1] * 0.85
    ax.plot([0.32, 0.36], [leg_y, leg_y], color=P.harm, linewidth=2.2)
    ax.text(0.37, leg_y, "any disagreement", color=P.ink, fontsize=10,
            ha="left", va="center")
    ax.plot([0.32, 0.36], [leg_y * 0.90, leg_y * 0.90],
            color=P.judgment, linewidth=2.2)
    ax.text(0.37, leg_y * 0.90, "≥2-level deviation",
            color=P.ink, fontsize=10, ha="left", va="center")
    ax.text(0.37, leg_y * 0.78,
            "(bar histogram: patient density per bin)",
            color=P.mute, fontsize=8.5, ha="left", va="center")

    # Right narrative
    nx0 = 0.74
    # Make room for the right-axis tick labels
    ax2.tick_params(axis="y", labelsize=8.5, pad=0)
    n_routed = int((gap < LLM_GAP).sum())
    pct_routed = 100.0 * n_routed / len(gap)
    err_in = err[gap < LLM_GAP].mean() * 100 if n_routed else 0.0
    err_out = err[gap >= LLM_GAP].mean() * 100
    ratio = err_in / max(err_out, 1e-9)

    fig.text(nx0, 0.66, "ROUTING DIAGNOSIS",
             color=P.mute, fontsize=9.0, fontweight="bold",
             ha="left", va="top", family="monospace")
    fig.text(nx0, 0.625, f"{pct_routed:.2f}%",
             color=P.judgment, fontsize=22, fontweight="bold",
             ha="left", va="top")
    fig.text(nx0, 0.575,
             "of OOF patients fall in the\nLLM-routing gap band.",
             color=P.ink, fontsize=10.5, ha="left", va="top",
             linespacing=1.45)

    fig.text(nx0, 0.475,
             f"Error rate inside the band:\n  {err_in:.1f}%",
             color=P.harm, fontsize=10.5, ha="left", va="top",
             linespacing=1.5, fontweight="bold")
    fig.text(nx0, 0.415,
             f"Error rate outside:\n  {err_out:.2f}%",
             color=P.ink, fontsize=10.5, ha="left", va="top",
             linespacing=1.5)
    fig.text(nx0, 0.343,
             f"The band is {ratio:.0f}× more error-prone\n"
             "than the rest of the OOF distribution —\n"
             "the architecture diagnoses its own\n"
             "uncertainty, then narrowly invokes the\n"
             "LLM only where uncertainty is real.",
             color=P.mute, fontsize=9.5, ha="left", va="top",
             linespacing=1.5)

    style.editorial_header(
        fig,
        title="The architecture diagnoses its own uncertainty",
        subtitle="OOF error rate vs ensemble confidence. The LLM is invoked only "
                 "where the ensemble itself reports it cannot decide.",
        title_y=0.955, subtitle_y=0.910, rule_y=0.875,
    )
    style.editorial_footer(
        fig,
        source="SOURCE  triagegeist analysis/oof_*.npy  ·  5-fold stratified CV, n=80,000",
    )

    out = Path(__file__).parent / "fig3_routing_calibration.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  (n_routed={n_routed}, pct={pct_routed:.2f}, "
          f"err_in={err_in:.2f}%, err_out={err_out:.2f}%)")


if __name__ == "__main__":
    draw()
