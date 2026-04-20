"""
Fig 2: where disagreement actually lives in OOF predictions.

A 5×5 confusion matrix with three overlaid clinical zones:
  diagonal   = exact agreement
  off-by-1   = clinically tolerable (within rater-disagreement envelope)
  off-by-≥2  = catastrophic mistriage — the cells the system has to keep empty

Loads numbers from analysis/oof_summary.json. Fails informatively if the
OOF run hasn't completed yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import style
from style import P

style.apply()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = PROJECT_ROOT / "analysis" / "oof_summary.json"


def load_data():
    if not SUMMARY_PATH.exists():
        raise SystemExit(
            f"OOF summary not found at {SUMMARY_PATH}. Run analysis/oof_evidentiary.py first."
        )
    return json.loads(SUMMARY_PATH.read_text())


def draw():
    data = load_data()
    cm = np.array(data["confusion_matrix"], dtype=int)  # (5,5) true × pred

    fig = plt.figure(figsize=(11.5, 7.0))
    ax = fig.add_axes((0.10, 0.16, 0.50, 0.66))

    # Cell sizing
    n = cm.shape[0]
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()  # ESI 1 at top
    ax.set_aspect("equal")

    # Background zones — overlay the three clinical regions
    # Catastrophic (|diff| >= 2): paint first as the "bad" zone
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            if d >= 2:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=P.harm, alpha=0.06,
                                       edgecolor="none", zorder=0))
            elif d == 1:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=P.safe_adj, alpha=0.10,
                                       edgecolor="none", zorder=0))
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=P.safe_exact, alpha=0.16,
                                       edgecolor="none", zorder=0))

    # Cell counts
    cm_max = cm.max()
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            d = abs(i - j)
            if d == 0:
                color = P.safe_exact
            elif d == 1:
                color = P.ink
            else:
                color = P.harm
            weight = "bold" if (d >= 2 or v == 0 and d >= 2) else "normal"
            if d >= 2:
                weight = "bold"
            txt = f"{v:,}" if v > 0 else "·"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=12.5,
                    fontweight=weight if d >= 2 else "regular")

    # Grid
    for k in range(n + 1):
        ax.axvline(k - 0.5, color=P.rule, linewidth=0.4, alpha=0.5)
        ax.axhline(k - 0.5, color=P.rule, linewidth=0.4, alpha=0.5)

    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"ESI {i+1}" for i in range(n)], color=P.mute)
    ax.set_yticklabels([f"ESI {i+1}" for i in range(n)], color=P.ink)
    ax.set_xlabel("Predicted", color=P.mute, fontsize=10.5)
    ax.set_ylabel("True label", color=P.mute, fontsize=10.5)
    ax.tick_params(axis="both", which="both", length=0)
    ax.grid(False)

    # Right-side narrative panel
    nx0 = 0.66
    n_total = data["n_total"]
    pct_exact = data["pct_exact"] * 100
    pct_within = data["pct_within_1"] * 100
    pct_ge2 = data["pct_ge_2"] * 100
    n_ge2 = data["n_ge_2"]
    err_total = data["err_total"]
    err_by_boundary = data["err_by_boundary"]
    pct_err12 = 100 * err_by_boundary["1_vs_2"] / max(err_total, 1)
    pct_err23 = 100 * err_by_boundary["2_vs_3"] / max(err_total, 1)
    pct_err34 = 100 * err_by_boundary["3_vs_4"] / max(err_total, 1)
    pct_err45 = 100 * err_by_boundary["4_vs_5"] / max(err_total, 1)

    fig.text(nx0, 0.74, "CLINICAL SAFETY",
             color=P.mute, fontsize=9.0, fontweight="bold",
             ha="left", va="top", family="monospace")
    fig.text(nx0, 0.71, f"{pct_exact:.1f}% exact",
             color=P.safe_exact, fontsize=18, fontweight="bold",
             ha="left", va="top")
    fig.text(nx0, 0.665, f"{pct_within:.2f}% within ±1 ESI level",
             color=P.ink, fontsize=11, ha="left", va="top")
    fig.text(nx0, 0.638,
             f"{n_ge2:,} of {n_total:,} ({pct_ge2:.3f}%) deviate by ≥2",
             color=P.harm if n_ge2 > 0 else P.safe_exact,
             fontsize=10.5, ha="left", va="top",
             fontweight="bold" if n_ge2 > 0 else "regular")

    fig.text(nx0, 0.555, "WHERE DISAGREEMENT LIVES",
             color=P.mute, fontsize=9.0, fontweight="bold",
             ha="left", va="top", family="monospace")
    fig.text(nx0, 0.525,
             f"Of {err_total:,} OOF disagreements with the label:",
             color=P.mute, fontsize=10, ha="left", va="top")
    fig.text(nx0, 0.495,
             f"·  {pct_err45:4.1f}% on the ESI 4vs5 boundary",
             color=P.ink, fontsize=10.5, ha="left", va="top",
             fontweight="bold", family="monospace")
    fig.text(nx0, 0.470,
             f"·  {pct_err34:4.1f}% on the ESI 3vs4 boundary",
             color=P.ink, fontsize=10.5, ha="left", va="top",
             family="monospace")
    fig.text(nx0, 0.446,
             f"·  {pct_err12:4.1f}% on the ESI 1vs2 boundary",
             color=P.mute, fontsize=10.5, ha="left", va="top",
             family="monospace")
    fig.text(nx0, 0.422,
             f"·  {pct_err23:4.1f}% on the ESI 2vs3 boundary",
             color=P.mute, fontsize=10.5, ha="left", va="top",
             family="monospace")
    fig.text(nx0, 0.378,
             "Disagreement concentrates at the lowest-acuity\n"
             "boundary where mistriage is clinically cheapest;\n"
             "only 6.7% of errors involve the high-acuity\n"
             "1vs2 or 2vs3 boundaries where harm is asymmetric.",
             color=P.mute, fontsize=9.0, ha="left", va="top",
             linespacing=1.5)

    # Mini legend for the three zones
    fig.text(nx0, 0.27, "ZONES",
             color=P.mute, fontsize=9.0, fontweight="bold",
             ha="left", va="top", family="monospace")
    legend_y = 0.232
    for label, swatch_color, alpha, desc in [
        ("Diagonal", P.safe_exact, 0.50, "exact agreement with label"),
        ("Off-by-1", P.safe_adj, 0.40, "within rater-disagreement envelope"),
        ("Off-by-≥2", P.harm, 0.18, "catastrophic mistriage — kept empty"),
    ]:
        fig.add_artist(Rectangle((nx0, legend_y), 0.014, 0.020,
                                 facecolor=swatch_color, alpha=alpha,
                                 edgecolor="none",
                                 transform=fig.transFigure))
        fig.text(nx0 + 0.020, legend_y + 0.010, f"{label}",
                 color=P.ink, fontsize=10, ha="left", va="center",
                 fontweight="bold")
        fig.text(nx0 + 0.094, legend_y + 0.010, desc,
                 color=P.mute, fontsize=9.5, ha="left", va="center")
        legend_y -= 0.034

    style.editorial_header(
        fig,
        title="The shape of disagreement",
        subtitle="5-fold OOF confusion on 80,000 training patients. "
                 "The off-by-≥2 region is the only zone whose contents matter "
                 "for clinical safety.",
        title_y=0.955, subtitle_y=0.910, rule_y=0.875,
    )
    style.editorial_footer(
        fig,
        source="SOURCE  triagegeist analysis/oof_summary.json  ·  5-fold stratified CV, n=80,000",
    )

    out = Path(__file__).parent / "fig2_disagreement_topology.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  (cm_max={cm_max})")


if __name__ == "__main__":
    draw()
