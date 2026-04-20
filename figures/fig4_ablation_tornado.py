"""
Fig 4: where the +0.0168 QWK Tier-B lift actually lives.

Six candidate feature families were added to the baseline ensemble. Only
one delivered lift. The figure makes the asymmetry unavoidable: one bar
towers, the rest sit on or below the zero line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import style
from style import P

style.apply()


# From analysis/*.json + project_tier_b_forensic_verdict.md (2026-04-18)
# Units: ΔQWK vs pure-baseline ensemble, fold-safe 5-fold CV.
FAMILIES = [
    # (label, n_features, delta_qwk, is_dominant, note)
    ("NEWS2 cohort-deviation",       1,  +0.0154, True,  "kept (disclosed)"),
    ("Other 4 temporal features",    4,  +0.0014, False, "marginal; kept in audit context only"),
    ("Confidence calibrator",        1,  +0.0002, False, "inert; retained for narrative"),
    ("Clinician style features",     9,  +0.0001, False, "inert"),
    ("Kuramoto phase deviations",   11,  -0.0002, False, "net-negative; rendered into LLM context"),
    ("Surprisal-basis features",    87,  -0.0013, False, "net-negative; colinear with raw bank signals"),
]

TOTAL_TIERB_LIFT = 0.0168


def draw():
    fig = plt.figure(figsize=(11.5, 6.6))
    ax = fig.add_axes((0.28, 0.18, 0.60, 0.62))

    deltas = np.array([f[2] for f in FAMILIES])
    dominant = [f[3] for f in FAMILIES]

    y = np.arange(len(FAMILIES))[::-1]  # top = first entry

    # Color logic: dominant = strong protocol blue, positive-marginal = muted blue,
    # zero-ish = baseline gray, negative = harm red
    colors = []
    for d, dom in zip(deltas, dominant):
        if dom:
            colors.append(P.accent)
        elif d > 0.0005:
            colors.append(P.accent_soft)
        elif d < -0.0005:
            colors.append(P.harm)
        else:
            colors.append(P.baseline)

    ax.barh(y, deltas, height=0.62, color=colors, edgecolor="none", alpha=0.92)

    # Zero rule
    ax.axvline(0, color=P.rule, linewidth=0.9, zorder=0)

    # X axis formatting
    ax.set_xlim(-0.004, 0.020)
    ax.set_xticks([-0.002, 0.000, 0.005, 0.010, 0.015])
    ax.set_xticklabels(["−0.002", "0", "+0.005", "+0.010", "+0.015"])
    ax.set_xlabel("ΔQWK vs baseline ensemble (fold-safe, 5-fold CV)",
                  color=P.mute, fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([])  # we draw labels manually for precision

    # Bar end annotations: delta value
    for yi, d, color in zip(y, deltas, colors):
        sign = "+" if d >= 0 else "−"
        txt = f"{sign}{abs(d):.4f}"
        xpos = d + (0.0005 if d >= 0 else -0.0005)
        ha = "left" if d >= 0 else "right"
        ax.text(xpos, yi, txt, color=color, fontsize=10.5,
                fontweight="bold", ha=ha, va="center")

    # Left-side labels: family name (bold) + (n features) in monospace
    for yi, (lab, nf, _, dom, note) in zip(y, FAMILIES):
        weight = "bold" if dom else "regular"
        ax.text(-0.0045, yi + 0.08, lab,
                color=P.ink, fontsize=10.5, fontweight=weight,
                ha="right", va="center", transform=ax.transData)
        ax.text(-0.0045, yi - 0.22, f"[{nf} feat]  {note}",
                color=P.mute, fontsize=8.5, family="monospace",
                ha="right", va="center", transform=ax.transData)

    # Callout annotation — the argument in one line
    ax.annotate(
        "",
        xy=(0.0154, y[0]), xytext=(0.0165, y[0] + 1.0),
        arrowprops=dict(arrowstyle="-", color=P.ink, lw=0.7,
                        connectionstyle="arc3,rad=0.2"),
    )
    ax.text(0.0172, y[0] + 0.95,
            "91% of the +0.0168 Tier-B lift",
            color=P.ink, fontsize=10, fontweight="bold",
            ha="left", va="bottom")
    ax.text(0.0172, y[0] + 0.55,
            "launders the declined complaint-to-ESI\n"
            "shortcut through cohort-conditional NEWS2.\n"
            "Kept, with the laundering disclosed.",
            color=P.mute, fontsize=8.8,
            ha="left", va="top", linespacing=1.4)

    # Cosmetic
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(P.rule)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)

    style.editorial_header(
        fig,
        title="Where the Tier-B lift actually lives",
        subtitle="Six candidate feature families were added to the baseline "
                 "ensemble. One delivered 91% of the lift. Disclosure makes "
                 "the finding into evidence, not a liability.",
        title_y=0.955, subtitle_y=0.905, rule_y=0.865,
    )
    style.editorial_footer(
        fig,
        source="SOURCE  triagegeist analysis/*.json  ·  regime-split 5-fold CV, fold-safe",
    )

    out = Path(__file__).parent / "fig4_ablation_tornado.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    draw()
