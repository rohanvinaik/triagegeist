"""
Cover image — 560 × 280 px exactly, as required by the Kaggle Writeup spec.

Stripped-back version of Fig 1's structural geometry. The hero figure with
text removed and proportions preserved. The point of the cover is to be
the invitation: the reader's eye registers the asymmetry (one large bar,
one nearly-invisible bar) and is drawn into the writeup to find out what
it means.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import style
from style import P

style.apply()

# Same data as Fig 1
SEGMENTS = [
    ("PROTOCOL",  2.88,  P.protocol),
    ("ENSEMBLE",  96.98, P.accent),
    ("LLM",       0.14,  P.judgment),
]


def draw():
    # Kaggle cover spec: 560×280 px
    fig = plt.figure(figsize=(5.60, 2.80), dpi=100)

    ax = fig.add_axes((0.05, 0.20, 0.78, 0.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    bar_h = 0.55
    ys = [2.45, 1.40, 0.35]

    for (name, pct, color), y in zip(SEGMENTS, ys):
        # Track
        track = FancyBboxPatch(
            (0, y), 100, bar_h,
            boxstyle="round,pad=0,rounding_size=0.10",
            linewidth=0.5, edgecolor=P.rule,
            facecolor="none", alpha=0.6,
        )
        ax.add_patch(track)
        # Filled
        fill = FancyBboxPatch(
            (0, y), pct, bar_h,
            boxstyle="round,pad=0,rounding_size=0.10",
            linewidth=0, facecolor=color, alpha=0.92,
        )
        ax.add_patch(fill)

        # Label inside the row, above the bar
        ax.text(0, y + bar_h + 0.10, name,
                color=P.ink, fontsize=8.0, fontweight="bold",
                ha="left", va="bottom")
        # Percentage at right edge, in figure coords for safe margin
        ax.text(102, y + bar_h / 2,
                f"{pct:.2f}%",
                color=color, fontsize=11, fontweight="bold",
                ha="left", va="center", clip_on=False)

    fig.text(0.05, 0.90, "TriageGeist",
             color=P.ink, fontsize=15, fontweight="bold",
             ha="left", va="top")
    fig.text(0.05, 0.81,
             "auditable triage at the cost of compute it deserves",
             color=P.mute, fontsize=8.5, fontstyle="italic",
             ha="left", va="top")

    fig.text(0.05, 0.07,
             "99.86%  zero-LLM  ·  0.14%  typed-contract LLM  ·  100%  certified",
             color=P.mute, fontsize=7.5, family="monospace",
             ha="left", va="bottom")

    out = Path(__file__).parent / "cover.png"
    # Force canvas size — bbox=None preserves figsize × dpi exactly
    import matplotlib as mpl
    with mpl.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0}):
        fig.savefig(out, dpi=100, facecolor="none")
    plt.close(fig)
    from PIL import Image
    sz = Image.open(out).size
    print(f"wrote {out}  size={sz}")


if __name__ == "__main__":
    draw()
