"""
Fig 1 (hero): the auditability triangle.

Three regimes of triage decision-making with their audit properties and
the fraction of 20,000 test patients each resolves. The number — 0.14%
LLM-routed — is the argument. The figure's job is to make that number
visually unavoidable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import style
from style import P

style.apply()

# Counts from submission_audit.json — the formal ones used in submission.csv
REGIMES = [
    {
        "name": "PROTOCOL",
        "subtitle": "hard rules",
        "count": 575,
        "pct": 2.88,
        "color": P.protocol,
        "audit": "Fully symbolic. Auditable as a decision tree —\n"
                 "every prediction traces to a single rule firing.",
        "example": "GCS ≤ 8 → ESI 1",
    },
    {
        "name": "COMMON PATTERN",
        "subtitle": "ensemble + QWK thresholds",
        "count": 19_396,
        "pct": 96.98,
        "color": P.accent,
        "audit": "Deterministic statistical inference. Per-prediction\n"
                 "feature attributions inspectable; model is reproducible.",
        "example": "124 features, CatBoost + LightGBM, QWK thresholds",
    },
    {
        "name": "JUDGMENT",
        "subtitle": "LLM under typed contract",
        "count": 29,
        "pct": 0.14,
        "color": P.judgment,
        "audit": "Stochastic reasoning, but structurally bounded.\n"
                 "Output restricted to closed-vocabulary typed schema;\n"
                 "independent certifier checks evidence consistency.",
        "example": "Top-2 probability gap < 0.20",
    },
]


def draw():
    fig = plt.figure(figsize=(12, 8.0))

    # content axes
    ax = fig.add_axes((0.04, 0.06, 0.88, 0.74))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10.8)
    ax.axis("off")

    # Vertical stacking: top = protocol, middle = common, bottom = judgment
    bar_height = 0.80
    y_positions = [8.2, 4.6, 1.0]

    for regime, y in zip(REGIMES, y_positions):
        # rounded filled segment — length proportional to share (linear scale)
        w = regime["pct"]
        bar = FancyBboxPatch(
            (0, y), w, bar_height,
            boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=0, facecolor=regime["color"], alpha=0.88,
        )
        ax.add_patch(bar)

        # background track showing the full 100%
        track = FancyBboxPatch(
            (0, y), 100, bar_height,
            boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=0.6, edgecolor=P.rule,
            facecolor="none", alpha=0.7,
        )
        ax.add_patch(track)

        # Regime name — above the bar
        ax.text(0, y + bar_height + 0.35,
                regime["name"],
                color=P.ink, fontsize=12.5, fontweight="bold",
                ha="left", va="bottom")
        ax.text(16.5, y + bar_height + 0.35,
                "·  " + regime["subtitle"],
                color=P.mute, fontsize=10.5,
                ha="left", va="bottom", fontstyle="italic")

        # Count + percentage — right side of the row
        pct_text = f"{regime['pct']:.2f}%"
        ax.text(101, y + bar_height / 2 + 0.12,
                pct_text,
                color=regime["color"], fontsize=18, fontweight="bold",
                ha="left", va="center")
        ax.text(101, y + bar_height / 2 - 0.35,
                f"{regime['count']:,} of 20,000",
                color=P.mute, fontsize=8.5,
                ha="left", va="center", family="monospace")

        # Audit descriptor — below the bar
        ax.text(0, y - 0.25, regime["audit"],
                color=P.ink, fontsize=9.5,
                ha="left", va="top", linespacing=1.4)

    # Closing frame: the argument, collapsed to one line
    fig.text(0.04, 0.84,
             "99.86% of triage decisions involve zero LLM reasoning.",
             color=P.ink, fontsize=11.5, fontweight="bold",
             ha="left", va="top")

    style.editorial_header(
        fig,
        title="Three regimes of triage decision-making",
        subtitle="Each decision in the test set routes to the scale of audit its "
                 "epistemic character deserves.",
        title_y=0.965, subtitle_y=0.930, rule_y=0.895,
    )
    style.editorial_footer(
        fig,
        source="SOURCE  triagegeist submission_audit.json  ·  n=20,000 test patients",
    )

    out = Path(__file__).parent / "fig1_auditability_triangle.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    draw()
