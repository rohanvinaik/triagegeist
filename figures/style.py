"""
Editorial chassis for the TriageGeist figure suite.

Palette, rcParams, editorial header/footer, name normalization. Every
figure imports from here; cross-figure coherence is enforced at the
style-module level.

Transparent PNG output, three spines off, mid-luminance text that
survives both Kaggle light (#F7F7F8) and Kaggle dark (#20232A) pages.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
from matplotlib.lines import Line2D


@dataclass(frozen=True)
class Palette:
    # text / structure
    ink: str = "#7A8290"       # primary text   — L≈0.20
    mute: str = "#8F97A4"      # axis labels    — L≈0.27
    rule: str = "#B0B6C0"      # divider rules
    grid: str = "#A4AAB4"      # gridlines      — alpha applied at rcParams

    # structural neutrals
    neutral: str = "#8896A8"
    baseline: str = "#B2B8C0"

    # regime colors — carry the three-regime argument across every figure
    # protocol = high-confidence symbolic, escalation = conservative clamp,
    # judgment = LLM-routed uncertain band.
    protocol: str = "#1F4D9B"      # deep blue  — clean, protocol-grade
    escalation: str = "#C5882D"    # amber      — cautionary escalation
    judgment: str = "#8A4B9B"      # purple     — genuine clinical ambiguity

    # clinical-safety encoding
    safe_exact: str = "#3B6F4A"    # green       — exact agreement
    safe_adj: str = "#8FA4A8"      # gray-teal   — off-by-one
    harm: str = "#B73A3F"          # red         — ≥2 level deviation

    # accent (sparingly used for specific annotations)
    accent: str = "#3367D6"
    accent_soft: str = "#7BA5D9"

    # diverging scale (heatmaps)
    heatmap_low: str = "#1F4D9B"
    heatmap_mid: str = "#BEC2C8"
    heatmap_high: str = "#B73A3F"


P = Palette()


# Scope-matching callout colors keyed by finding, not category
CALLOUT_COLORS = {
    "zero_catastrophic": P.safe_exact,
    "protocol_layer": P.protocol,
    "escalation_layer": P.escalation,
    "judgment_layer": P.judgment,
    "disclosed_artifact": P.escalation,
    "validated_hard_split": P.judgment,
}


# Proper-noun normalization — never let a raw identifier reach a label
_PRETTY_MAP = [
    ("temporal_news2_deviation", "NEWS2 cohort-deviation"),
    ("news2", "NEWS2"),
    ("gcs", "GCS"),
    ("esi_1", "ESI 1"),
    ("esi_2", "ESI 2"),
    ("esi_3", "ESI 3"),
    ("esi_4", "ESI 4"),
    ("esi_5", "ESI 5"),
    ("catboost", "CatBoost"),
    ("lightgbm", "LightGBM"),
    ("kuramoto", "Kuramoto"),
    ("qwk", "QWK"),
    ("llm", "LLM"),
]


def prettify(s: str) -> str:
    lowered = s.lower()
    for raw, nice in _PRETTY_MAP:
        if raw in lowered:
            lowered = lowered.replace(raw, nice)
    if lowered == s.lower():
        return s  # no match — leave user casing
    return lowered


def apply() -> None:
    """Install editorial rcParams. Call once at top of every figure script."""
    mpl.rcParams.update({
        # transparency — dual-mode survival
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,

        # typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 10,

        # editorial frame
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,

        "axes.labelcolor": P.mute,
        "axes.titlecolor": P.ink,
        "xtick.color": P.mute,
        "ytick.color": P.ink,
        "xtick.major.size": 0,
        "ytick.major.size": 0,

        "grid.color": P.grid,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.55,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "axes.axisbelow": True,

        "legend.frameon": False,

        # save defaults
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def editorial_header(fig, title: str, subtitle: str = "",
                     title_x: float = 0.02, title_y: float = 0.965,
                     subtitle_y: float = 0.930, rule_y: float = 0.910) -> None:
    """Bold title, muted subtitle, thin horizontal rule."""
    fig.text(title_x, title_y, title, color=P.ink,
             fontsize=15, fontweight="bold", ha="left", va="top")
    if subtitle:
        fig.text(title_x, subtitle_y, subtitle, color=P.mute,
                 fontsize=10.5, ha="left", va="top")
    fig.add_artist(Line2D(
        [title_x, 1 - title_x], [rule_y, rule_y],
        color=P.rule, linewidth=0.75, transform=fig.transFigure,
    ))


def editorial_footer(fig, source: str,
                     x: float = 0.02, y: float = 0.02) -> None:
    """Monospace metadata strip at figure bottom."""
    fig.text(x, y, source, color=P.mute, fontsize=8.0,
             family="monospace", ha="left", va="bottom")


# Target page backgrounds — used by preview_dualbg
LIGHT_BG = (247, 247, 248)
DARK_BG = (32, 35, 42)
