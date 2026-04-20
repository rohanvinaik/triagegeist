#!/usr/bin/env python3
"""Statistical analysis of the easy-vs-hard regime split.

Parses per-fold (easy_qwk, hard_qwk) pairs from the regime-split benchmark
log and runs:
  - Paired t-test on (easy_qwk_f - hard_qwk_f) across folds (H0: mean diff = 0)
  - Wilcoxon signed-rank test (nonparametric paired equivalent)
  - Cohen's d effect size on the paired differences
  - Levene's test on variances (H0: Var(easy) = Var(hard))
  - 95% bootstrap CI on the mean paired difference

With only n=5 folds the parametric tests are low-power and should be
interpreted in conjunction with the effect size + the cross-variant
consistency. A significant result on n=5 is a strong signal; a
non-significant one is not evidence of absence.

Usage:
  python analysis/regime_stats.py [--log /path/to/bench.log]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats


FOLD_LINE_RE = re.compile(
    r"\[(?P<variant>\w+)\] fold (?P<fold>\d+)\s+"
    r"all: mF1=(?P<all_mf1>[\d.]+) qwk=(?P<all_qwk>[\d.]+)\s+\|\s+"
    r"easy \(n=(?P<easy_n>\d+)\) qwk=(?P<easy_qwk>[\d.]+)\s+\|\s+"
    r"hard \(n=(?P<hard_n>\d+)\) qwk=(?P<hard_qwk>[\d.]+)"
)


def parse_log(log_path: Path) -> dict[str, list[dict]]:
    """Parse per-fold (easy, hard) QWK pairs from the benchmark log."""
    text = log_path.read_text()
    out: dict[str, list[dict]] = {}
    for m in FOLD_LINE_RE.finditer(text):
        variant = m.group("variant")
        out.setdefault(variant, []).append({
            "fold": int(m.group("fold")),
            "all_mf1": float(m.group("all_mf1")),
            "all_qwk": float(m.group("all_qwk")),
            "easy_n": int(m.group("easy_n")),
            "easy_qwk": float(m.group("easy_qwk")),
            "hard_n": int(m.group("hard_n")),
            "hard_qwk": float(m.group("hard_qwk")),
        })
    return out


def cohens_d_paired(diffs: np.ndarray) -> float:
    """Cohen's d for paired differences = mean(d) / std(d)."""
    if len(diffs) < 2 or diffs.std(ddof=1) == 0:
        return float("nan")
    return float(diffs.mean() / diffs.std(ddof=1))


def bootstrap_ci(arr: np.ndarray, n_boot: int = 10_000,
                 ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Bootstrap percentile CI for mean of arr."""
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def analyse_variant(folds: list[dict], variant: str) -> dict:
    """Run stats on per-fold easy vs hard QWK for one variant."""
    easy = np.array([f["easy_qwk"] for f in folds])
    hard = np.array([f["hard_qwk"] for f in folds])
    diffs = easy - hard

    # Paired t-test
    t_stat, t_p = stats.ttest_rel(easy, hard)
    # Wilcoxon signed-rank (nonparametric)
    if np.all(diffs == 0):
        w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = stats.wilcoxon(easy, hard, zero_method="zsplit")
    # Variance equality (Levene on 5 points is weak but report anyway)
    lev_stat, lev_p = stats.levene(easy, hard, center="median")
    # Bootstrap CI on mean diff
    ci_lo, ci_hi = bootstrap_ci(diffs)

    return {
        "variant": variant,
        "n_folds": len(folds),
        "easy_mean": float(easy.mean()),
        "easy_std": float(easy.std(ddof=1)) if len(easy) > 1 else float("nan"),
        "hard_mean": float(hard.mean()),
        "hard_std": float(hard.std(ddof=1)) if len(hard) > 1 else float("nan"),
        "var_ratio_easy_over_hard": float(
            (easy.var(ddof=1) / hard.var(ddof=1))
            if hard.var(ddof=1) > 0 else float("nan")
        ),
        "mean_diff": float(diffs.mean()),
        "mean_diff_ci95": [ci_lo, ci_hi],
        "cohens_d": cohens_d_paired(diffs),
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
        "levene_stat": float(lev_stat),
        "levene_p": float(lev_p),
        "per_fold_diffs": diffs.tolist(),
    }


def compare_variants(folds_a: list[dict], folds_b: list[dict],
                      name_a: str, name_b: str, subset: str) -> dict:
    """Compare a metric across two variants, paired by fold."""
    key = f"{subset}_qwk"
    a = np.array([f[key] for f in folds_a])
    b = np.array([f[key] for f in folds_b])
    if len(a) != len(b) or len(a) < 2:
        return {"note": "insufficient paired folds"}
    diffs = b - a  # positive → b beats a
    t_stat, t_p = stats.ttest_rel(b, a)
    try:
        w_stat, w_p = stats.wilcoxon(b, a, zero_method="zsplit")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    ci_lo, ci_hi = bootstrap_ci(diffs)
    return {
        "pair": f"{name_b} - {name_a}",
        "subset": subset,
        "mean_diff": float(diffs.mean()),
        "mean_diff_ci95": [ci_lo, ci_hi],
        "cohens_d": cohens_d_paired(diffs),
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(t_p),
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "wilcoxon_p": float(w_p) if not np.isnan(w_p) else None,
        "per_fold_diffs": diffs.tolist(),
    }


def pretty_print(result: dict) -> None:
    """Human-readable report."""
    print("=" * 70)
    print("Regime-split statistical analysis")
    print("=" * 70)

    for variant, summary in result["per_variant"].items():
        print(f"\n[{variant}] — n_folds={summary['n_folds']}")
        print(f"  easy qwk: mean={summary['easy_mean']:.4f}  "
              f"std={summary['easy_std']:.4f}")
        print(f"  hard qwk: mean={summary['hard_mean']:.4f}  "
              f"std={summary['hard_std']:.4f}")
        print(f"  variance ratio (easy/hard): "
              f"{summary['var_ratio_easy_over_hard']:.3f}")
        print(f"  paired diff (easy - hard): "
              f"mean={summary['mean_diff']:+.4f}  "
              f"95% CI={summary['mean_diff_ci95']}")
        print(f"  Cohen's d: {summary['cohens_d']:+.3f}")
        print(f"  paired t-test:  t={summary['paired_t_stat']:+.3f}  "
              f"p={summary['paired_t_p']:.4f}")
        if summary.get("wilcoxon_stat") is not None:
            print(f"  Wilcoxon:       W={summary['wilcoxon_stat']:.1f}  "
                  f"p={summary['wilcoxon_p']:.4f}")
        print(f"  Levene (var):   F={summary['levene_stat']:.3f}  "
              f"p={summary['levene_p']:.4f}")
        print(f"  per-fold diffs: "
              f"{[f'{d:+.4f}' for d in summary['per_fold_diffs']]}")

    if "cross_variant" in result:
        print("\n" + "=" * 70)
        print("Cross-variant comparisons")
        print("=" * 70)
        for cmp in result["cross_variant"]:
            print(f"\n{cmp['pair']} on {cmp['subset']} subset:")
            print(f"  mean diff: {cmp['mean_diff']:+.4f}  "
                  f"95% CI={cmp['mean_diff_ci95']}")
            print(f"  Cohen's d: {cmp['cohens_d']:+.3f}")
            print(f"  paired t-test: t={cmp['paired_t_stat']:+.3f}  "
                  f"p={cmp['paired_t_p']:.4f}")
            if cmp.get("wilcoxon_stat") is not None:
                print(f"  Wilcoxon:     W={cmp['wilcoxon_stat']:.1f}  "
                      f"p={cmp['wilcoxon_p']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log",
                    default="/tmp/triagegeist_regime_bench.log",
                    help="Benchmark log to parse")
    ap.add_argument("--out",
                    default=str(Path(__file__).parent / "regime_stats.json"),
                    help="Output JSON")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    parsed = parse_log(log_path)
    if not parsed:
        print("No fold lines found in log.", file=sys.stderr)
        sys.exit(1)

    # Per-variant stats
    per_variant = {v: analyse_variant(folds, v)
                   for v, folds in parsed.items()}

    # Cross-variant comparisons (phase vs baseline, full vs baseline, full vs phase)
    cross = []
    variants_present = set(parsed.keys())
    for a, b in [("baseline", "phase"), ("baseline", "full_stack"),
                 ("phase", "full_stack"),
                 ("baseline", "hard_bucket_v2"),
                 ("phase", "hard_bucket_v2"),
                 ("full_stack", "hard_bucket_v2")]:
        if a in variants_present and b in variants_present:
            for subset in ("easy", "hard", "all"):
                cmp = compare_variants(parsed[a], parsed[b], a, b, subset)
                cross.append(cmp)

    result = {
        "source_log": str(log_path),
        "variants_parsed": sorted(parsed.keys()),
        "per_variant": per_variant,
        "cross_variant": cross,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    pretty_print(result)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
