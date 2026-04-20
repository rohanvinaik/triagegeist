#!/usr/bin/env python3
"""
TriageGeist submission pipeline CLI.

Usage:
    # Benchmark on training data (with CV evaluation)
    python submission/pipeline.py --benchmark

    # Generate test submission
    python submission/pipeline.py --submit

    # Benchmark with custom confidence threshold
    python submission/pipeline.py --benchmark --threshold 0.65

    # Geometry-only mode (no model, just rules + coherence)
    python submission/pipeline.py --benchmark --geometry-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (
    COHERENCE_CONFIDENCE_THRESHOLD,
    PipelineResult,
    evaluate_pipeline,
    run_full_pipeline,
    run_geometric_scales,
)


DATA_DIR = PROJECT_ROOT / "data" / "extracted"
MODEL_DIR = PROJECT_ROOT / "models"
SUBMISSION_DIR = Path(__file__).resolve().parent


def cmd_benchmark(args):
    """Run benchmark on training data."""
    import pandas as pd

    print("=" * 60)
    print("TriageGeist Benchmark")
    print("=" * 60)

    if args.geometry_only:
        # Geometry-only mode: just rules + coherence, no model
        train = pd.read_csv(DATA_DIR / "train.csv")
        complaints = pd.read_csv(DATA_DIR / "chief_complaints.csv")
        history = pd.read_csv(DATA_DIR / "patient_history.csv")

        decisions, decomps = run_geometric_scales(train, complaints, history)

        # Evaluate geometry-only predictions
        y_true = train["triage_acuity"].values
        y_pred = [d.esi_prediction for d in decisions]
        confidences = [d.confidence for d in decisions]

        from sklearn.metrics import accuracy_score, classification_report, f1_score

        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        accuracy = accuracy_score(y_true, y_pred)

        print(f"\nGeometry-Only Results:")
        print(f"  Macro F1:    {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")
        print(f"  Accuracy:    {accuracy:.4f}")

        # Confidence distribution
        import numpy as np
        confs = np.array(confidences)
        above = (confs >= args.threshold).sum()
        print(f"\n  Confidence >= {args.threshold}: "
              f"{above}/{len(confs)} ({100*above/len(confs):.1f}%)")
        print(f"  Mean confidence: {confs.mean():.3f}")

        # Method breakdown
        methods = {}
        for d in decisions:
            methods[d.method] = methods.get(d.method, 0) + 1
        for method, count in sorted(methods.items()):
            method_mask = [i for i, d in enumerate(decisions) if d.method == method]
            method_f1 = f1_score(
                [y_true[i] for i in method_mask],
                [y_pred[i] for i in method_mask],
                average="macro",
            )
            print(f"  {method}: n={count}, macro F1={method_f1:.4f}")

        print(f"\n{classification_report(y_true, y_pred, target_names=['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5'])}")
        return

    # Full pipeline benchmark
    result = run_full_pipeline(
        DATA_DIR,
        mode="benchmark",
        confidence_threshold=args.threshold,
    )

    train = pd.read_csv(DATA_DIR / "train.csv")
    metrics = evaluate_pipeline(result, train.set_index("patient_id")["triage_acuity"])

    print(f"\nPipeline stats: {json.dumps(result.stats, indent=2, default=str)}")

    # Save benchmark report
    report_path = SUBMISSION_DIR / "benchmark_report.json"
    report = {
        "metrics": metrics,
        "stats": result.stats,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nBenchmark report saved to {report_path}")


def cmd_submit(args):
    """Generate test submission CSV."""
    import pandas as pd

    print("=" * 60)
    print("TriageGeist Submission")
    print("=" * 60)

    if args.geometry_only:
        # Geometry-only submission
        test = pd.read_csv(DATA_DIR / "test.csv")
        complaints = pd.read_csv(DATA_DIR / "chief_complaints.csv")
        history = pd.read_csv(DATA_DIR / "patient_history.csv")

        decisions, _ = run_geometric_scales(test, complaints, history)

        submission = pd.DataFrame({
            "patient_id": [d.patient_id for d in decisions],
            "triage_acuity": [d.esi_prediction for d in decisions],
        })
    else:
        result = run_full_pipeline(
            DATA_DIR,
            mode="submit",
            model_dir=MODEL_DIR,
            confidence_threshold=args.threshold,
        )

        submission = pd.DataFrame({
            "patient_id": result.predictions.index,
            "triage_acuity": result.predictions.values,
        })

    output_path = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Distribution:\n{submission['triage_acuity'].value_counts().sort_index()}")


def main():
    parser = argparse.ArgumentParser(description="TriageGeist Pipeline")
    sub = parser.add_subparsers(dest="command")

    # Benchmark
    bench = sub.add_parser("benchmark", help="Benchmark on training data")
    bench.add_argument("--threshold", type=float,
                       default=COHERENCE_CONFIDENCE_THRESHOLD,
                       help="Confidence threshold for geometry vs model")
    bench.add_argument("--geometry-only", action="store_true",
                       help="Run geometry-only (no model)")

    # Submit
    submit = sub.add_parser("submit", help="Generate submission CSV")
    submit.add_argument("--threshold", type=float,
                        default=COHERENCE_CONFIDENCE_THRESHOLD,
                        help="Confidence threshold for geometry vs model")
    submit.add_argument("--geometry-only", action="store_true",
                        help="Geometry-only submission (no model)")

    # Legacy flags for convenience
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark mode")
    parser.add_argument("--submit", action="store_true",
                        help="Run submit mode")
    parser.add_argument("--threshold", type=float,
                        default=COHERENCE_CONFIDENCE_THRESHOLD)
    parser.add_argument("--geometry-only", action="store_true")

    args = parser.parse_args()

    if args.command == "benchmark" or getattr(args, "benchmark", False):
        cmd_benchmark(args)
    elif args.command == "submit" or getattr(args, "submit", False):
        cmd_submit(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
