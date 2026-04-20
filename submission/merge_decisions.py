#!/usr/bin/env python3
"""
Merge Claude sub-agent decisions into final submission.

Takes baseline model predictions + LLM decisions for uncertain cases
and produces the final submission CSV.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SUBMISSION_DIR = Path(__file__).resolve().parent
DATA_DIR = SUBMISSION_DIR.parent / "data" / "extracted"


def main():
    # Load baseline predictions (model + rules)
    baseline = json.load(open(SUBMISSION_DIR / "baseline_preds.json"))
    print(f"Baseline predictions: {len(baseline)}")

    # Load all decision batches
    llm_decisions = {}
    for batch_file in sorted(SUBMISSION_DIR.glob("decisions_batch_*.json")):
        decisions = json.load(open(batch_file))
        for d in decisions:
            llm_decisions[d["patient_id"]] = d["esi"]
        print(f"  {batch_file.name}: {len(decisions)} decisions")

    print(f"Total LLM decisions: {len(llm_decisions)}")

    # Merge: LLM overrides model for uncertain cases
    n_changed = 0
    n_same = 0
    final_preds = {}

    for pid, info in baseline.items():
        if pid in llm_decisions:
            llm_esi = llm_decisions[pid]
            model_esi = info["pred"]
            if llm_esi != model_esi:
                n_changed += 1
            else:
                n_same += 1
            final_preds[pid] = llm_esi
        else:
            final_preds[pid] = info["pred"]

    print(f"\nLLM changed {n_changed} predictions, agreed with {n_same}")
    print(f"Unchanged (confident model + rules): "
          f"{len(baseline) - len(llm_decisions)}")

    # Build submission
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission = pd.DataFrame({
        "patient_id": sample["patient_id"],
        "triage_acuity": [final_preds[pid] for pid in sample["patient_id"]],
    })

    output_path = SUBMISSION_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)

    print(f"\nSubmission saved to {output_path}")
    print(f"Shape: {submission.shape}")
    print(f"Distribution:\n{submission['triage_acuity'].value_counts().sort_index()}")

    # Sanity checks
    assert len(submission) == len(sample)
    assert list(submission.columns) == list(sample.columns)
    assert submission["triage_acuity"].between(1, 5).all()
    print("\nSanity check passed!")


if __name__ == "__main__":
    main()
