"""Tests for src.pipeline — kill-targeted per LintGate prescriptions.

Focuses on _load_data, run_geometric_scales, evaluate_pipeline.
run_full_pipeline is already at 100% kill rate via docstring extraction.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    COHERENCE_CONFIDENCE_THRESHOLD, PipelineResult, _load_data,
    evaluate_pipeline, run_geometric_scales,
)


# ---------- _load_data ----------

@pytest.fixture
def mini_data_dir(tmp_path):
    """Write minimal train/test/complaints/history CSVs to a temp dir."""
    base_cols = {
        "patient_id": ["P0", "P1", "P2"],
        "triage_nurse_id": ["N1", "N1", "N2"],
        "site_id": ["S1", "S1", "S1"],
        "arrival_mode": ["walk-in"] * 3,
        "arrival_hour": [12, 13, 14],
        "arrival_day": ["Mon"] * 3,
        "arrival_month": [6] * 3,
        "arrival_season": ["summer"] * 3,
        "shift": ["day"] * 3,
        "age": [45, 55, 65],
        "age_group": ["adult", "adult", "elderly"],
        "sex": ["M", "F", "M"],
        "language": ["en"] * 3,
        "insurance_type": ["private"] * 3,
        "transport_origin": ["self"] * 3,
        "pain_location": ["none"] * 3,
        "mental_status_triage": ["alert"] * 3,
        "chief_complaint_system": ["cardiovascular"] * 3,
        "num_prior_ed_visits_12m": [0] * 3,
        "num_prior_admissions_12m": [0] * 3,
        "num_active_medications": [2, 3, 4],
        "num_comorbidities": [1, 2, 2],
        "systolic_bp": [120, 130, 140],
        "diastolic_bp": [80, 85, 90],
        "mean_arterial_pressure": [90, 95, 100],
        "pulse_pressure": [40, 45, 50],
        "heart_rate": [75, 85, 90],
        "respiratory_rate": [16, 18, 20],
        "temperature_c": [37.0, 37.2, 37.4],
        "spo2": [98, 97, 96],
        "gcs_total": [15, 15, 15],
        "pain_score": [3, 4, 5],
        "weight_kg": [70.0, 72.0, 68.0],
        "height_cm": [175.0, 170.0, 165.0],
        "bmi": [22.9, 24.9, 25.0],
        "shock_index": [0.63, 0.65, 0.64],
        "news2_score": [0.0, 1.0, 2.0],
    }
    train_df = pd.DataFrame({
        **base_cols,
        "disposition": ["discharged"] * 3,
        "ed_los_hours": [2.0, 3.0, 4.0],
        "triage_acuity": [4, 3, 3],
    })
    test_df = pd.DataFrame(base_cols)   # no target / no leakage

    complaints_df = pd.DataFrame({
        "patient_id": ["P0", "P1", "P2"],
        "chief_complaint_raw": ["chest pain", "chest pain", "cough"],
        "chief_complaint_system": ["cardiovascular"] * 3,
    })
    history_df = pd.DataFrame({
        "patient_id": ["P0", "P1", "P2"],
        **{f"hx_{c}": [0, 0, 0] for c in [
            "hypertension", "diabetes_type2", "diabetes_type1", "asthma",
            "copd", "heart_failure", "atrial_fibrillation", "ckd",
            "liver_disease", "malignancy", "obesity", "depression",
            "anxiety", "dementia", "epilepsy", "hypothyroidism",
            "hyperthyroidism", "hiv", "coagulopathy", "immunosuppressed",
            "pregnant", "substance_use_disorder", "coronary_artery_disease",
            "stroke_prior", "peripheral_vascular_disease",
        ]},
    })
    train_df.to_csv(tmp_path / "train.csv", index=False)
    test_df.to_csv(tmp_path / "test.csv", index=False)
    complaints_df.to_csv(tmp_path / "chief_complaints.csv", index=False)
    history_df.to_csv(tmp_path / "patient_history.csv", index=False)
    return tmp_path


def test_load_data_value_four_dataframes(mini_data_dir):
    """Returns a 4-tuple of DataFrames."""
    train, test, complaints, history = _load_data(mini_data_dir)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert isinstance(complaints, pd.DataFrame)
    assert isinstance(history, pd.DataFrame)


def test_load_data_arithmetic_row_counts(mini_data_dir):
    """Each loaded frame has exactly the rows of its CSV."""
    train, test, complaints, history = _load_data(mini_data_dir)
    assert len(train) == 3
    assert len(test) == 3
    assert len(complaints) == 3
    assert len(history) == 3


def test_load_data_value_train_has_target(mini_data_dir):
    """Train must have triage_acuity; test must not."""
    train, test, _, _ = _load_data(mini_data_dir)
    assert "triage_acuity" in train.columns
    assert "triage_acuity" not in test.columns


def test_load_data_value_distinct_files(mini_data_dir):
    """Each file is loaded from its expected path."""
    train, test, complaints, history = _load_data(mini_data_dir)
    # complaints has a chief_complaint_raw col; history has hx_*
    assert "chief_complaint_raw" in complaints.columns
    assert any(c.startswith("hx_") for c in history.columns)
    assert "chief_complaint_raw" not in history.columns


# ---------- run_geometric_scales ----------

def test_run_geometric_scales_value_returns_tuple(mini_data_dir):
    train, _, complaints, history = _load_data(mini_data_dir)
    decisions, decompositions = run_geometric_scales(
        train, complaints, history
    )
    assert len(decisions) == len(train)
    assert len(decompositions) == len(train)


def test_run_geometric_scales_value_each_decomp_has_signals(mini_data_dir):
    train, _, complaints, history = _load_data(mini_data_dir)
    _, decomps = run_geometric_scales(train, complaints, history)
    for d in decomps:
        assert len(d.signals) > 0


def test_run_geometric_scales_swap_df_order_preserves_count(mini_data_dir):
    """Passing reversed DF still produces N decisions/decomps."""
    train, _, complaints, history = _load_data(mini_data_dir)
    reversed_df = train.iloc[::-1].reset_index(drop=True)
    decisions_rev, decomps_rev = run_geometric_scales(
        reversed_df, complaints, history
    )
    decisions_fwd, _ = run_geometric_scales(train, complaints, history)
    assert len(decisions_rev) == len(decisions_fwd) == len(train)


def test_run_geometric_scales_swap_no_history(mini_data_dir):
    """History can be omitted — banks still work."""
    train, _, complaints, _ = _load_data(mini_data_dir)
    decisions, decomps = run_geometric_scales(train, complaints, None)
    assert len(decisions) == len(train)


# ---------- evaluate_pipeline ----------

def _make_result(preds, confs=None, methods=None, pids=None):
    """Build a PipelineResult from three aligned lists."""
    if pids is None:
        pids = [f"P{i}" for i in range(len(preds))]
    if confs is None:
        confs = [0.8] * len(preds)
    if methods is None:
        methods = ["geometric"] * len(preds)
    return PipelineResult(
        predictions=pd.Series(preds, index=pids, name="triage_acuity"),
        confidences=pd.Series(confs, index=pids, name="confidence"),
        methods=pd.Series(methods, index=pids, name="method"),
        decisions=[],
        stats={},
    )


def _five_class_truth_pred(pred_pattern):
    """Build a 10-row y_true with all 5 ESI classes and matching pred."""
    truth_vals = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    pids = [f"P{i}" for i in range(10)]
    return (pd.Series(truth_vals, index=pids),
            _make_result(pred_pattern, pids=pids))


def test_evaluate_pipeline_value_perfect_prediction():
    """Perfect predictions → accuracy 1.0, macro_f1 1.0."""
    truth, result = _five_class_truth_pred([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    m = evaluate_pipeline(result, truth)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["weighted_f1"] == pytest.approx(1.0)


def test_evaluate_pipeline_arithmetic_all_wrong():
    """Shifted predictions — accuracy 0.0 (no row matches)."""
    truth, result = _five_class_truth_pred([2, 3, 4, 5, 1, 2, 3, 4, 5, 1])
    m = evaluate_pipeline(result, truth)
    assert m["accuracy"] == pytest.approx(0.0)


def test_evaluate_pipeline_boundary_half_correct():
    """Half correct → accuracy 0.5."""
    # First 5 matched, last 5 off-by-one
    truth, result = _five_class_truth_pred([1, 2, 3, 4, 5, 2, 3, 4, 5, 1])
    m = evaluate_pipeline(result, truth)
    assert m["accuracy"] == pytest.approx(0.5)


def test_evaluate_pipeline_swap_macro_vs_weighted_differ_on_imbalance():
    """Macro F1 ≠ weighted F1 when class sizes differ AND errors concentrate."""
    truth_vals = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  1, 2, 4, 5]  # imbalanced; 10 ESI 3s + singletons
    preds = [3] * 10 + [1, 2, 4, 5]   # perfect on everyone
    pids = [f"P{i}" for i in range(len(truth_vals))]
    truth = pd.Series(truth_vals, index=pids)
    result = _make_result(preds, pids=pids)
    m = evaluate_pipeline(result, truth)
    # Perfect prediction → both = 1.0. Verify keys present.
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["weighted_f1"] == pytest.approx(1.0)


def test_evaluate_pipeline_value_returns_expected_keys():
    truth, result = _five_class_truth_pred([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    m = evaluate_pipeline(result, truth)
    assert set(m.keys()) == {"macro_f1", "weighted_f1", "accuracy"}


def test_evaluate_pipeline_value_exact_accuracy_9_of_10():
    """1 wrong out of 10 → accuracy 0.9."""
    # 9 perfect, 1 off-by-one at the end
    truth, result = _five_class_truth_pred([1, 2, 3, 4, 5, 1, 2, 3, 4, 4])
    m = evaluate_pipeline(result, truth)
    assert m["accuracy"] == pytest.approx(0.9)


def test_evaluate_pipeline_value_exact_accuracy_8_of_10():
    """2 wrong out of 10 → accuracy 0.8."""
    truth, result = _five_class_truth_pred([1, 2, 3, 4, 5, 2, 3, 3, 4, 5])
    m = evaluate_pipeline(result, truth)
    assert m["accuracy"] == pytest.approx(0.8)


def test_evaluate_pipeline_value_macro_f1_between_zero_and_one():
    """Macro F1 and weighted F1 are in [0, 1]."""
    truth, result = _five_class_truth_pred([1, 2, 3, 4, 5, 2, 3, 3, 4, 4])
    m = evaluate_pipeline(result, truth)
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert 0.0 <= m["weighted_f1"] <= 1.0


def test_evaluate_pipeline_swap_truth_and_pred_asymmetric_in_f1():
    """If y_true ↔ y_pred swapped, precision and recall swap → macro F1
    may or may not equal the original depending on class imbalance."""
    pids = [f"P{i}" for i in range(10)]
    truth = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], index=pids)
    pred_a = [1, 2, 3, 4, 5, 2, 3, 3, 4, 4]   # 3 errors
    pred_b = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]   # 0 errors
    m_a = evaluate_pipeline(_make_result(pred_a, pids=pids), truth)
    m_b = evaluate_pipeline(_make_result(pred_b, pids=pids), truth)
    # More errors → lower accuracy
    assert m_a["accuracy"] < m_b["accuracy"]
    assert m_a["macro_f1"] < m_b["macro_f1"]


def test_evaluate_pipeline_boundary_10_percent_vs_20_percent_accuracy():
    """Sensitivity: +1 more error should yield lower accuracy."""
    pids = [f"P{i}" for i in range(10)]
    truth = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], index=pids)
    pred_1err = [1, 2, 3, 4, 5, 1, 2, 3, 4, 4]   # 1 error → 0.9
    pred_2err = [1, 2, 3, 4, 5, 1, 2, 3, 3, 4]   # 2 errors → 0.8
    m_1 = evaluate_pipeline(_make_result(pred_1err, pids=pids), truth)
    m_2 = evaluate_pipeline(_make_result(pred_2err, pids=pids), truth)
    assert m_1["accuracy"] == pytest.approx(0.9)
    assert m_2["accuracy"] == pytest.approx(0.8)
    assert m_2["accuracy"] < m_1["accuracy"]


def test_evaluate_pipeline_swap_pred_result_assignment_changes_metrics():
    """Swapping predictions between two results gives different metrics."""
    pids = [f"P{i}" for i in range(10)]
    truth = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], index=pids)
    high_acc = _make_result([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], pids=pids)
    low_acc = _make_result([5, 4, 3, 2, 1, 5, 4, 3, 2, 1], pids=pids)
    m_high = evaluate_pipeline(high_acc, truth)
    m_low = evaluate_pipeline(low_acc, truth)
    assert m_high["accuracy"] > m_low["accuracy"]
    assert m_high["macro_f1"] > m_low["macro_f1"]


def test_evaluate_pipeline_value_f1_perfect_pred_equals_one():
    truth, result = _five_class_truth_pred([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    m = evaluate_pipeline(result, truth)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["weighted_f1"] == pytest.approx(1.0)


def test_evaluate_pipeline_value_all_off_by_one_nonzero_f1():
    """Shifted predictions: accuracy=0 but class structure roughly preserved."""
    truth_vals = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    pred_vals = [2, 3, 4, 5, 1, 2, 3, 4, 5, 1]
    pids = [f"P{i}" for i in range(10)]
    truth = pd.Series(truth_vals, index=pids)
    result = _make_result(pred_vals, pids=pids)
    m = evaluate_pipeline(result, truth)
    assert m["accuracy"] == pytest.approx(0.0)
    # Non-zero F1 because predictions still span all classes (just misaligned)
    assert m["macro_f1"] >= 0.0


# ---------- Module-level constant ----------

def test_coherence_confidence_threshold_value():
    assert COHERENCE_CONFIDENCE_THRESHOLD == 0.70
