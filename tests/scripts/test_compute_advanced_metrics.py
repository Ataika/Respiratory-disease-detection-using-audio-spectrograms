"""Regression tests for the pure-math functions in
scripts/compute_advanced_metrics.py.

Loaded by file path (scripts/ isn't a package) rather than restructuring
the operational script into an importable module.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "compute_advanced_metrics.py"

spec = importlib.util.spec_from_file_location("compute_advanced_metrics", SCRIPT_PATH)
cam = importlib.util.module_from_spec(spec)
sys.modules["compute_advanced_metrics"] = cam
spec.loader.exec_module(cam)


def test_icbhi_score_counts_any_abnormal_class_as_a_correct_detection():
    """A crackle cycle predicted as wheeze must still count as a correct
    Se detection under the official binary ICBHI scoring convention —
    this was a real bug: the previous implementation required the exact
    4-class label to match, which silently penalized Se for any
    cross-abnormal-class confusion.
    """
    targets = np.array([0, 0, 1, 2, 3])
    preds = np.array([0, 1, 2, 3, 0])
    # true=0 -> pred=0 (correct, normal); true=0 -> pred=1 (wrong, normal)
    # true=1 -> pred=2 (abnormal predicted as different abnormal: correct Se)
    # true=2 -> pred=3 (abnormal predicted as different abnormal: correct Se)
    # true=3 -> pred=0 (abnormal predicted as normal: wrong Se)
    result = cam.icbhi_score(targets, preds)
    assert result["specificity_sp"] == pytest.approx(0.5)   # 1/2 normals correct
    assert result["sensitivity_se"] == pytest.approx(2 / 3)  # 2/3 abnormals correct
    assert result["icbhi_score"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_icbhi_score_perfect_predictions():
    targets = np.array([0, 0, 1, 2, 3])
    preds = np.array([0, 0, 1, 2, 3])
    result = cam.icbhi_score(targets, preds)
    assert result["sensitivity_se"] == pytest.approx(1.0)
    assert result["specificity_sp"] == pytest.approx(1.0)
    assert result["icbhi_score"] == pytest.approx(1.0)


def test_icbhi_score_all_predicted_normal():
    """Predicting `normal` for everything should give Se=0, Sp=1 —
    exactly the "always predict the majority class" failure mode this
    metric exists to catch.
    """
    targets = np.array([0, 0, 1, 2, 3])
    preds = np.array([0, 0, 0, 0, 0])
    result = cam.icbhi_score(targets, preds)
    assert result["sensitivity_se"] == pytest.approx(0.0)
    assert result["specificity_sp"] == pytest.approx(1.0)
    assert result["icbhi_score"] == pytest.approx(0.5)


def test_expected_calibration_error_zero_for_perfect_calibration():
    """If confidence always equals actual accuracy within a bin, ECE
    should be ~0. Confidence is read as probs[i, preds[i]], so every
    row's probability at its own predicted index must actually equal
    the intended confidence.
    """
    targets = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 0])
    # 2/4 correct, all four predictions made at 0.5 confidence
    probs = np.array([
        [0.5, 0.5, 0.0, 0.0],  # pred=0 (correct), confidence 0.5
        [0.5, 0.5, 0.0, 0.0],  # pred=1 (wrong),   confidence 0.5
        [0.0, 0.5, 0.5, 0.0],  # pred=1 (correct), confidence 0.5
        [0.5, 0.5, 0.0, 0.0],  # pred=0 (wrong),   confidence 0.5
    ])
    result = cam.expected_calibration_error(targets, preds, probs, n_bins=10)
    # accuracy in the (0.4, 0.5] bin = 2/4 = 0.5 = avg confidence -> ECE = 0
    assert result["ece"] == pytest.approx(0.0, abs=1e-9)


def test_mcnemar_identical_predictions_gives_p_one():
    targets = np.array([0, 1, 2, 3, 0])
    preds_a = np.array([0, 1, 2, 3, 1])   # 4/5 correct
    preds_b = np.array([0, 1, 2, 3, 1])   # identical -> no discordant pairs
    result = cam.mcnemar_test(targets, preds_a, preds_b)
    assert result["b_a_right_b_wrong"] == 0
    assert result["c_a_wrong_b_right"] == 0
    assert result["p_value"] == pytest.approx(1.0)


def test_mcnemar_detects_discordant_pairs():
    """A wins on 2 samples where B is wrong; B never wins where A is
    wrong -> b=2, c=0, and this asymmetry should not vanish silently.
    """
    targets = np.array([0, 1, 2, 3, 0])
    preds_a = np.array([0, 1, 2, 3, 0])   # 5/5 correct
    preds_b = np.array([0, 1, 2, 3, 1])   # 4/5 correct, wrong exactly where A is right
    result = cam.mcnemar_test(targets, preds_a, preds_b)
    assert result["b_a_right_b_wrong"] == 1
    assert result["c_a_wrong_b_right"] == 0
