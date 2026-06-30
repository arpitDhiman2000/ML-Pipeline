"""Unit tests for the honest metric set."""

from __future__ import annotations

import numpy as np
import pytest

from threat_detection.evaluation import metrics

pytestmark = pytest.mark.unit


def test_pr_auc_perfect_separation() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.pr_auc(y, scores) == pytest.approx(1.0)


def test_mcc_perfect() -> None:
    y = np.array([0, 1, 0, 1])
    assert metrics.mcc(y, y) == pytest.approx(1.0)


def test_per_class_recall_counts() -> None:
    y_true = np.array(["BENIGN", "DoS", "DoS", "PortScan"])
    y_pred = np.array(["BENIGN", "DoS", "BENIGN", "PortScan"])
    rec = metrics.per_class_recall(y_true, y_pred)
    assert rec["BENIGN"] == pytest.approx(1.0)
    assert rec["DoS"] == pytest.approx(0.5)
    assert rec["PortScan"] == pytest.approx(1.0)
    assert "DDoS" not in rec  # absent class is omitted, not 0


def test_threshold_meets_target_recall() -> None:
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(800), np.ones(200)]).astype(int)
    scores = np.concatenate([rng.uniform(0, 0.6, 800), rng.uniform(0.4, 1.0, 200)])
    choice = metrics.select_threshold_for_recall(y, scores, target_recall=0.80)
    assert choice.recall >= 0.80 - 1e-9


def test_evaluate_tabular_flat_dict_has_min_recall() -> None:
    y_labels = np.array(["BENIGN", "DoS", "DDoS", "BENIGN"])
    pred_labels = np.array(["BENIGN", "DoS", "DDoS", "BENIGN"])
    y_bin = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    m = metrics.evaluate_tabular(y_labels, pred_labels, y_bin, scores, target_recall=0.5)
    flat = m.to_flat_dict()
    assert "pr_auc" in flat
    assert "min_per_class_recall" in flat
    assert flat["min_per_class_recall"] == pytest.approx(1.0)
