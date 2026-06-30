"""Unit tests for the eval gate (the promotion rule)."""

from __future__ import annotations

import pytest

from threat_detection.config import EvalGateConfig
from threat_detection.evaluation.gate import evaluate_gate

pytestmark = pytest.mark.unit


@pytest.fixture
def gate_cfg() -> EvalGateConfig:
    return EvalGateConfig(min_pr_auc=0.80, min_per_class_recall=0.50, max_p99_latency_ms=50)


def test_passes_when_all_thresholds_met(gate_cfg: EvalGateConfig) -> None:
    decision = evaluate_gate({"pr_auc": 0.9, "min_per_class_recall": 0.7}, gate_cfg)
    assert decision.passed
    assert bool(decision) is True


def test_fails_on_low_pr_auc(gate_cfg: EvalGateConfig) -> None:
    decision = evaluate_gate({"pr_auc": 0.5, "min_per_class_recall": 0.7}, gate_cfg)
    assert not decision.passed
    assert any("FAIL pr_auc" in r for r in decision.reasons)


def test_fails_on_low_per_class_recall(gate_cfg: EvalGateConfig) -> None:
    decision = evaluate_gate({"pr_auc": 0.9, "min_per_class_recall": 0.1}, gate_cfg)
    assert not decision.passed
    assert any("min_per_class_recall" in r and "FAIL" in r for r in decision.reasons)


def test_fails_on_regression_vs_incumbent(gate_cfg: EvalGateConfig) -> None:
    decision = evaluate_gate(
        {"pr_auc": 0.82, "min_per_class_recall": 0.7},
        gate_cfg,
        incumbent={"pr_auc": 0.88},
    )
    assert not decision.passed
    assert any("regression" in r for r in decision.reasons)


def test_latency_budget(gate_cfg: EvalGateConfig) -> None:
    ok = evaluate_gate({"pr_auc": 0.9, "min_per_class_recall": 0.7}, gate_cfg, p99_latency_ms=30)
    bad = evaluate_gate({"pr_auc": 0.9, "min_per_class_recall": 0.7}, gate_cfg, p99_latency_ms=80)
    assert ok.passed
    assert not bad.passed
