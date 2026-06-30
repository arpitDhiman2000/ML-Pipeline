"""The evaluation gate — the automated promotion rule.

This is the single most reused decision in the project: the same gate is run
(1) at training time to decide whether to register a model, (2) in CI (Sprint 5)
to block a deploy on regression, and (3) in the retraining loop (Sprint 6) to
decide whether a drift-triggered candidate may replace production.

A candidate is promotable iff:
  * PR-AUC >= floor, AND
  * the worst per-class recall >= floor (no class is silently ignored), AND
  * it does not regress PR-AUC vs the incumbent (if one is provided), AND
  * p99 latency <= budget (only checked once latency is supplied, Sprint 4+).

Encoding it as data (not scattered ``if`` statements) is what makes it testable
and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from threat_detection.config import EvalGateConfig


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed


def evaluate_gate(
    metrics: dict[str, float],
    config: EvalGateConfig,
    *,
    incumbent: dict[str, float] | None = None,
    p99_latency_ms: float | None = None,
) -> GateDecision:
    """Apply the promotion rule to a candidate's metrics.

    Args:
        metrics: Candidate metrics (must include ``pr_auc``; ``min_per_class_recall``
            is used if present).
        config: Gate thresholds.
        incumbent: Current production metrics, if any, for regression checks.
        p99_latency_ms: Measured p99 latency; skipped when None (pre-serving).

    Returns:
        A :class:`GateDecision` with a human-readable reason per check.
    """
    reasons: list[str] = []
    passed = True

    pr = metrics.get("pr_auc", 0.0)
    if pr >= config.min_pr_auc:
        reasons.append(f"PASS pr_auc {pr:.3f} >= {config.min_pr_auc}")
    else:
        passed = False
        reasons.append(f"FAIL pr_auc {pr:.3f} < {config.min_pr_auc}")

    if "min_per_class_recall" in metrics:
        mpcr = metrics["min_per_class_recall"]
        if mpcr >= config.min_per_class_recall:
            reasons.append(f"PASS min_per_class_recall {mpcr:.3f} >= {config.min_per_class_recall}")
        else:
            passed = False
            reasons.append(f"FAIL min_per_class_recall {mpcr:.3f} < {config.min_per_class_recall}")

    if incumbent is not None:
        inc_pr = incumbent.get("pr_auc", 0.0)
        if pr >= inc_pr:
            reasons.append(f"PASS no PR-AUC regression ({pr:.3f} >= incumbent {inc_pr:.3f})")
        else:
            passed = False
            reasons.append(f"FAIL PR-AUC regression ({pr:.3f} < incumbent {inc_pr:.3f})")

    if p99_latency_ms is not None:
        if p99_latency_ms <= config.max_p99_latency_ms:
            reasons.append(f"PASS p99 {p99_latency_ms:.1f}ms <= {config.max_p99_latency_ms}ms")
        else:
            passed = False
            reasons.append(f"FAIL p99 {p99_latency_ms:.1f}ms > {config.max_p99_latency_ms}ms")

    return GateDecision(passed=passed, reasons=reasons)
