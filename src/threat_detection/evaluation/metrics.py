"""Honest, imbalance-aware metrics.

Why these and not accuracy (interview point): the data is ~75% benign, so a
trivial "all-benign" classifier scores ~75% accuracy while catching zero
attacks. We therefore headline:
  * PR-AUC (average precision) — focuses on the minority attack class;
  * Recall — a missed attack is the expensive error in security;
  * MCC — a single balanced number that stays honest under heavy imbalance;
  * per-class recall — aggregate metrics hide a model blind to rare classes
    (e.g. Infiltration).
Threshold selection is a *business* decision: pick the operating point on the
PR curve that hits a target recall, then report the resulting precision/alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from threat_detection.data import schema


def pr_auc(y_true_binary: np.ndarray, attack_scores: np.ndarray) -> float:
    """Area under the precision-recall curve (a.k.a. average precision)."""
    return float(average_precision_score(y_true_binary, attack_scores))


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation coefficient."""
    return float(matthews_corrcoef(y_true, y_pred))


def per_class_recall(y_true_labels: np.ndarray, y_pred_labels: np.ndarray) -> dict[str, float]:
    """Recall for each label in the canonical class space (0.0 if class absent)."""
    out: dict[str, float] = {}
    for label in schema.ALL_LABELS:
        mask = y_true_labels == label
        n = int(mask.sum())
        if n == 0:
            continue
        out[label] = float((y_pred_labels[mask] == label).sum() / n)
    return out


@dataclass
class ThresholdChoice:
    threshold: float
    precision: float
    recall: float


def select_threshold_for_recall(
    y_true_binary: np.ndarray, attack_scores: np.ndarray, target_recall: float
) -> ThresholdChoice:
    """Pick the highest-precision threshold that still achieves >= target recall.

    This encodes the security trade-off: we fix a recall floor (don't miss
    attacks) and then maximise precision (minimise analyst alert volume).
    """
    precision, recall, thresholds = precision_recall_curve(y_true_binary, attack_scores)
    # precision_recall_curve returns len(thresholds)+1 points; align by dropping
    # the last precision/recall entry (which corresponds to recall=0 / no threshold).
    precision, recall = precision[:-1], recall[:-1]
    eligible = recall >= target_recall
    if not eligible.any():
        # Can't reach target recall; fall back to the max-recall point.
        idx = int(np.argmax(recall))
    else:
        # Among thresholds meeting the recall floor, take the most precise.
        masked_precision = np.where(eligible, precision, -np.inf)
        idx = int(np.argmax(masked_precision))
    return ThresholdChoice(
        threshold=float(thresholds[idx]),
        precision=float(precision[idx]),
        recall=float(recall[idx]),
    )


@dataclass
class TabularMetrics:
    """Headline metrics for one evaluation, plus the chosen operating point."""

    pr_auc: float
    mcc: float
    recall_at_threshold: float
    precision_at_threshold: float
    threshold: float
    per_class_recall: dict[str, float] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, float]:
        """Flatten for MLflow logging (per-class recall keyed as recall_<class>)."""
        flat: dict[str, float] = {
            "pr_auc": self.pr_auc,
            "mcc": self.mcc,
            "recall_at_threshold": self.recall_at_threshold,
            "precision_at_threshold": self.precision_at_threshold,
            "threshold": self.threshold,
        }
        for label, value in self.per_class_recall.items():
            flat[f"recall_{label.replace(' ', '_')}"] = value
        if self.per_class_recall:
            flat["min_per_class_recall"] = min(self.per_class_recall.values())
        return flat


def evaluate_tabular(
    y_true_labels: np.ndarray,
    y_pred_labels: np.ndarray,
    y_true_binary: np.ndarray,
    attack_scores: np.ndarray,
    target_recall: float,
) -> TabularMetrics:
    """Compute the full headline metric set for the tabular model."""
    choice = select_threshold_for_recall(y_true_binary, attack_scores, target_recall)
    y_pred_binary = (attack_scores >= choice.threshold).astype(int)
    return TabularMetrics(
        pr_auc=pr_auc(y_true_binary, attack_scores),
        mcc=mcc(y_true_binary, y_pred_binary),
        recall_at_threshold=float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        precision_at_threshold=float(
            precision_score(y_true_binary, y_pred_binary, zero_division=0)
        ),
        threshold=choice.threshold,
        per_class_recall=per_class_recall(y_true_labels, y_pred_labels),
    )
