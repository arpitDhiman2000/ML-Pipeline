"""Re-evaluate the trained tabular model and apply the eval gate.

This is the CI promotion check: it loads the saved classifier + the held-out
test split, recomputes the honest metric set, and runs :func:`evaluate_gate`.
CI calls it and fails the build (non-zero exit) when the gate fails — that is
literally how a red gate blocks the deploy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threat_detection.config import AppConfig, get_config
from threat_detection.data import schema
from threat_detection.evaluation.gate import GateDecision, evaluate_gate
from threat_detection.evaluation.metrics import TabularMetrics, evaluate_tabular
from threat_detection.features import target
from threat_detection.models.classifier import AttackClassifier
from threat_detection.paths import Paths


def evaluate_current_model(config: AppConfig | None = None) -> tuple[TabularMetrics, GateDecision]:
    """Score the saved model on the test split and run the gate."""
    cfg = config or get_config()
    df = pd.read_parquet(Paths.processed_file("tabular_test"))
    X = df[list(schema.FLOW_FEATURES)].to_numpy(dtype=np.float32)

    clf = AttackClassifier.load(Paths.classifier_model())
    attack_scores = clf.attack_proba(X)
    pred_labels = target.decode_labels(clf.predict(X))

    metrics = evaluate_tabular(
        y_true_labels=df[schema.LABEL_COLUMN].to_numpy(),
        y_pred_labels=pred_labels,
        y_true_binary=df[schema.BINARY_TARGET_COLUMN].to_numpy(),
        attack_scores=attack_scores,
        target_recall=cfg.xgboost.operating_recall,
    )
    decision = evaluate_gate(metrics.to_flat_dict(), cfg.eval_gate)
    return metrics, decision
