"""Train the tabular path (Isolation Forest + XGBoost) with MLflow tracking.

End to end: load the leak-free processed splits, fit both models, evaluate with
the honest metric set, log everything to MLflow (DagsHub when configured), save
native artifacts for serving, register the classifier in the MLflow Model
Registry, and run the eval gate. The gate result is logged but never crashes
training — promotion is a *decision*, surfaced for CI/retraining to act on.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from threat_detection.config import AppConfig, get_config
from threat_detection.data import schema
from threat_detection.evaluation.gate import evaluate_gate
from threat_detection.evaluation.metrics import TabularMetrics, evaluate_tabular
from threat_detection.features import target
from threat_detection.logging_utils import get_logger
from threat_detection.models.anomaly import AnomalyDetector
from threat_detection.models.classifier import AttackClassifier
from threat_detection.paths import Paths
from threat_detection.tracking import configure_mlflow

log = get_logger(__name__)

REGISTERED_MODEL_NAME = "threat-detection-tabular"
EXPERIMENT = "threat-detection-tabular"


def _load_split(name: str):
    df = pd.read_parquet(Paths.processed_file(f"tabular_{name}"))
    X = df[list(schema.FLOW_FEATURES)].to_numpy(dtype=np.float32)
    labels = df[schema.LABEL_COLUMN].to_numpy()
    y_label = target.encode_labels(df[schema.LABEL_COLUMN])
    y_bin = df[schema.BINARY_TARGET_COLUMN].to_numpy()
    return X, labels, y_label, y_bin


def _safe_log_model(model_obj, flavor: str, artifact_path: str, register: bool) -> None:
    """Log (and optionally register) a model, tolerating registry-less backends.

    DagsHub's MLflow may not support the model registry; if registration fails we
    fall back to logging the model as a plain artifact so training never breaks.
    """
    import mlflow

    flavor_mod = getattr(mlflow, flavor)
    name = REGISTERED_MODEL_NAME if register else None
    try:
        flavor_mod.log_model(model_obj, artifact_path=artifact_path, registered_model_name=name)
    except Exception as exc:  # registry support varies by backend (e.g. DagsHub)
        log.warning("mlflow.register_failed", error=str(exc), action="log_without_registry")
        flavor_mod.log_model(model_obj, artifact_path=artifact_path)


def train(config: AppConfig | None = None, *, register: bool = True) -> TabularMetrics:
    """Train + evaluate + log the tabular models. Returns the test metrics."""
    cfg = config or get_config()

    X_train, _, y_label_train, y_bin_train = _load_split("train")
    X_val, _, y_label_val, _ = _load_split("val")
    X_test, labels_test, _, y_bin_test = _load_split("test")

    prevalence = float(y_bin_train.mean())
    log.info("train.start", train_rows=len(X_train), attack_prevalence=prevalence)

    configure_mlflow(EXPERIMENT)
    import mlflow

    with mlflow.start_run(run_name="tabular") as run:
        # --- params ---
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "attack_prevalence": round(prevalence, 4),
                **{f"if_{k}": v for k, v in cfg.isolation_forest.model_dump().items()},
                **{f"xgb_{k}": v for k, v in cfg.xgboost.model_dump().items()},
            }
        )

        # --- Isolation Forest (contamination tied to prevalence) ---
        anomaly = AnomalyDetector.from_prevalence(
            cfg.isolation_forest, attack_prevalence=prevalence, seed=cfg.seed
        )
        anomaly.fit(X_train)
        mlflow.log_param("if_contamination_resolved", anomaly.contamination)

        # --- XGBoost (class-weighted, early-stopped on val) ---
        clf = AttackClassifier(cfg.xgboost, seed=cfg.seed)
        clf.fit(X_train, y_label_train, X_val=X_val, y_val=y_label_val)

        # --- evaluate on the held-out test split ---
        attack_scores = clf.attack_proba(X_test)
        pred_labels = target.decode_labels(clf.predict(X_test))
        metrics = evaluate_tabular(
            y_true_labels=labels_test,
            y_pred_labels=pred_labels,
            y_true_binary=y_bin_test,
            attack_scores=attack_scores,
            target_recall=cfg.xgboost.operating_recall,
        )
        mlflow.log_metrics(metrics.to_flat_dict())
        log.info("train.metrics", **metrics.to_flat_dict())

        # --- persist native artifacts (the source of truth for serving) ---
        anomaly.save(Paths.anomaly_model())
        clf.save(Paths.classifier_model())

        # --- log/register models in MLflow ---
        _safe_log_model(clf.model, "xgboost", "classifier", register=register)
        _safe_log_model(anomaly.model, "sklearn", "anomaly", register=False)

        # --- eval gate (decision only; logged, never fatal) ---
        decision = evaluate_gate(metrics.to_flat_dict(), cfg.eval_gate)
        mlflow.set_tag("gate_passed", str(decision.passed))
        mlflow.log_dict({"passed": decision.passed, "reasons": decision.reasons}, "gate.json")
        log.info("train.gate", passed=decision.passed, reasons=decision.reasons)

        mlflow.log_dict(asdict(metrics), "metrics.json")
        log.info("train.done", run_id=run.info.run_id)

    return metrics
