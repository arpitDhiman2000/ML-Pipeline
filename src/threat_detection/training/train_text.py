"""Train the LSTM text classifier with a TF-IDF baseline bake-off.

End to end: load the processed text splits, encode with the frozen tokenizer
artifact, train the LSTM (class-weighted, early-stopped, per-epoch curves logged
to MLflow), pick the operating threshold on val for the target recall, evaluate
on test, and bake it off against the TF-IDF baseline. The LSTM must beat the
baseline on F1 to justify itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from threat_detection.config import AppConfig, get_config
from threat_detection.data import schema
from threat_detection.evaluation.metrics import pr_auc, select_threshold_for_recall
from threat_detection.features.text import TextTokenizer
from threat_detection.logging_utils import get_logger
from threat_detection.models.text_baseline import TfidfBaseline
from threat_detection.models.text_lstm import TextClassifier
from threat_detection.paths import Paths
from threat_detection.registry import TEXT_MODEL
from threat_detection.tracking import configure_mlflow

log = get_logger(__name__)

REGISTERED_MODEL_NAME = TEXT_MODEL
EXPERIMENT = "threat-detection-text"


def _load_split(name: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_parquet(Paths.processed_file(f"text_{name}"))
    texts = df[schema.TEXT_COLUMN].astype(str).tolist()
    y = df[schema.TEXT_LABEL_COLUMN].to_numpy(dtype=np.int64)
    return texts, y


def _safe_log_pytorch(net, register: bool) -> None:
    import mlflow

    name = REGISTERED_MODEL_NAME if register else None
    try:
        mlflow.pytorch.log_model(net, artifact_path="text_lstm", registered_model_name=name)
    except Exception as exc:  # registry support varies by backend (e.g. DagsHub)
        log.warning("mlflow.register_failed", error=str(exc), action="log_without_registry")
        mlflow.pytorch.log_model(net, artifact_path="text_lstm")


def train(config: AppConfig | None = None, *, register: bool = True) -> dict[str, float]:
    """Train + bake off the text models. Returns the headline metrics."""
    cfg = config or get_config()

    train_texts, y_train = _load_split("train")
    val_texts, y_val = _load_split("val")
    test_texts, y_test = _load_split("test")

    tokenizer = TextTokenizer.load(Paths.tokenizer_artifact())
    log.info("text.train.start", train_rows=len(train_texts), vocab=tokenizer.vocab_size)

    configure_mlflow(EXPERIMENT)
    import mlflow

    with mlflow.start_run(run_name="text-lstm") as run:
        mlflow.log_params(
            {"seed": cfg.seed, "vocab_size": tokenizer.vocab_size, **cfg.lstm.model_dump()}
        )

        # --- LSTM ---
        clf = TextClassifier(cfg.lstm, tokenizer, seed=cfg.seed)
        clf.fit(train_texts, y_train, val_texts=val_texts, val_y=y_val)
        history = clf.history_
        for epoch, tl in enumerate(history["train_loss"]):
            mlflow.log_metric("train_loss", tl, step=epoch)
        for epoch, vl in enumerate(history["val_loss"]):
            mlflow.log_metric("val_loss", vl, step=epoch)

        # operating threshold chosen on val for the target recall
        val_scores = clf.predict_proba(val_texts)
        choice = select_threshold_for_recall(y_val, val_scores, cfg.lstm.operating_recall)
        clf.threshold = choice.threshold

        # --- evaluate LSTM on test ---
        test_scores = clf.predict_proba(test_texts)
        lstm_pred = (test_scores >= clf.threshold).astype(int)
        lstm_f1 = float(f1_score(y_test, lstm_pred, zero_division=0))
        lstm_pr_auc = pr_auc(y_test, test_scores)

        # --- baseline bake-off ---
        base = TfidfBaseline(seed=cfg.seed).fit(train_texts, y_train)
        base_pred = base.predict(test_texts)
        base_f1 = float(f1_score(y_test, base_pred, zero_division=0))
        base_pr_auc = pr_auc(y_test, base.predict_proba(test_texts))

        metrics = {
            "lstm_f1": lstm_f1,
            "lstm_pr_auc": lstm_pr_auc,
            "lstm_threshold": clf.threshold,
            "baseline_f1": base_f1,
            "baseline_pr_auc": base_pr_auc,
            "f1_uplift_vs_baseline": lstm_f1 - base_f1,
        }
        mlflow.log_metrics(metrics)
        mlflow.set_tag("beats_baseline", str(lstm_f1 >= base_f1))
        log.info("text.train.metrics", **metrics)

        # --- persist + register ---
        clf.save(Paths.text_model())
        _safe_log_pytorch(clf.net, register=register)
        mlflow.log_dict(metrics, "text_metrics.json")
        log.info("text.train.done", run_id=run.info.run_id)

    return metrics
