"""Train the fusion meta-learner over the three model scores.

Alignment caveat (documented honestly): the tabular and text corpora are
independent, so there is no natural (flow, payload) pairing. We construct a
synthetic JOINT set: each flow is paired with a payload whose maliciousness is
correlated with the flow's label (an attack flow more often co-occurs with a
malicious command). This gives the meta-learner a genuinely informative
3-vector. The label is the flow's ground-truth ``is_attack``.

We then compare the learned meta-learner's PR-AUC against naive score averaging
to justify the stacked layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threat_detection.config import AppConfig, get_config
from threat_detection.data import schema
from threat_detection.evaluation.metrics import pr_auc
from threat_detection.features.text import TextTokenizer
from threat_detection.logging_utils import get_logger
from threat_detection.models.anomaly import AnomalyDetector
from threat_detection.models.classifier import AttackClassifier
from threat_detection.models.fusion import FusionMetaLearner
from threat_detection.models.text_lstm import TextClassifier
from threat_detection.paths import Paths
from threat_detection.registry import FUSION_MODEL
from threat_detection.tracking import configure_mlflow

log = get_logger(__name__)
EXPERIMENT = "threat-detection-fusion"


def _pair_payloads(
    y_bin: np.ndarray,
    text_df: pd.DataFrame,
    cfg: AppConfig,
    rng: np.random.Generator,
) -> list[str]:
    """Pick a payload per flow, correlating maliciousness with the flow label."""
    mal = text_df.loc[text_df[schema.TEXT_LABEL_COLUMN] == 1, schema.TEXT_COLUMN].to_numpy()
    ben = text_df.loc[text_df[schema.TEXT_LABEL_COLUMN] == 0, schema.TEXT_COLUMN].to_numpy()
    payloads: list[str] = []
    for is_attack in y_bin:
        p_mal = cfg.fusion.p_mal_given_attack if is_attack else cfg.fusion.p_mal_given_benign
        pool = mal if rng.random() < p_mal else ben
        payloads.append(str(rng.choice(pool)))
    return payloads


def _build_scores(
    X: np.ndarray,
    payloads: list[str],
    anomaly: AnomalyDetector,
    clf: AttackClassifier,
    text: TextClassifier,
) -> np.ndarray:
    """Assemble the [anomaly_score, attack_proba, text_proba] matrix."""
    return np.column_stack(
        [anomaly.anomaly_score(X), clf.attack_proba(X), text.predict_proba(payloads)]
    )


def _load_split(name: str):
    df = pd.read_parquet(Paths.processed_file(f"tabular_{name}"))
    X = df[list(schema.FLOW_FEATURES)].to_numpy(dtype=np.float32)
    y_bin = df[schema.BINARY_TARGET_COLUMN].to_numpy()
    return X, y_bin


def train(config: AppConfig | None = None, *, register: bool = True) -> dict[str, float]:
    """Train + evaluate the fusion meta-learner. Returns headline metrics."""
    cfg = config or get_config()
    rng = np.random.default_rng(cfg.seed)

    # Load the already-trained component models (saved during Sprints 2-3).
    tokenizer = TextTokenizer.load(Paths.tokenizer_artifact())
    anomaly = AnomalyDetector.load(Paths.anomaly_model())
    clf = AttackClassifier.load(Paths.classifier_model())
    text = TextClassifier.load(Paths.text_model(), tokenizer)
    text_df = pd.read_parquet(Paths.processed_file("text_train"))

    X_train, y_train = _load_split("train")
    X_test, y_test = _load_split("test")

    train_scores = _build_scores(
        X_train, _pair_payloads(y_train, text_df, cfg, rng), anomaly, clf, text
    )
    test_scores = _build_scores(
        X_test, _pair_payloads(y_test, text_df, cfg, rng), anomaly, clf, text
    )

    configure_mlflow(EXPERIMENT)
    import mlflow

    with mlflow.start_run(run_name="fusion") as run:
        mlflow.log_params(cfg.fusion.model_dump())

        fusion = FusionMetaLearner(cfg.fusion, seed=cfg.seed).fit(train_scores, y_train)

        meta_scores = fusion.predict_proba(test_scores)
        avg_scores = FusionMetaLearner.average_baseline(test_scores)
        metrics = {
            "fusion_pr_auc": pr_auc(y_test, meta_scores),
            "averaging_pr_auc": pr_auc(y_test, avg_scores),
        }
        metrics["pr_auc_uplift_vs_averaging"] = (
            metrics["fusion_pr_auc"] - metrics["averaging_pr_auc"]
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tag(
            "beats_averaging", str(metrics["fusion_pr_auc"] >= metrics["averaging_pr_auc"])
        )
        log.info("fusion.metrics", **metrics)

        fusion.save(Paths.fusion_model())
        try:
            mlflow.sklearn.log_model(
                fusion.pipeline,
                artifact_path="fusion",
                registered_model_name=FUSION_MODEL if register else None,
            )
        except Exception as exc:  # registry support varies by backend
            log.warning("mlflow.register_failed", error=str(exc))
            mlflow.sklearn.log_model(fusion.pipeline, artifact_path="fusion")

        mlflow.log_dict(metrics, "fusion_metrics.json")
        log.info("fusion.done", run_id=run.info.run_id)

    return metrics
