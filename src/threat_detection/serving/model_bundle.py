"""The serving model bundle: load all artifacts once, score an event.

Serving loads the *wrapper artifacts* saved at training time (the source of truth,
DVC-versioned and baked into the image in Sprint 5). The MLflow ``@production``
alias is the governance record of which version those bytes correspond to; it is
surfaced via /model-info. Loading once at startup keeps per-request latency low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threat_detection.config import AppConfig, get_config
from threat_detection.features import target
from threat_detection.features.tabular import TabularPreprocessor
from threat_detection.features.text import TextTokenizer
from threat_detection.logging_utils import get_logger
from threat_detection.models.anomaly import AnomalyDetector
from threat_detection.models.classifier import AttackClassifier
from threat_detection.models.fusion import FusionMetaLearner
from threat_detection.models.text_lstm import TextClassifier
from threat_detection.paths import Paths
from threat_detection.serving import explain

log = get_logger(__name__)


class ModelBundle:
    """All fitted artifacts + the scoring logic that fuses them."""

    def __init__(
        self,
        *,
        preprocessor: TabularPreprocessor,
        anomaly: AnomalyDetector,
        classifier: AttackClassifier,
        text: TextClassifier,
        fusion: FusionMetaLearner,
        config: AppConfig,
    ):
        self.preprocessor = preprocessor
        self.anomaly = anomaly
        self.classifier = classifier
        self.text = text
        self.fusion = fusion
        self.config = config

    @classmethod
    def load(cls, config: AppConfig | None = None) -> ModelBundle:
        cfg = config or get_config()
        tokenizer = TextTokenizer.load(Paths.tokenizer_artifact())
        bundle = cls(
            preprocessor=TabularPreprocessor.load(Paths.preprocessor_artifact()),
            anomaly=AnomalyDetector.load(Paths.anomaly_model()),
            classifier=AttackClassifier.load(Paths.classifier_model()),
            text=TextClassifier.load(Paths.text_model(), tokenizer),
            fusion=FusionMetaLearner.load(Paths.fusion_model()),
            config=cfg,
        )
        log.info("bundle.loaded")
        return bundle

    def _tabular_scores(self, features: dict[str, float] | None):
        """Return (anomaly_score, attack_proba, predicted_label, scaled_row|None)."""
        if not features:
            return 0.0, 0.0, "UNKNOWN", None
        X = self.preprocessor.transform(pd.DataFrame([features]))
        anomaly_score = float(self.anomaly.anomaly_score(X)[0])
        attack_proba = float(self.classifier.attack_proba(X)[0])
        class_id = int(self.classifier.predict(X)[0])
        label = str(target.decode_labels(np.array([class_id]))[0])
        return anomaly_score, attack_proba, label, X[0]

    def score(self, features: dict[str, float] | None, payload: str | None) -> dict:
        """Score one event into the ScoreResponse shape (minus latency).

        Decision routing (graceful degradation): the fusion meta-learner was
        trained with all three scores present, so feeding fabricated "benign"
        defaults for an absent modality would let it override a genuine signal
        from the modality that IS present. Instead, when only one modality is
        available we defer to that model's own decision.
        """
        has_tabular = bool(features)
        has_text = payload is not None

        anomaly_score, attack_proba, predicted_label, scaled_row = self._tabular_scores(features)
        text_proba = (
            float(self.text.predict_proba([payload])[0])
            if has_text
            else self.config.fusion.missing_text_score
        )

        if has_tabular and has_text:
            # Both signals -> the learned meta-learner fuses them.
            fusion_in = np.array([[anomaly_score, attack_proba, text_proba]])
            threat_prob = float(self.fusion.predict_proba(fusion_in)[0])
            threshold = self.config.fusion.decision_threshold
            decision_source = "fusion"
        elif has_text:
            # Payload only -> the text classifier decides on its own threshold.
            threat_prob = text_proba
            threshold = self.text.threshold
            decision_source = "text-only"
        else:
            # Flow only -> fuse tabular signals with a neutral text prior; the
            # tabular signals (real here) drive the meta-learner.
            fusion_in = np.array([[anomaly_score, attack_proba, text_proba]])
            threat_prob = float(self.fusion.predict_proba(fusion_in)[0])
            threshold = self.config.fusion.decision_threshold
            decision_source = "tabular-only"

        is_threat = threat_prob >= threshold

        top_features: list[dict] = []
        if scaled_row is not None:
            top_features = explain.tabular_top_features(
                self.classifier.model.feature_importances_,
                scaled_row,
                self.config.serving.top_k_explanations,
            )

        return {
            "threat_probability": round(threat_prob, 6),
            "is_threat": bool(is_threat),
            # Confidence = how far the probability is from the 0.5 midpoint
            # (model certainty), independent of the business decision threshold.
            "confidence": round(min(abs(threat_prob - 0.5) * 2.0, 1.0), 6),
            "predicted_class": predicted_label,
            "decision_source": decision_source,
            "component_scores": {
                "anomaly_score": round(anomaly_score, 6),
                "attack_proba": round(attack_proba, 6),
                "text_proba": round(text_proba, 6),
            },
            "explanation": {
                "top_features": top_features,
                "triggering_indicators": explain.payload_indicators(payload),
            },
        }
