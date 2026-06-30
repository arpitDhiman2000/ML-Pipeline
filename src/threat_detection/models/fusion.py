"""Stacked meta-learner that fuses the three scores into one decision.

Why a learned meta-layer over simple averaging (interview point): the three
scorers live on different scales and have different reliabilities — the Isolation
Forest emits an unbounded anomaly score, XGBoost a calibrated attack probability,
the LSTM a text probability. Averaging them treats unequal, unaligned signals as
equal. A logistic meta-learner (standardise → logistic regression) learns the
right weighting from data. ``compare_to_averaging`` quantifies the lift.

Feature order is fixed and explicit so train and serve always agree.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from threat_detection.config import FusionConfig

ARTIFACT_VERSION = 1
FUSION_FEATURES = ("anomaly_score", "attack_proba", "text_proba")


class FusionMetaLearner:
    """Logistic meta-learner over [anomaly_score, attack_proba, text_proba]."""

    def __init__(self, config: FusionConfig, *, seed: int = 42):
        self.config = config
        self.seed = seed
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        )
        self._fitted = False

    def fit(self, scores: np.ndarray, y: np.ndarray) -> FusionMetaLearner:
        self.pipeline.fit(np.asarray(scores, dtype=np.float64), np.asarray(y))
        self._fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Final threat probability per event."""
        if not self._fitted:
            raise RuntimeError("FusionMetaLearner used before fit/load")
        return self.pipeline.predict_proba(np.asarray(scores, dtype=np.float64))[:, 1]

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (self.predict_proba(scores) >= self.config.decision_threshold).astype(np.int64)

    @staticmethod
    def average_baseline(scores: np.ndarray) -> np.ndarray:
        """Min-max each column to [0, 1], then average — the naive alternative."""
        s = np.asarray(scores, dtype=np.float64)
        lo = s.min(axis=0, keepdims=True)
        rng = np.ptp(s, axis=0, keepdims=True)
        rng[rng == 0] = 1.0
        return ((s - lo) / rng).mean(axis=1)

    def save(self, path: Path) -> Path:
        if not self._fitted:
            raise RuntimeError("Refusing to save an unfitted FusionMetaLearner")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": ARTIFACT_VERSION,
                "config": self.config.model_dump(),
                "seed": self.seed,
                "features": list(FUSION_FEATURES),
                "pipeline": self.pipeline,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> FusionMetaLearner:
        payload = joblib.load(path)
        obj = cls(FusionConfig.model_validate(payload["config"]), seed=payload["seed"])
        obj.pipeline = payload["pipeline"]
        obj._fitted = True
        return obj
