"""Isolation Forest anomaly stage — the unsupervised zero-day safety net.

Why it's here (interview point): a supervised classifier only knows attack types
present in its training labels. The Isolation Forest needs no labels, so it flags
anomalous flows that match no known attack — the safety net for novel/zero-day
attacks. It isolates outliers by random partitioning (anomalies need fewer splits
=> shorter path length => higher anomaly score), is linear-time, and scales to
millions of flows where One-Class SVM is O(n^2).

Key knob: ``contamination`` is *tied to the observed attack prevalence*, not
guessed — that is the defensible answer to "why that value?".
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from threat_detection.config import IsolationForestConfig

ARTIFACT_VERSION = 1


class AnomalyDetector:
    """Thin, persistable wrapper around sklearn's IsolationForest."""

    def __init__(self, config: IsolationForestConfig, *, contamination: float | str, seed: int):
        self.config = config
        self.contamination = contamination
        self.seed = seed
        self.model = IsolationForest(
            n_estimators=config.n_estimators,
            max_samples=config.max_samples,
            contamination=contamination,
            random_state=seed,
            n_jobs=-1,
        )
        self._fitted = False

    @classmethod
    def from_prevalence(
        cls, config: IsolationForestConfig, *, attack_prevalence: float, seed: int
    ) -> AnomalyDetector:
        """Build a detector with contamination resolved from config.

        If ``config.contamination == 'auto'`` we tie it to the observed attack
        prevalence (clamped to sklearn's valid (0, 0.5] range); otherwise we use
        the explicit float from config.
        """
        if config.contamination == "auto":
            contamination: float | str = float(min(max(attack_prevalence, 1e-3), 0.5))
        else:
            contamination = config.contamination
        return cls(config, contamination=contamination, seed=seed)

    def fit(self, X: np.ndarray) -> AnomalyDetector:
        self.model.fit(np.asarray(X, dtype=np.float32))
        self._fitted = True
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Higher == more anomalous (we negate sklearn's decision_function)."""
        if not self._fitted:
            raise RuntimeError("AnomalyDetector used before fit/load")
        return -self.model.decision_function(np.asarray(X, dtype=np.float32))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 == anomaly, 0 == normal (sklearn returns -1/1)."""
        raw = self.model.predict(np.asarray(X, dtype=np.float32))
        return (raw == -1).astype(np.int64)

    def save(self, path: Path) -> Path:
        if not self._fitted:
            raise RuntimeError("Refusing to save an unfitted AnomalyDetector")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": ARTIFACT_VERSION,
                "config": self.config.model_dump(),
                "contamination": self.contamination,
                "seed": self.seed,
                "model": self.model,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> AnomalyDetector:
        payload = joblib.load(path)
        obj = cls(
            IsolationForestConfig.model_validate(payload["config"]),
            contamination=payload["contamination"],
            seed=payload["seed"],
        )
        obj.model = payload["model"]
        obj._fitted = True
        return obj
