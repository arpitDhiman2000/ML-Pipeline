"""XGBoost multi-class attack classifier — the supervised stage.

Why XGBoost over a linear model or a neural net (interview point): flow features
are non-linear, mixed-scale, and interacting; gradient-boosted trees capture this
with strong native regularisation and handle imbalance via per-sample weights.
On tabular, medium-size data they match or beat NNs with far less tuning.

This wrapper trains on the canonical 8-class label space and exposes:
  * ``predict`` — the predicted class id,
  * ``predict_proba`` — full per-class probabilities,
  * ``attack_proba`` — 1 - P(BENIGN), the single "is this an attack?" score the
    binary metrics and the downstream fusion use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from threat_detection.config import XGBoostConfig
from threat_detection.features import target

ARTIFACT_VERSION = 1


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights inversely proportional to class frequency.

    The tree-equivalent of class weighting: it pushes the model toward rare
    attack classes without the memory blow-up / overfit of naive oversampling.
    """
    classes, counts = np.unique(y, return_counts=True)
    freq = {int(c): n for c, n in zip(classes, counts, strict=True)}
    total = len(y)
    n_classes = len(classes)
    # weight = total / (n_classes * count_of_its_class)
    return np.array([total / (n_classes * freq[int(c)]) for c in y], dtype=np.float32)


class AttackClassifier:
    """Persistable XGBoost multi-class classifier over the canonical labels."""

    def __init__(self, config: XGBoostConfig, *, seed: int):
        self.config = config
        self.seed = seed
        self.num_classes = target.NUM_CLASSES
        # NOTE: do NOT pass num_class to the sklearn wrapper — it infers the
        # class count from y. Passing both raises a "num_class mismatch" error.
        self.model = XGBClassifier(
            objective="multi:softprob",
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_weight=config.min_child_weight,
            reg_lambda=config.reg_lambda,
            reg_alpha=config.reg_alpha,
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
            early_stopping_rounds=config.early_stopping_rounds,
        )
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> AttackClassifier:
        X = np.asarray(X, dtype=np.float32)
        sample_weight = class_balanced_weights(y)
        if X_val is not None:
            eval_set = [(np.asarray(X_val, dtype=np.float32), y_val)]
        else:
            # No validation set => early stopping is impossible; disable it.
            eval_set = None
            self.model.set_params(early_stopping_rounds=None)
        self.model.fit(X, y, sample_weight=sample_weight, eval_set=eval_set, verbose=False)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("AttackClassifier used before fit/load")
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=np.float32)).astype(np.int64)

    def attack_proba(self, X: np.ndarray) -> np.ndarray:
        """P(attack) = 1 - P(BENIGN). The single binary threat score.

        Robust to the model having seen only a subset of classes: we locate the
        BENIGN column via ``classes_`` rather than assuming a fixed position.
        """
        proba = self.predict_proba(X)
        benign_id = target.LABEL_TO_INDEX["BENIGN"]
        classes = np.asarray(self.model.classes_)
        match = np.where(classes == benign_id)[0]
        if match.size == 0:
            return np.ones(len(proba), dtype=np.float64)  # benign never seen
        return 1.0 - proba[:, int(match[0])]

    def save(self, path: Path) -> Path:
        if not self._fitted:
            raise RuntimeError("Refusing to save an unfitted AttackClassifier")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Persist the booster in XGBoost's native JSON next to a sidecar of meta.
        self.model.save_model(str(path))
        joblib_meta = path.with_suffix(path.suffix + ".meta.joblib")
        import joblib

        joblib.dump(
            {"version": ARTIFACT_VERSION, "config": self.config.model_dump(), "seed": self.seed},
            joblib_meta,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> AttackClassifier:
        import joblib

        meta = joblib.load(path.with_suffix(path.suffix + ".meta.joblib"))
        obj = cls(XGBoostConfig.model_validate(meta["config"]), seed=meta["seed"])
        obj.model.load_model(str(path))
        obj._fitted = True
        return obj
