"""Fitted tabular preprocessing artifact (the train/serve parity guarantee).

The single most important production property here: the *exact same* fitted
object that scaled the training data is serialized, versioned alongside the
model, and reloaded at inference. There is no second code path that "re-derives"
the scaling — that is how train/serve skew (the #1 silent production bug) is
eliminated. ``TabularPreprocessor.transform`` also reindexes incoming columns to
the canonical schema order, so a live event with columns in a different order
(or with extras) is handled identically to training.

Pipeline: Inf->NaN  ->  median impute (fit on TRAIN only)  ->  standardize.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from threat_detection.config import PreprocessingConfig
from threat_detection.data import schema

ARTIFACT_VERSION = 1


class InfinityToNaN(BaseEstimator, TransformerMixin):
    """Replace +/-Inf with NaN so the imputer can fill them.

    A stateless, picklable transformer (must be importable at load time). Placed
    first in the pipeline so imputation sees NaN, never Inf.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float64)
        return np.where(np.isinf(arr), np.nan, arr)


def _make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    if kind == "none":
        return "passthrough"
    raise ValueError(f"Unknown scaler: {kind}")


class TabularPreprocessor:
    """Fit-once / reuse-everywhere preprocessing for the flow features."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.feature_names: list[str] = list(schema.FLOW_FEATURES)
        steps = [
            ("inf", InfinityToNaN()),
            ("impute", SimpleImputer(strategy=config.numeric_imputer_strategy)),
        ]
        scaler = _make_scaler(config.scaler)
        if scaler != "passthrough":
            steps.append(("scale", scaler))
        self.pipeline: Pipeline = Pipeline(steps)
        self._fitted = False

    # -- column handling -----------------------------------------------------
    def _select(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex to canonical feature order; missing cols become NaN.

        This is what makes serving robust to column reordering / omissions — the
        live request is coerced to exactly the columns the model was trained on.
        """
        return df.reindex(columns=self.feature_names)

    # -- fit / transform -----------------------------------------------------
    def fit(self, df: pd.DataFrame) -> TabularPreprocessor:
        self.pipeline.fit(self._select(df))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TabularPreprocessor.transform called before fit/load")
        out = self.pipeline.transform(self._select(df))
        return np.asarray(out, dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def n_features_out(self) -> int:
        return len(self.feature_names)

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path) -> Path:
        """Serialize the fitted artifact (pipeline + schema + config)."""
        if not self._fitted:
            raise RuntimeError("Refusing to save an unfitted preprocessor")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": ARTIFACT_VERSION,
                "config": self.config.model_dump(),
                "feature_names": self.feature_names,
                "pipeline": self.pipeline,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> TabularPreprocessor:
        payload = joblib.load(path)
        if payload.get("version") != ARTIFACT_VERSION:
            raise ValueError(
                f"Artifact version mismatch: file={payload.get('version')} "
                f"expected={ARTIFACT_VERSION}"
            )
        obj = cls(PreprocessingConfig.model_validate(payload["config"]))
        obj.feature_names = payload["feature_names"]
        obj.pipeline = payload["pipeline"]
        obj._fitted = True
        return obj
