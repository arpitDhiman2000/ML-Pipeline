"""Unit tests for the fusion meta-learner."""

from __future__ import annotations

import numpy as np
import pytest

from threat_detection.config import FusionConfig
from threat_detection.evaluation.metrics import pr_auc
from threat_detection.models.fusion import FusionMetaLearner

pytestmark = pytest.mark.unit


def _scores(n: int = 600, seed: int = 0):
    """anomaly (huge scale, informative), attack (informative), text (pure noise)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    anomaly = (y * 8 + rng.normal(scale=3, size=n)) * 1000.0
    attack = np.clip(y * 0.6 + rng.normal(scale=0.25, size=n), 0, 1)
    text = rng.uniform(0, 1, size=n)  # uninformative
    return np.column_stack([anomaly, attack, text]), y


def test_fit_predict_shapes() -> None:
    X, y = _scores()
    fusion = FusionMetaLearner(FusionConfig(), seed=0).fit(X, y)
    proba = fusion.predict_proba(X)
    assert proba.shape == (len(X),)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_average_baseline_in_unit_interval() -> None:
    X, _ = _scores()
    avg = FusionMetaLearner.average_baseline(X)
    assert (avg >= 0).all() and (avg <= 1).all()


def test_meta_learner_beats_averaging() -> None:
    """Learned weighting should ignore the noisy column that averaging includes."""
    X, y = _scores(seed=1)
    split = len(X) // 2
    fusion = FusionMetaLearner(FusionConfig(), seed=0).fit(X[:split], y[:split])
    meta = pr_auc(y[split:], fusion.predict_proba(X[split:]))
    avg = pr_auc(y[split:], FusionMetaLearner.average_baseline(X[split:]))
    assert meta >= avg


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        FusionMetaLearner(FusionConfig()).predict_proba(np.zeros((1, 3)))


def test_save_load_parity(tmp_path) -> None:
    X, y = _scores()
    fusion = FusionMetaLearner(FusionConfig(), seed=0).fit(X, y)
    expected = fusion.predict_proba(X)
    path = tmp_path / "fusion.joblib"
    fusion.save(path)
    reloaded = FusionMetaLearner.load(path)
    np.testing.assert_allclose(expected, reloaded.predict_proba(X), rtol=1e-6)
