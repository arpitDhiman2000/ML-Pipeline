"""Unit tests for the tabular model wrappers (fit/predict/inference/persist)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threat_detection.config import IsolationForestConfig, PreprocessingConfig, XGBoostConfig
from threat_detection.data import schema
from threat_detection.features import target
from threat_detection.features.tabular import TabularPreprocessor
from threat_detection.models.anomaly import AnomalyDetector
from threat_detection.models.classifier import AttackClassifier, class_balanced_weights

pytestmark = pytest.mark.unit


@pytest.fixture
def xy(raw_flows: pd.DataFrame):
    pre = TabularPreprocessor(PreprocessingConfig())
    X = pre.fit_transform(raw_flows[list(schema.FLOW_FEATURES)])
    labels = raw_flows[schema.LABEL_COLUMN]
    return X, labels, target.encode_labels(labels), target.to_binary_target(labels)


# ----------------------------- anomaly -------------------------------------
def test_contamination_auto_ties_to_prevalence() -> None:
    cfg = IsolationForestConfig(contamination="auto", n_estimators=50)
    det = AnomalyDetector.from_prevalence(cfg, attack_prevalence=0.23, seed=0)
    assert det.contamination == pytest.approx(0.23)


def test_contamination_explicit_is_respected() -> None:
    cfg = IsolationForestConfig(contamination=0.1, n_estimators=50)
    det = AnomalyDetector.from_prevalence(cfg, attack_prevalence=0.4, seed=0)
    assert det.contamination == 0.1


def test_anomaly_fit_score_predict(xy) -> None:
    X, _, _, _ = xy
    det = AnomalyDetector(IsolationForestConfig(n_estimators=50), contamination=0.25, seed=0)
    det.fit(X)
    scores = det.anomaly_score(X)
    preds = det.predict(X)
    assert scores.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_anomaly_save_load_parity(xy, tmp_path) -> None:
    X, _, _, _ = xy
    det = AnomalyDetector(IsolationForestConfig(n_estimators=50), contamination=0.25, seed=0).fit(X)
    expected = det.anomaly_score(X)
    det.save(tmp_path / "anomaly.joblib")
    reloaded = AnomalyDetector.load(tmp_path / "anomaly.joblib")
    np.testing.assert_array_equal(expected, reloaded.anomaly_score(X))


# ----------------------------- classifier ----------------------------------
def test_class_balanced_weights_inverse_frequency() -> None:
    y = np.array([0, 0, 0, 1])  # class 0 common, class 1 rare
    w = class_balanced_weights(y)
    assert w[3] > w[0]  # rare class up-weighted


def _fast_xgb() -> XGBoostConfig:
    return XGBoostConfig(n_estimators=20, max_depth=3, early_stopping_rounds=5)


def test_classifier_fit_predict_shapes(xy) -> None:
    X, _, y_label, _ = xy
    clf = AttackClassifier(_fast_xgb(), seed=0).fit(X, y_label)
    proba = clf.predict_proba(X)
    assert proba.shape[0] == len(X)
    preds = clf.predict(X)
    assert preds.shape == (len(X),)


def test_attack_proba_in_unit_interval(xy) -> None:
    X, _, y_label, _ = xy
    clf = AttackClassifier(_fast_xgb(), seed=0).fit(X, y_label)
    ap = clf.attack_proba(X)
    assert ap.shape == (len(X),)
    assert (ap >= 0.0).all() and (ap <= 1.0).all()


def test_classifier_save_load_parity(xy, tmp_path) -> None:
    X, _, y_label, _ = xy
    clf = AttackClassifier(_fast_xgb(), seed=0).fit(X, y_label)
    expected = clf.predict_proba(X)
    path = tmp_path / "classifier.json"
    clf.save(path)
    reloaded = AttackClassifier.load(path)
    np.testing.assert_allclose(expected, reloaded.predict_proba(X), rtol=1e-5, atol=1e-6)
