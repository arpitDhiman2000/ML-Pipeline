"""Integration test: train the tabular path end to end on temp data.

Generates -> preprocesses -> trains, logging to a LOCAL MLflow file store (no
network), and asserts the models are saved and metrics are sane. This is the
"model inference" coverage from the resume bullet, exercised through the real
training entrypoint.
"""

from __future__ import annotations

import importlib

import pytest

from threat_detection.config import AppConfig

pytestmark = pytest.mark.integration


@pytest.fixture
def trainable_cfg(small_app_cfg: AppConfig) -> AppConfig:
    cfg = small_app_cfg.model_copy(deep=True)
    cfg.data.tabular.n_rows = 5000
    cfg.xgboost.n_estimators = 30
    cfg.xgboost.early_stopping_rounds = 5
    cfg.isolation_forest.n_estimators = 50
    return cfg


def test_train_tabular_end_to_end(
    trainable_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("THREAT_DETECTION_DATA_ROOT", str(tmp_path))
    # Force a local MLflow file store so the test never touches DagsHub.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")

    from threat_detection import paths as paths_mod

    importlib.reload(paths_mod)
    import threat_detection.data.generate as gen_mod
    import threat_detection.features.build as build_mod
    import threat_detection.training.train_tabular as train_mod

    importlib.reload(gen_mod)
    importlib.reload(build_mod)
    importlib.reload(train_mod)

    gen_mod.generate_raw_data(trainable_cfg, force=True)
    build_mod.run_preprocessing(trainable_cfg)

    metrics = train_mod.train(trainable_cfg, register=False)

    # models persisted for serving
    assert paths_mod.Paths.anomaly_model().exists()
    assert paths_mod.Paths.classifier_model().exists()

    # metrics are in valid ranges
    flat = metrics.to_flat_dict()
    assert 0.0 <= flat["pr_auc"] <= 1.0
    assert 0.0 <= flat["recall_at_threshold"] <= 1.0
    assert "min_per_class_recall" in flat

    # the synthetic signal is strong, so PR-AUC should be clearly better than chance
    assert flat["pr_auc"] > 0.5
