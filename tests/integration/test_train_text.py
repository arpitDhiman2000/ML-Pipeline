"""Integration test: train the LSTM text path end to end on temp data."""

from __future__ import annotations

import importlib

import pytest

from threat_detection.config import AppConfig

pytestmark = pytest.mark.integration


@pytest.fixture
def trainable_cfg(small_app_cfg: AppConfig) -> AppConfig:
    cfg = small_app_cfg.model_copy(deep=True)
    cfg.data.text.n_rows = 3000
    cfg.lstm.embedding_dim = 16
    cfg.lstm.hidden_dim = 16
    cfg.lstm.max_epochs = 4
    cfg.lstm.device = "cpu"
    return cfg


def test_train_text_end_to_end(
    trainable_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("THREAT_DETECTION_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")

    from threat_detection import paths as paths_mod

    importlib.reload(paths_mod)
    import threat_detection.data.generate as gen_mod
    import threat_detection.features.build as build_mod
    import threat_detection.training.train_text as train_mod

    importlib.reload(gen_mod)
    importlib.reload(build_mod)
    importlib.reload(train_mod)

    gen_mod.generate_raw_data(trainable_cfg, force=True)
    build_mod.run_preprocessing(trainable_cfg)

    metrics = train_mod.train(trainable_cfg, register=False)

    # artifact saved for serving
    assert paths_mod.Paths.text_model().exists()
    assert paths_mod.Paths.text_model().with_suffix(".pt.meta.joblib").exists()

    # metrics valid, and the LSTM should be a competent classifier on this signal
    assert 0.0 <= metrics["lstm_f1"] <= 1.0
    assert 0.0 <= metrics["baseline_f1"] <= 1.0
    assert metrics["lstm_pr_auc"] > 0.6
