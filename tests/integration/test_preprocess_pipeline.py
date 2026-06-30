"""Integration test: full preprocessing stage end to end.

Generates raw data into a temp data root, runs the real preprocessing stage,
and asserts the production guarantees hold across the actual disk artifacts:
fitted artifacts persist, processed splits are leak-free (finite), and the
reloaded preprocessor (the serving path) reuses the exact training-time fit.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from threat_detection.config import AppConfig
from threat_detection.data import schema

pytestmark = pytest.mark.integration


@pytest.fixture
def big_cfg(small_app_cfg: AppConfig) -> AppConfig:
    # Larger n_rows so stratified splitting of rare classes is stable.
    cfg = small_app_cfg.model_copy(deep=True)
    cfg.data.tabular.n_rows = 6000
    cfg.data.text.n_rows = 3000
    return cfg


def test_full_preprocess_pipeline(
    big_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("THREAT_DETECTION_DATA_ROOT", str(tmp_path))

    from threat_detection import paths as paths_mod

    importlib.reload(paths_mod)
    import threat_detection.data.generate as gen_mod
    import threat_detection.features.build as build_mod

    importlib.reload(gen_mod)
    importlib.reload(build_mod)

    gen_mod.generate_raw_data(big_cfg, force=True)
    result = build_mod.run_preprocessing(big_cfg)

    # --- all splits produced, with a roughly correct test proportion ---
    assert result["tabular"]["train"] > result["tabular"]["test"] > 0
    assert result["text"]["train"] > 0

    # --- fitted artifacts persisted ---
    assert paths_mod.Paths.preprocessor_artifact().exists()
    assert paths_mod.Paths.tokenizer_artifact().exists()

    # --- processed tabular is leak-free (fully finite) and well-formed ---
    proc = pd.read_parquet(paths_mod.Paths.processed_file("tabular_test"))
    feat = proc[list(schema.FLOW_FEATURES)].to_numpy(dtype=float)
    assert np.isfinite(feat).all()
    assert schema.BINARY_TARGET_COLUMN in proc.columns
    assert set(proc[schema.BINARY_TARGET_COLUMN].unique()).issubset({0, 1})

    # --- serving path parity: reloaded artifact transforms finite & shaped ---
    from threat_detection.data.loaders import load_tabular
    from threat_detection.features.cleaning import clean_flows
    from threat_detection.features.tabular import TabularPreprocessor

    pre = TabularPreprocessor.load(paths_mod.Paths.preprocessor_artifact())
    raw = load_tabular(paths_mod.Paths.tabular_raw())
    served = clean_flows(raw.head(100), drop_duplicates=True, training=False)
    X = pre.transform(served[list(schema.FLOW_FEATURES)])
    assert X.shape == (100, 78)
    assert np.isfinite(X).all()

    # --- tokenizer reloads and encodes to fixed length ---
    from threat_detection.features.text import TextTokenizer

    tok = TextTokenizer.load(paths_mod.Paths.tokenizer_artifact())
    encoded = tok.encode("'; DROP TABLE users; --")
    assert len(encoded) == big_cfg.preprocessing.text.max_len
