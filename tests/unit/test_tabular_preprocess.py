"""Unit tests for the fitted tabular preprocessor.

These pin the production guarantees this artifact exists to provide:
  * deterministic, finite output (no NaN/Inf leaks into the model),
  * train/serve parity (a reloaded artifact transforms identically),
  * robustness to column reordering / omission at inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threat_detection.config import PreprocessingConfig
from threat_detection.data import schema
from threat_detection.features.tabular import InfinityToNaN, TabularPreprocessor

pytestmark = pytest.mark.unit


@pytest.fixture
def features(raw_flows: pd.DataFrame) -> pd.DataFrame:
    return raw_flows[list(schema.FLOW_FEATURES)]


def test_infinity_to_nan_transformer() -> None:
    arr = np.array([[np.inf, 1.0], [-np.inf, 2.0]])
    out = InfinityToNaN().fit_transform(arr)
    assert np.isnan(out[0, 0]) and np.isnan(out[1, 0])
    assert out[0, 1] == 1.0


def test_output_shape_and_dtype(features: pd.DataFrame) -> None:
    pre = TabularPreprocessor(PreprocessingConfig())
    X = pre.fit_transform(features)
    assert X.shape == (len(features), 78)
    assert X.dtype == np.float32


def test_no_nan_or_inf_in_output(features: pd.DataFrame) -> None:
    """Input has injected Inf/NaN; output must be fully finite."""
    assert not np.isfinite(features.to_numpy(dtype=float)).all()  # dirty input
    pre = TabularPreprocessor(PreprocessingConfig())
    X = pre.fit_transform(features)
    assert np.isfinite(X).all()


def test_transform_before_fit_raises(features: pd.DataFrame) -> None:
    pre = TabularPreprocessor(PreprocessingConfig())
    with pytest.raises(RuntimeError, match="before fit"):
        pre.transform(features)


def test_standardization_centers_features(features: pd.DataFrame) -> None:
    pre = TabularPreprocessor(PreprocessingConfig())
    X = pre.fit_transform(features)
    # StandardScaler => per-column mean ~0, std ~1 on the training data.
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-2)


def test_save_load_parity(tmp_path, features: pd.DataFrame) -> None:
    """The reloaded artifact must transform IDENTICALLY — the core parity test."""
    pre = TabularPreprocessor(PreprocessingConfig())
    pre.fit(features)
    expected = pre.transform(features)

    path = tmp_path / "preprocessor.joblib"
    pre.save(path)
    reloaded = TabularPreprocessor.load(path)
    actual = reloaded.transform(features)

    np.testing.assert_array_equal(expected, actual)


def test_column_reorder_robustness(features: pd.DataFrame) -> None:
    """A live event with shuffled columns is coerced to schema order."""
    pre = TabularPreprocessor(PreprocessingConfig())
    expected = pre.fit_transform(features)

    shuffled = features.sample(frac=1.0, axis=1, random_state=1)  # reorder columns
    actual = pre.transform(shuffled)
    np.testing.assert_array_equal(expected, actual)


def test_missing_column_is_imputed(features: pd.DataFrame) -> None:
    """A dropped column at inference becomes NaN -> imputed, not a crash."""
    pre = TabularPreprocessor(PreprocessingConfig())
    pre.fit(features)
    dropped = features.drop(columns=["Flow Duration"])
    X = pre.transform(dropped)
    assert X.shape == (len(features), 78)
    assert np.isfinite(X).all()


def test_save_refuses_unfitted(tmp_path) -> None:
    pre = TabularPreprocessor(PreprocessingConfig())
    with pytest.raises(RuntimeError, match="unfitted"):
        pre.save(tmp_path / "x.joblib")
