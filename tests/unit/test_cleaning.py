"""Unit tests for row-level cleaning (train vs serve behaviour)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threat_detection.data import schema
from threat_detection.features.cleaning import clean_flows, replace_inf_with_nan

pytestmark = pytest.mark.unit


def test_replace_inf_with_nan_targets_features_only() -> None:
    df = pd.DataFrame(
        {
            "Flow Bytes/s": [np.inf, 1.0, -np.inf],
            "Flow Duration": [1.0, 2.0, 3.0],
        }
    )
    out = replace_inf_with_nan(df, columns=["Flow Bytes/s", "Flow Duration"])
    assert out["Flow Bytes/s"].isna().sum() == 2
    assert not np.isinf(out.to_numpy(dtype=float)).any()
    # input not mutated
    assert np.isinf(df["Flow Bytes/s"]).sum() == 2


def test_training_mode_drops_duplicates(raw_flows: pd.DataFrame) -> None:
    assert raw_flows.duplicated().any()  # generator injected dupes
    cleaned = clean_flows(raw_flows, drop_duplicates=True, training=True)
    assert not cleaned.duplicated().any()
    assert len(cleaned) < len(raw_flows)


def test_serve_mode_preserves_every_row(raw_flows: pd.DataFrame) -> None:
    """At inference we must score every event — no dropping rows."""
    served = clean_flows(raw_flows, drop_duplicates=True, training=False)
    assert len(served) == len(raw_flows)


def test_training_mode_removes_inf(raw_flows: pd.DataFrame) -> None:
    cleaned = clean_flows(raw_flows, drop_duplicates=True, training=True)
    block = cleaned[list(schema.RATE_FEATURES)].to_numpy(dtype=float)
    assert not np.isinf(block).any()  # Inf converted to NaN
