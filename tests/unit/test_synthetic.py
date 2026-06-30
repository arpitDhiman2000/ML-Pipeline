"""Unit tests for the synthetic data generator.

These tests pin the generator's *contract*: schema fidelity, reproducibility,
the imbalance story, the injected data-quality issues, and drift behaviour.
They are what let later sprints trust the data they build on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threat_detection.config import TabularDataConfig, TextDataConfig
from threat_detection.data import schema, synthetic

pytestmark = pytest.mark.unit


# ----------------------------- tabular -------------------------------------
def test_tabular_has_canonical_schema(small_tabular_cfg: TabularDataConfig) -> None:
    df = synthetic.generate_tabular(small_tabular_cfg, seed=0)
    assert list(df.columns) == list(schema.tabular_columns())


def test_tabular_is_deterministic(small_tabular_cfg: TabularDataConfig) -> None:
    """Same (seed, params) -> byte-identical frame. The basis of dvc repro."""
    a = synthetic.generate_tabular(small_tabular_cfg, seed=123)
    b = synthetic.generate_tabular(small_tabular_cfg, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_tabular_seed_changes_output(small_tabular_cfg: TabularDataConfig) -> None:
    a = synthetic.generate_tabular(small_tabular_cfg, seed=1)
    b = synthetic.generate_tabular(small_tabular_cfg, seed=2)
    assert not a.equals(b)


def test_class_imbalance_is_respected(small_tabular_cfg: TabularDataConfig) -> None:
    df = synthetic.generate_tabular(small_tabular_cfg, seed=0)
    benign_share = (df[schema.LABEL_COLUMN] == schema.BENIGN_LABEL).mean()
    # Duplicates perturb the ratio slightly; assert it's in the right ballpark.
    assert abs(benign_share - small_tabular_cfg.benign_ratio) < 0.05


def test_all_attack_classes_present(small_tabular_cfg: TabularDataConfig) -> None:
    df = synthetic.generate_tabular(small_tabular_cfg, seed=0)
    present = set(df[schema.LABEL_COLUMN].unique())
    # Every configured attack class should appear at least once.
    assert set(small_tabular_cfg.attack_mix).issubset(present)


def test_inf_nan_injected_into_rate_columns(small_tabular_cfg: TabularDataConfig) -> None:
    df = synthetic.generate_tabular(small_tabular_cfg, seed=0)
    rate_block = df[list(schema.RATE_FEATURES)]
    assert np.isinf(rate_block.to_numpy(dtype=float)).any()
    assert rate_block.isna().to_numpy().any()


def test_non_rate_columns_have_no_inf(small_tabular_cfg: TabularDataConfig) -> None:
    df = synthetic.generate_tabular(small_tabular_cfg, seed=0)
    non_rate = [f for f in schema.FLOW_FEATURES if f not in schema.RATE_FEATURES]
    block = df[non_rate].to_numpy(dtype=float)
    assert not np.isinf(block).any()


def test_duplicates_present(small_tabular_cfg: TabularDataConfig) -> None:
    df = synthetic.generate_tabular(small_tabular_cfg, seed=0)
    assert df.duplicated().any()


def test_drift_shifts_attack_signal(small_tabular_cfg: TabularDataConfig) -> None:
    """Drift inflates attack signal features but leaves benign rows alone."""
    base = synthetic.generate_tabular(small_tabular_cfg, seed=5, drift_scale=1.0)
    drifted = synthetic.generate_tabular(small_tabular_cfg, seed=5, drift_scale=3.0)
    feat = "Total Fwd Packets"
    attack = lambda d: d.loc[d[schema.LABEL_COLUMN] != schema.BENIGN_LABEL, feat].mean()  # noqa: E731
    assert attack(drifted) > attack(base)


# ------------------------------- text --------------------------------------
def test_text_schema_and_labels(small_text_cfg: TextDataConfig) -> None:
    df = synthetic.generate_text(small_text_cfg, seed=0)
    assert list(df.columns) == [schema.TEXT_COLUMN, schema.TEXT_LABEL_COLUMN]
    assert set(df[schema.TEXT_LABEL_COLUMN].unique()).issubset({0, 1})


def test_text_is_deterministic(small_text_cfg: TextDataConfig) -> None:
    a = synthetic.generate_text(small_text_cfg, seed=42)
    b = synthetic.generate_text(small_text_cfg, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_text_malicious_ratio(small_text_cfg: TextDataConfig) -> None:
    df = synthetic.generate_text(small_text_cfg, seed=0)
    mal_share = df[schema.TEXT_LABEL_COLUMN].mean()
    assert abs(mal_share - small_text_cfg.malicious_ratio) < 0.02
