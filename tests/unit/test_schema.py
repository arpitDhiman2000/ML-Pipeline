"""Unit tests for the canonical data schema (the cross-component contract)."""

from __future__ import annotations

import pytest

from threat_detection.data import schema

pytestmark = pytest.mark.unit


def test_exactly_78_flow_features() -> None:
    assert len(schema.FLOW_FEATURES) == 78
    assert len(set(schema.FLOW_FEATURES)) == 78  # no duplicates


def test_label_space_is_benign_plus_attacks() -> None:
    assert schema.ALL_LABELS[0] == schema.BENIGN_LABEL
    assert set(schema.ATTACK_CLASSES).issubset(set(schema.ALL_LABELS))
    assert schema.BENIGN_LABEL not in schema.ATTACK_CLASSES


def test_rate_and_signal_features_are_valid_columns() -> None:
    for feat in (*schema.RATE_FEATURES, *schema.SIGNAL_FEATURES):
        assert feat in schema.FLOW_FEATURES


def test_tabular_columns_end_with_label() -> None:
    cols = schema.tabular_columns()
    assert cols[-1] == schema.LABEL_COLUMN
    assert len(cols) == 79
