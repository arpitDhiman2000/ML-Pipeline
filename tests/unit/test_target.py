"""Unit tests for target derivation and the stable label mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threat_detection.data import schema
from threat_detection.features import target

pytestmark = pytest.mark.unit


def test_benign_is_class_zero() -> None:
    assert target.LABEL_TO_INDEX[schema.BENIGN_LABEL] == 0


def test_mapping_is_stable_and_total() -> None:
    assert len(schema.ALL_LABELS) == target.NUM_CLASSES
    # round-trip every class id
    for label, idx in target.LABEL_TO_INDEX.items():
        assert target.INDEX_TO_LABEL[idx] == label


def test_encode_decode_roundtrip() -> None:
    labels = pd.Series(["BENIGN", "DoS", "PortScan", "Infiltration"])
    enc = target.encode_labels(labels)
    dec = target.decode_labels(enc)
    assert list(dec) == list(labels)


def test_binary_target() -> None:
    labels = pd.Series(["BENIGN", "DoS", "BENIGN", "DDoS"])
    binary = target.to_binary_target(labels)
    assert binary.tolist() == [0, 1, 0, 1]
    assert binary.dtype == np.int64


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="Unknown labels"):
        target.encode_labels(pd.Series(["BENIGN", "NotARealClass"]))
