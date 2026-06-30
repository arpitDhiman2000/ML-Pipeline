"""Unit tests for MLflow tracking configuration (env-driven, no network)."""

from __future__ import annotations

import pytest

from threat_detection import tracking

pytestmark = pytest.mark.unit


def test_defaults_to_local_file_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(tracking, "load_env", lambda: None)  # ignore any local .env
    assert tracking.get_tracking_uri() == tracking.LOCAL_TRACKING_URI


def test_reads_uri_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://dagshub.com/u/r.mlflow")
    monkeypatch.setattr(tracking, "load_env", lambda: None)
    assert tracking.get_tracking_uri() == "https://dagshub.com/u/r.mlflow"


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("https://dagshub.com/u/r.mlflow", True),
        ("http://127.0.0.1:5000", True),
        ("file:./mlruns", False),
        ("./mlruns", False),
    ],
)
def test_is_remote(uri: str, expected: bool) -> None:
    assert tracking.is_remote(uri) is expected
