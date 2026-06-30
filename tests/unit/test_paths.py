"""Unit tests for path resolution and env-var override."""

from __future__ import annotations

import importlib

import pytest

from threat_detection import paths as paths_mod

pytestmark = pytest.mark.unit


def test_project_root_contains_pyproject() -> None:
    assert (paths_mod.project_root() / "pyproject.toml").exists()


def test_data_root_override(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """THREAT_DETECTION_DATA_ROOT relocates the data zones (CI/cloud portability)."""
    monkeypatch.setenv("THREAT_DETECTION_DATA_ROOT", str(tmp_path))
    importlib.reload(paths_mod)
    raw = paths_mod.Paths.raw_zone()
    assert str(tmp_path) in str(raw)


def test_ensure_creates_parent(tmp_path) -> None:
    target = tmp_path / "nested" / "dir" / "file.parquet"
    returned = paths_mod.Paths.ensure(target)
    assert returned == target
    assert target.parent.is_dir()
