"""Unit tests for registry promotion helpers (mocked client, no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from threat_detection import registry

pytestmark = pytest.mark.unit


class _FakeClient:
    def __init__(self, versions: dict[str, list[int]]):
        self._versions = versions
        self.alias_calls: list[tuple[str, str, str]] = []

    def search_model_versions(self, query: str):
        # query looks like: name='<model>'
        name = query.split("'")[1]
        return [SimpleNamespace(version=str(v)) for v in self._versions.get(name, [])]

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.alias_calls.append((name, alias, version))


def test_latest_version_picks_max() -> None:
    client = _FakeClient({"m": [1, 3, 2]})
    assert registry.latest_version(client, "m") == "3"


def test_latest_version_none_when_unregistered() -> None:
    client = _FakeClient({})
    assert registry.latest_version(client, "missing") is None


def test_production_uri_format() -> None:
    assert registry.production_uri("m") == "models:/m@production"


def test_promote_sets_alias_on_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient({"threat-detection-tabular": [1, 2]})
    monkeypatch.setattr(registry, "get_client", lambda: client)

    promoted = registry.promote("threat-detection-tabular")

    assert promoted == "2"
    assert client.alias_calls == [("threat-detection-tabular", "production", "2")]


def test_promote_specific_version(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient({"m": [1, 2, 3]})
    monkeypatch.setattr(registry, "get_client", lambda: client)
    registry.promote("m", version="1", alias="staging")
    assert client.alias_calls == [("m", "staging", "1")]


def test_promote_raises_when_no_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeClient({})
    monkeypatch.setattr(registry, "get_client", lambda: client)
    with pytest.raises(ValueError, match="No registered versions"):
        registry.promote("nope")
