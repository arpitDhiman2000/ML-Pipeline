"""Centralised, environment-overridable filesystem paths.

Why this module exists (interview point): nothing in the codebase builds a path
with string concatenation. Every path flows from here, derived from a single
project root. CI and the cloud runtime can relocate the data root with one env
var (``THREAT_DETECTION_DATA_ROOT``) without changing any source code — this is
what keeps the pipeline portable across laptop, CI runner, and EC2.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml


def project_root() -> Path:
    """Return the repository root.

    Resolved by walking up from this file until a directory containing
    ``pyproject.toml`` is found. Falls back to three levels up.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[2]


@lru_cache(maxsize=1)
def _paths_config() -> dict:
    cfg_file = project_root() / "configs" / "paths.yaml"
    with cfg_file.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve(relative: str) -> Path:
    """Resolve a config-relative path against the (optionally overridden) root."""
    override = os.environ.get("THREAT_DETECTION_DATA_ROOT")
    base = Path(override) if override else project_root()
    return (base / relative).resolve()


class Paths:
    """Lazy accessor for all project paths defined in ``configs/paths.yaml``."""

    @staticmethod
    def root() -> Path:
        return project_root()

    @staticmethod
    def raw_zone() -> Path:
        return _resolve(_paths_config()["zones"]["raw"])

    @staticmethod
    def interim_zone() -> Path:
        return _resolve(_paths_config()["zones"]["interim"])

    @staticmethod
    def processed_zone() -> Path:
        return _resolve(_paths_config()["zones"]["processed"])

    @staticmethod
    def external_zone() -> Path:
        return _resolve(_paths_config()["zones"]["external"])

    @staticmethod
    def tabular_raw() -> Path:
        return _resolve(_paths_config()["datasets"]["tabular_raw"])

    @staticmethod
    def text_raw() -> Path:
        return _resolve(_paths_config()["datasets"]["text_raw"])

    @staticmethod
    def artifacts_root() -> Path:
        return _resolve(_paths_config()["artifacts"]["root"])

    @staticmethod
    def reports_dir() -> Path:
        return _resolve(_paths_config()["artifacts"]["reports"])

    @staticmethod
    def preprocessor_artifact() -> Path:
        return _resolve(_paths_config()["artifacts"]["preprocessor"])

    @staticmethod
    def tokenizer_artifact() -> Path:
        return _resolve(_paths_config()["artifacts"]["tokenizer"])

    @staticmethod
    def processed_file(name: str) -> Path:
        """Path to a processed dataset, e.g. ``processed_file('tabular_train')``."""
        return Paths.processed_zone() / f"{name}.parquet"

    @staticmethod
    def ensure(path: Path) -> Path:
        """Create the directory (or the file's parent) if missing; return it."""
        target = path if path.suffix == "" else path.parent
        target.mkdir(parents=True, exist_ok=True)
        return path
