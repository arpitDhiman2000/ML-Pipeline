"""Unit tests for typed configuration loading and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from threat_detection.config import AppConfig, load_config

pytestmark = pytest.mark.unit


def test_project_params_yaml_loads_and_validates() -> None:
    """The real params.yaml must parse into a valid AppConfig."""
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.data.tabular.n_rows > 0
    assert 0.0 < cfg.data.tabular.benign_ratio < 1.0


def test_attack_mix_must_sum_to_one(tmp_path: Path) -> None:
    """A misconfigured attack_mix fails loudly at load time, not at train time."""
    bad = tmp_path / "params.yaml"
    bad.write_text(
        textwrap.dedent(
            """
            seed: 1
            data:
              tabular:
                n_rows: 10
                benign_ratio: 0.75
                attack_mix: {DoS: 0.5, DDoS: 0.2}   # sums to 0.7, invalid
                inf_nan_fraction: 0.0
                duplicate_fraction: 0.0
              text:
                n_rows: 10
                malicious_ratio: 0.3
            split: {test_size: 0.2, val_size: 0.1, stratify: true}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match=r"attack_mix must sum to 1\.0"):
        load_config(bad)


def test_invalid_ratio_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "params.yaml"
    bad.write_text(
        textwrap.dedent(
            """
            seed: 1
            data:
              tabular:
                n_rows: 10
                benign_ratio: 1.5
                attack_mix: {DoS: 1.0}
                inf_nan_fraction: 0.0
                duplicate_fraction: 0.0
              text:
                n_rows: 10
                malicious_ratio: 0.3
            split: {test_size: 0.2, val_size: 0.1, stratify: true}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(bad)
