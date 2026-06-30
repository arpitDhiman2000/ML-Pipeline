"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from threat_detection.config import (
    AppConfig,
    DataConfig,
    DriftConfig,
    SplitConfig,
    TabularDataConfig,
    TextDataConfig,
)


@pytest.fixture
def small_tabular_cfg() -> TabularDataConfig:
    """A tiny tabular config for fast, deterministic tests."""
    return TabularDataConfig(
        n_rows=2000,
        benign_ratio=0.75,
        attack_mix={
            "DoS": 0.35,
            "DDoS": 0.25,
            "PortScan": 0.20,
            "Brute Force": 0.10,
            "Web Attack": 0.06,
            "Botnet": 0.03,
            "Infiltration": 0.01,
        },
        inf_nan_fraction=0.02,
        duplicate_fraction=0.05,
    )


@pytest.fixture
def small_text_cfg() -> TextDataConfig:
    return TextDataConfig(n_rows=1000, malicious_ratio=0.30)


@pytest.fixture
def small_app_cfg(
    small_tabular_cfg: TabularDataConfig, small_text_cfg: TextDataConfig
) -> AppConfig:
    return AppConfig(
        seed=7,
        data=DataConfig(tabular=small_tabular_cfg, text=small_text_cfg, drift=DriftConfig()),
        split=SplitConfig(test_size=0.2, val_size=0.1, stratify=True),
    )
