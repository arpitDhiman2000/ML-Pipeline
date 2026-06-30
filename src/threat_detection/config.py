"""Typed configuration loaded from ``params.yaml``.

Why typed config (interview point): ``params.yaml`` is the DVC-tracked source of
truth, but raw dicts are error-prone (typos, wrong types silently accepted).
We parse it once into Pydantic models so that an invalid parameter fails loudly
at startup rather than producing a subtly wrong model hours into training. The
same validated config object is used by training, serving, and tests, which is
half of how we guarantee train/serve consistency.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from threat_detection.paths import project_root


class DriftConfig(BaseModel):
    enabled: bool = False
    shift_scale: float = 1.0


class TabularDataConfig(BaseModel):
    n_rows: int = Field(gt=0)
    benign_ratio: float = Field(gt=0.0, lt=1.0)
    attack_mix: dict[str, float]
    inf_nan_fraction: float = Field(ge=0.0, le=1.0)
    duplicate_fraction: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _attack_mix_sums_to_one(self) -> TabularDataConfig:
        total = sum(self.attack_mix.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"data.tabular.attack_mix must sum to 1.0, got {total:.6f}")
        return self


class TextDataConfig(BaseModel):
    n_rows: int = Field(gt=0)
    malicious_ratio: float = Field(gt=0.0, lt=1.0)


class DataConfig(BaseModel):
    tabular: TabularDataConfig
    text: TextDataConfig
    drift: DriftConfig = DriftConfig()


class SplitConfig(BaseModel):
    test_size: float = Field(gt=0.0, lt=1.0)
    val_size: float = Field(ge=0.0, lt=1.0)
    stratify: bool = True


class TextTokenizerConfig(BaseModel):
    """Config for the command/payload tokenizer (used by the LSTM in Sprint 3)."""

    level: Literal["char", "word"] = "char"
    max_len: int = Field(default=256, gt=0)
    min_freq: int = Field(default=1, ge=1)
    max_vocab: int = Field(default=5000, gt=0)
    lowercase: bool = True


class PreprocessingConfig(BaseModel):
    """Tabular + text preprocessing parameters."""

    numeric_imputer_strategy: Literal["median", "mean", "most_frequent"] = "median"
    scaler: Literal["standard", "robust", "none"] = "standard"
    drop_duplicates: bool = True
    text: TextTokenizerConfig = TextTokenizerConfig()


class AppConfig(BaseModel):
    """Top-level validated view over ``params.yaml``."""

    seed: int = 42
    data: DataConfig
    split: SplitConfig

    # Forward-compatible holders for parameters added in later sprints.
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    isolation_forest: dict = {}
    xgboost: dict = {}
    lstm: dict = {}
    fusion: dict = {}
    eval_gate: dict = {}
    drift_monitor: dict = {}


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load and validate ``params.yaml`` into an :class:`AppConfig`.

    Pass an explicit path in tests to load a fixture params file.
    """
    cfg_path = Path(path) if path is not None else project_root() / "params.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Cached accessor for the default project config (for app/runtime use)."""
    return load_config()
