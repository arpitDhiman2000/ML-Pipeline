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


class IsolationForestConfig(BaseModel):
    """Unsupervised anomaly stage (the zero-day safety net)."""

    # "auto" -> tie contamination to the observed attack prevalence in train.
    contamination: float | Literal["auto"] = "auto"
    n_estimators: int = Field(default=200, gt=0)
    max_samples: float | Literal["auto"] = "auto"


class XGBoostConfig(BaseModel):
    """Supervised multi-class attack classifier."""

    max_depth: int = Field(default=6, gt=0)
    learning_rate: float = Field(default=0.1, gt=0.0)
    n_estimators: int = Field(default=200, gt=0)
    subsample: float = Field(default=0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.8, gt=0.0, le=1.0)
    min_child_weight: float = Field(default=1.0, ge=0.0)
    reg_lambda: float = Field(default=1.0, ge=0.0)
    reg_alpha: float = Field(default=0.0, ge=0.0)
    early_stopping_rounds: int = Field(default=20, gt=0)
    tune: bool = False  # enable Optuna search (off by default = fast)
    n_trials: int = Field(default=15, gt=0)
    operating_recall: float = Field(default=0.80, gt=0.0, lt=1.0)  # PR-curve target


class EvalGateConfig(BaseModel):
    """Promotion rule encoded as data (used by CI in Sprint 5, retrain in Sprint 6)."""

    min_pr_auc: float = Field(default=0.80, ge=0.0, le=1.0)
    min_per_class_recall: float = Field(default=0.50, ge=0.0, le=1.0)
    max_p99_latency_ms: float = Field(default=50.0, gt=0.0)  # enforced from Sprint 4


class LSTMConfig(BaseModel):
    """PyTorch LSTM text classifier (command/payload threat detection)."""

    embedding_dim: int = Field(default=64, gt=0)
    hidden_dim: int = Field(default=64, gt=0)
    num_layers: int = Field(default=1, gt=0)
    bidirectional: bool = True
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    lr: float = Field(default=1e-3, gt=0.0)
    batch_size: int = Field(default=256, gt=0)
    max_epochs: int = Field(default=8, gt=0)
    patience: int = Field(default=3, gt=0)  # early-stopping patience on val loss
    operating_recall: float = Field(default=0.80, gt=0.0, lt=1.0)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class AppConfig(BaseModel):
    """Top-level validated view over ``params.yaml``."""

    seed: int = 42
    data: DataConfig
    split: SplitConfig

    preprocessing: PreprocessingConfig = PreprocessingConfig()
    isolation_forest: IsolationForestConfig = IsolationForestConfig()
    xgboost: XGBoostConfig = XGBoostConfig()
    eval_gate: EvalGateConfig = EvalGateConfig()
    lstm: LSTMConfig = LSTMConfig()

    # Forward-compatible holders for parameters added in later sprints.
    fusion: dict = {}
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
