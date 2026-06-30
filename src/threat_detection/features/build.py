"""Preprocessing orchestration — the DVC `preprocess` stage.

Pipeline (leak-free by construction):
  1. Load raw flows/text via the canonical loaders.
  2. Clean (Inf->NaN, drop duplicate flows) on the FULL set, BEFORE splitting,
     so identical rows can't straddle train/test (a subtle leak).
  3. Stratified train/val/test split.
  4. Fit the preprocessor / tokenizer on the TRAIN split ONLY, then transform
     all splits. Fitting on train only is the whole leak-free story: the test
     set never influences the median/scale/vocab.
  5. Persist the fitted artifacts and the transformed splits.

The fitted artifacts (preprocessor.joblib, tokenizer.json) are the objects that
serving will reload — same bytes, no second code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from threat_detection.config import AppConfig, get_config
from threat_detection.data import schema
from threat_detection.data.loaders import load_tabular, load_text
from threat_detection.features import target
from threat_detection.features.cleaning import clean_flows
from threat_detection.features.tabular import TabularPreprocessor
from threat_detection.features.text import TextTokenizer
from threat_detection.logging_utils import get_logger
from threat_detection.paths import Paths

log = get_logger(__name__)


def _three_way_split(
    df: pd.DataFrame, stratify_on: np.ndarray, cfg: AppConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split honouring split config proportions."""
    seed = cfg.seed
    strat = stratify_on if cfg.split.stratify else None

    train_val, test = train_test_split(
        df, test_size=cfg.split.test_size, random_state=seed, stratify=strat
    )
    if cfg.split.val_size <= 0:
        return train_val, train_val.iloc[0:0], test

    # val_size is expressed as a fraction of the FULL set; convert to a fraction
    # of the remaining train_val partition.
    rel_val = cfg.split.val_size / (1.0 - cfg.split.test_size)
    strat_tv = train_val["__strat__"] if cfg.split.stratify else None
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=seed, stratify=strat_tv
    )
    return train, val, test


def _process_tabular(cfg: AppConfig) -> dict[str, int]:
    raw = load_tabular(Paths.tabular_raw())
    cleaned = clean_flows(raw, drop_duplicates=cfg.preprocessing.drop_duplicates, training=True)

    # Carry a stratification key alongside the frame through the split.
    cleaned = cleaned.copy()
    cleaned["__strat__"] = cleaned[schema.LABEL_COLUMN]
    train, val, test = _three_way_split(cleaned, cleaned["__strat__"].to_numpy(), cfg)

    pre = TabularPreprocessor(cfg.preprocessing)
    feature_cols = list(schema.FLOW_FEATURES)
    pre.fit(train[feature_cols])  # FIT ON TRAIN ONLY
    pre.save(Paths.preprocessor_artifact())

    counts: dict[str, int] = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part.empty:
            continue
        X = pre.transform(part[feature_cols])
        out = pd.DataFrame(X, columns=feature_cols)
        out[schema.LABEL_COLUMN] = part[schema.LABEL_COLUMN].to_numpy()
        out[schema.BINARY_TARGET_COLUMN] = target.to_binary_target(part[schema.LABEL_COLUMN])
        path = Paths.processed_file(f"tabular_{name}")
        Paths.ensure(path)
        out.to_parquet(path, index=False)
        counts[name] = len(out)
        log.info("tabular.processed", split=name, rows=len(out), path=str(path))

    # Sanity: transformed features must be finite (no NaN/Inf leaks downstream).
    sample = pre.transform(train[feature_cols][:1000])
    assert np.isfinite(sample).all(), "Preprocessed features contain NaN/Inf"
    return counts


def _process_text(cfg: AppConfig) -> dict[str, int]:
    raw = load_text(Paths.text_raw())
    strat = raw[schema.TEXT_LABEL_COLUMN].to_numpy()
    raw = raw.copy()
    raw["__strat__"] = strat
    train, val, test = _three_way_split(raw, strat, cfg)

    tok = TextTokenizer(cfg.preprocessing.text)
    tok.fit(train[schema.TEXT_COLUMN].astype(str).tolist())  # FIT ON TRAIN ONLY
    tok.save(Paths.tokenizer_artifact())
    log.info("text.tokenizer", vocab_size=tok.vocab_size)

    counts: dict[str, int] = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part.empty:
            continue
        out = part[[schema.TEXT_COLUMN, schema.TEXT_LABEL_COLUMN]].reset_index(drop=True)
        path = Paths.processed_file(f"text_{name}")
        Paths.ensure(path)
        out.to_parquet(path, index=False)
        counts[name] = len(out)
        log.info("text.processed", split=name, rows=len(out), path=str(path))
    return counts


def run_preprocessing(config: AppConfig | None = None) -> dict[str, dict[str, int]]:
    """Run the full preprocessing stage; returns per-modality split row counts."""
    cfg = config or get_config()
    log.info("preprocess.start")
    result = {"tabular": _process_tabular(cfg), "text": _process_text(cfg)}
    log.info("preprocess.done", **{f"{k}_splits": len(v) for k, v in result.items()})
    return result
