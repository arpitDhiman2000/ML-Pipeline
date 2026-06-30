"""Row-level cleaning of raw flow data.

This is deliberately separate from the fitted sklearn transform (``tabular.py``):
cleaning changes *rows* (dropping duplicates) and so must behave differently at
train vs serve time, whereas the fitted transform changes *columns* identically
in both. Conflating the two is a classic source of train/serve skew.

Train time:  Inf -> NaN, drop duplicate flows, drop all-NaN rows.
Serve time:  Inf -> NaN only. We never drop a live event — every event must get
             a score — and we cannot "deduplicate" a single incoming request.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threat_detection.data import schema
from threat_detection.logging_utils import get_logger

log = get_logger(__name__)


def replace_inf_with_nan(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Replace +/-Inf with NaN so the downstream imputer can handle them.

    Real CICIDS2017 rate columns (bytes/s, packets/s) contain Inf from
    divide-by-near-zero-duration. Imputers can't fill Inf, so we normalise to
    NaN first. Returns a copy; the input is not mutated.
    """
    out = df.copy()
    cols = columns if columns is not None else list(schema.FLOW_FEATURES)
    present = [c for c in cols if c in out.columns]
    out[present] = out[present].replace([np.inf, -np.inf], np.nan)
    return out


def clean_flows(df: pd.DataFrame, *, drop_duplicates: bool, training: bool) -> pd.DataFrame:
    """Clean a raw flow table.

    Args:
        df: Raw flow DataFrame (features + optional Label).
        drop_duplicates: Whether duplicate-dropping is enabled in config.
        training: True for the training path (drops dupes / all-NaN rows);
            False for the serving path (preserves every row).

    Returns:
        A cleaned copy of ``df``.
    """
    n_before = len(df)
    out = replace_inf_with_nan(df)

    if training:
        if drop_duplicates:
            out = out.drop_duplicates().reset_index(drop=True)
        # Drop rows where every feature is NaN — they carry no signal.
        feature_cols = [c for c in schema.FLOW_FEATURES if c in out.columns]
        all_nan = out[feature_cols].isna().all(axis=1)
        if all_nan.any():
            out = out.loc[~all_nan].reset_index(drop=True)
        log.info(
            "clean_flows.train",
            rows_in=n_before,
            rows_out=len(out),
            dropped=n_before - len(out),
        )
    else:
        log.debug("clean_flows.serve", rows=len(out))

    return out
