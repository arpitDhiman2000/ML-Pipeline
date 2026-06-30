"""Generate the raw data zone (Sprint 0 / DVC stage `generate_data`).

If real CICIDS2017 CSVs already exist at the configured raw path, generation is
skipped for the tabular side so real data is never overwritten. This is the
function the DVC pipeline calls, and the one the CLI exposes as `td generate`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from threat_detection.config import AppConfig, get_config
from threat_detection.data import synthetic
from threat_detection.logging_utils import get_logger
from threat_detection.paths import Paths

log = get_logger(__name__)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    Paths.ensure(path)
    df.to_parquet(path, index=False)


def generate_raw_data(
    config: AppConfig | None = None,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Generate (or confirm) the raw tabular and text datasets.

    Args:
        config: Validated app config; defaults to the project's ``params.yaml``.
        force: Regenerate even if synthetic files already exist.

    Returns:
        Mapping of dataset name -> path written.
    """
    cfg = config or get_config()
    tabular_path = Paths.tabular_raw()
    text_path = Paths.text_raw()

    drift_scale = cfg.data.drift.shift_scale if cfg.data.drift.enabled else 1.0

    # --- Tabular ---
    real_csv_dir = tabular_path.parent  # data/raw/cicids2017/
    has_real_csv = real_csv_dir.is_dir() and any(real_csv_dir.glob("*.csv"))
    if has_real_csv:
        log.info("tabular.real_data_present", dir=str(real_csv_dir), action="skip_synth")
    elif tabular_path.exists() and not force:
        log.info("tabular.exists", path=str(tabular_path), action="skip")
    else:
        log.info("tabular.generating", n_rows=cfg.data.tabular.n_rows, drift=drift_scale)
        df = synthetic.generate_tabular(cfg.data.tabular, cfg.seed, drift_scale=drift_scale)
        _write_parquet(df, tabular_path)
        log.info("tabular.written", path=str(tabular_path), rows=len(df))

    # --- Text ---
    if text_path.exists() and not force:
        log.info("text.exists", path=str(text_path), action="skip")
    else:
        log.info("text.generating", n_rows=cfg.data.text.n_rows)
        df_text = synthetic.generate_text(cfg.data.text, cfg.seed)
        _write_parquet(df_text, text_path)
        log.info("text.written", path=str(text_path), rows=len(df_text))

    return {"tabular": tabular_path, "text": text_path}
