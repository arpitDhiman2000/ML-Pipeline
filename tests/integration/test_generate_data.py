"""Integration test: generate raw data to disk, then read it back canonically.

This exercises the real seam between the generator, the parquet writer, and the
loaders — the path the DVC `generate_data` stage actually takes.
"""

from __future__ import annotations

import pytest

from threat_detection.config import AppConfig
from threat_detection.data import schema
from threat_detection.data.loaders import load_tabular, load_text

pytestmark = pytest.mark.integration


def test_generate_then_load_roundtrip(
    small_app_cfg: AppConfig, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # Relocate the data root so the test never touches the real data zone.
    monkeypatch.setenv("THREAT_DETECTION_DATA_ROOT", str(tmp_path))
    import importlib

    from threat_detection import paths as paths_mod

    importlib.reload(paths_mod)
    # generate.py imports Paths at call time, so reload propagates the override.
    import threat_detection.data.generate as gen_mod

    importlib.reload(gen_mod)

    written = gen_mod.generate_raw_data(small_app_cfg, force=True)

    assert written["tabular"].exists()
    assert written["text"].exists()

    tab = load_tabular(written["tabular"])
    assert list(tab.columns) == list(schema.tabular_columns())
    assert len(tab) >= small_app_cfg.data.tabular.n_rows  # >= due to duplicates

    txt = load_text(written["text"])
    assert list(txt.columns) == [schema.TEXT_COLUMN, schema.TEXT_LABEL_COLUMN]
    assert len(txt) == small_app_cfg.data.text.n_rows
