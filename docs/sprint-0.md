# Sprint 0 — Scaffold & Data

**Goal:** a reproducible repo, environment, and data foundation that every later
sprint builds on.

## What was built

| Area | Deliverable |
|------|-------------|
| Environment | `pyproject.toml` pinned to **Python 3.12** via `uv`, with per-sprint dependency groups (`dev`, `ml`, `text`, `serving`, `mlops`, `cloud`) so each environment stays light. |
| Repo layout | Layered `src/threat_detection/` package (`data`, `ingestion`, `features`, `models`, `training`, `evaluation`, `serving`) + `tests/`, `configs/`, `docs/`. |
| Config | `params.yaml` (single source of truth, DVC-tracked) parsed into **typed Pydantic** models (`config.py`) that fail loudly on bad params. `configs/paths.yaml` + `paths.py` give env-overridable paths (`THREAT_DETECTION_DATA_ROOT`). |
| Data schema | `data/schema.py` — the canonical contract: 78 CICFlowMeter features (incl. the real duplicate `Fwd Header Length.1`), 7 attack classes + BENIGN. |
| Synthetic data | `data/synthetic.py` — schema-faithful generator with **severe imbalance (~75% benign), injected Inf/NaN in rate columns, and duplicate flows** (mirrors real CICIDS2017 quirks). Fully seeded → byte-reproducible. Has a `drift_scale` hook for Sprint 6. |
| Real-ready loaders | `data/loaders.py` — strips the leading-space column quirk and maps granular real labels (`DoS Hulk`, `Web Attack – XSS`, …) to the coarse class space. Drop real CSVs in `data/raw/cicids2017/` and the code is unchanged. |
| CLI | `td generate`, `td show-config` (Typer). |
| Reproducibility | Git + **DVC** initialised with a local remote; `dvc.yaml` defines the `generate_data` stage; `dvc repro` regenerates the exact dataset and writes `dvc.lock`. |
| Quality | structured JSON logging (`structlog`), `ruff` + `mypy` config, `pre-commit` hooks. |
| Tests | 23 tests (unit + 1 integration): schema fidelity, config validation, generator determinism/imbalance/quality-injection/drift, path override, generate→load roundtrip. |

## Why it's done

`git checkout` + `dvc pull` (or `dvc repro`) reproduces the exact dataset on a
clean clone — the Sprint-0 acceptance criterion. All 23 tests green, lint clean.

## Key interview points seeded here
- **Train/serve consistency starts at config**: one validated `AppConfig` used everywhere.
- **Reproducibility**: seeded generator + DVC DAG + `params.yaml` = any run is re-runnable.
- **Real-world data quality**: the generator injects the exact failure modes (Inf/NaN, dupes, imbalance) so preprocessing in Sprint 1 is genuine, not a toy.
