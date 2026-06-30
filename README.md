# Threat Detection ML Pipeline

A near-real-time network threat detector with a full production MLOps backbone.
Rebuilt to be **defensible in an interview**: every component is a decision you
can justify as *"I chose X over Y because Z, validated with W."*

* **Tabular path** — Isolation Forest (unsupervised anomaly) + XGBoost (supervised attack classification) over CICFlowMeter flow features.
* **Text path** — LSTM over command/payload strings (malicious vs benign).
* **Fusion** — a stacked meta-learner turns three scores into one decision.
* **MLOps** — DVC (data/versioning), MLflow (tracking/registry), FastAPI (serving), Docker + GitHub Actions (CI/CD), AWS (S3/ECR/EC2), Evidently (drift) → gated retraining, Kibana (analyst dashboard).

Data is anchored on the **CICIDS2017** schema. A synthetic generator produces a
schema-faithful dataset (severe imbalance, Inf/NaN rate columns, duplicate
flows) so the whole pipeline runs in seconds; dropping real CICIDS2017 CSVs into
`data/raw/cicids2017/` swaps in real data with no code change.

## Repository layout

```
threat-detection/
├─ src/threat_detection/
│  ├─ data/          # schema, synthetic generator, real-CSV loaders   (Sprint 0)
│  ├─ ingestion/     # stream consumers, S3 writers                    (later)
│  ├─ features/      # leak-free transforms, tokenizer                 (Sprint 1)
│  ├─ models/        # anomaly / classifier / text_lstm / fusion       (Sprint 2-4)
│  ├─ training/      # train loops, Optuna, MLflow logging             (Sprint 2-4)
│  ├─ evaluation/    # metrics, eval gate, drift checks                (Sprint 2,6)
│  ├─ serving/       # FastAPI app, schemas, model loader              (Sprint 4)
│  ├─ config.py      # typed params.yaml loader
│  ├─ paths.py       # env-overridable filesystem paths
│  ├─ logging_utils.py
│  └─ cli.py         # `td` command-line interface
├─ tests/            # unit + integration (pytest)
├─ pipelines/        # (DVC stages live in dvc.yaml)
├─ docker/           # Dockerfile, compose                             (Sprint 5)
├─ .github/workflows/# CI/CD                                           (Sprint 5)
├─ configs/          # YAML config (paths, etc.)
├─ params.yaml       # single source of truth for all parameters
├─ dvc.yaml          # reproducible pipeline DAG
└─ pyproject.toml    # deps (uv), tooling config
```

## Prerequisites

* **Python 3.12** (managed by `uv`; do not use 3.14 — ML wheels lag)
* **uv** (already installed), **Docker**, **Git**

## Quickstart (Sprint 0)

```powershell
# 1. Create the 3.12 virtual env and install Sprint-0 dependencies
uv sync

# 2. Verify the config and generate the raw data zone
uv run td show-config
uv run td generate

# 3. Run the test suite
uv run pytest
```

See [`docs/`](docs/) — each sprint appends a short "what & how" note.

### Cloud (DagsHub: DVC remote + hosted MLflow)

Experiments and versioned data live on DagsHub. One-time setup (your account +
token) is in [`docs/dagshub.md`](docs/dagshub.md). Quick check after setup:

```powershell
Copy-Item .env.example .env   # then fill in your DagsHub MLflow creds
uv run td mlflow-smoke        # logs a run; see it in the DagsHub Experiments tab
```

## Dependency groups

Installed per sprint to keep environments light (see `pyproject.toml`):

| Group     | Sprint | Adds                                            |
|-----------|--------|-------------------------------------------------|
| `dev`,`mlops` | 0–1 | pytest/ruff/mypy, MLflow, DVC, Evidently     |
| `ml`      | 2      | scikit-learn, xgboost, imbalanced-learn, optuna |
| `text`    | 3      | torch                                           |
| `serving` | 4      | fastapi, uvicorn, prometheus-client             |
| `cloud`   | 5–6    | boto3, dvc-s3, locust                           |

```powershell
uv sync --group ml        # example: add Sprint-2 deps
uv sync --all-groups      # everything
```
