# Sprint 5 — Containerize + CI/CD + Cloud

**Goal (résumé bullet #1):** a reproducible Docker deploy with an automated gate —
*"Automated MLOps deployment via Docker, GitHub Actions CI/CD, and AWS."* Built
local-first so CI works immediately; AWS is opt-in.

## What was built

| Artifact | Responsibility |
|----------|----------------|
| [`Dockerfile`](../Dockerfile) | **Multi-stage** build (uv builder → slim runtime), non-root user, healthcheck, serves the FastAPI app. Only `ml`/`text`/`serving` groups → lean image. |
| [`.dockerignore`](../.dockerignore) | Keeps `.venv`, `data/`, `mlruns/`, `.git` out of the build context. |
| [`docker-compose.yml`](../docker-compose.yml) | One-command local run; mounts `artifacts/` read-only so you can retrain without rebuilding. |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | Runs on every push/PR: **lint → format-check → tests → eval gate**. The gate job trains on synthetic data and runs `td check-gate`; a red gate fails the run. No cloud creds needed. |
| [`.github/workflows/cd.yml`](../.github/workflows/cd.yml) | **build → push to ECR → deploy to EC2**, gated on CI success. **Dormant** until the `ENABLE_CD` repo variable is `true`, so the repo is green before any AWS setup. |
| [`evaluation/gate_check.py`](../src/threat_detection/evaluation/gate_check.py) + `td check-gate` | Re-evaluates the saved model on the test split and applies the eval gate; **exits non-zero on failure** — this is what blocks the deploy. |
| [`docs/aws.md`](aws.md) | One-time AWS setup: CLI, ECR, S3 zones, least-privilege IAM, EC2, GitHub secrets/vars. |

## The pipeline (what a merge to `main` does)
```
push/PR ─▶ CI: lint ─▶ tests ─▶ eval-gate ──(green)──▶ CD (if ENABLE_CD):
                                  │                       dvc pull artifacts
                                  └─(red)─▶ blocked       ─▶ docker build
                                                          ─▶ push to ECR
                                                          ─▶ ssh deploy to EC2
```

## Eval-gate calibration (interview point)
The gate is calibrated to the **current baseline** (PR-AUC ≈ 0.96; worst per-class
recall ≈ 0.20 on rare Infiltration). Floors are `min_pr_auc 0.90`,
`min_per_class_recall 0.15`: the gate's role is to **catch regressions** and a
model that goes blind to a class — not to demand recall the data can't support.
Raise the floors as data/model improve.

## Interview points seeded here
- **Reproducible image**: multi-stage + locked deps ⇒ laptop == CI == prod; no "works on my machine".
- **Gate blocks deploy**: a measurable promotion rule runs in CI; red gate ⇒ no deploy.
- **Least privilege + cost control**: scoped IAM policy; free-tier EC2 you stop when idle.
- **Two DVC remotes**: DagsHub (collab) + S3 (production) — `dvc push -r s3remote`.

## Tests added (5 new → **118 total**, 111 fast + 7 slow)
`test_devops.py`: Dockerfile is multi-stage & non-root, `.dockerignore` excludes
heavy dirs, compose parses & exposes 8000, CI runs the gate after tests, CD is
dormant by default.

## Image size (CPU torch)
The PyPI torch wheel bundles ~7 GB of CUDA libs we never use (we serve on CPU).
`pyproject.toml` pins torch to the **CPU-only index** (`[tool.uv.sources]` +
`[[tool.uv.index]] pytorch-cpu`), cutting the image **9.55 GB → 3.41 GB** so it
fits free-tier EC2's 8 GB disk. Verified: container starts, `/health` ok,
`/score` returns decisions in ~20 ms.

## Done when ✓
A push runs CI (lint → tests → eval gate); a failing gate blocks the deploy. The
image builds and serves locally via `docker compose up`. AWS deploy activates by
setting `ENABLE_CD` + secrets (docs/aws.md).
