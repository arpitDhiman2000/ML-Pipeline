"""Unit tests for the DevOps configs — catch typos before they hit a runner.

These don't run Docker/Actions; they assert the files exist and parse, and that
the key wiring (gate step, dormancy guard, slim runtime) is present.
"""

from __future__ import annotations

import pytest
import yaml

from threat_detection.paths import project_root

pytestmark = pytest.mark.unit

ROOT = project_root()


def test_dockerfile_is_multistage_and_nonroot() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert text.count("FROM ") >= 2  # multi-stage
    assert "USER appuser" in text  # runs non-root
    assert "uvicorn" in text  # serves the app


def test_dockerignore_excludes_heavy_dirs() -> None:
    text = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in (".venv/", "data/", ".git/", "mlruns/"):
        assert pattern in text


def test_compose_parses_and_exposes_port() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    assert "8000:8000" in api["ports"]


def test_ci_workflow_runs_gate() -> None:
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    jobs = ci["jobs"]
    assert "lint-test" in jobs
    assert "eval-gate" in jobs
    # gate depends on tests passing first
    assert jobs["eval-gate"]["needs"] == "lint-test"
    steps = " ".join(str(s) for s in jobs["eval-gate"]["steps"])
    assert "check-gate" in steps


def test_cd_workflow_is_dormant_by_default() -> None:
    cd = yaml.safe_load((ROOT / ".github/workflows/cd.yml").read_text(encoding="utf-8"))
    job = cd["jobs"]["build-and-deploy"]
    # Guarded by the ENABLE_CD opt-in variable so it never fails before AWS setup.
    assert "ENABLE_CD" in job["if"]
    steps = " ".join(str(s) for s in job["steps"])
    assert "ecr" in steps.lower()
