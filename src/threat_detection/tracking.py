"""MLflow tracking configuration (DagsHub-hosted or local file store).

Why env-driven (interview point): the *same* training code logs to a local
``./mlruns`` on a laptop, to a hosted DagsHub MLflow server in the cloud, or to
an AWS-backed server in CI — chosen entirely by ``MLFLOW_TRACKING_URI`` with no
code change. Credentials never live in code; they come from ``.env`` (gitignored)
or real environment variables.

DagsHub setup (see docs/dagshub.md):
    MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
    MLFLOW_TRACKING_USERNAME=<user>
    MLFLOW_TRACKING_PASSWORD=<dagshub-token>
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from threat_detection.logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_EXPERIMENT = "threat-detection"
LOCAL_TRACKING_URI = "file:./mlruns"


def load_env() -> None:
    """Load ``.env`` into the process environment if present (idempotent)."""
    load_dotenv(override=False)


def get_tracking_uri() -> str:
    """Resolve the MLflow tracking URI (defaults to a local file store)."""
    load_env()
    return os.environ.get("MLFLOW_TRACKING_URI", LOCAL_TRACKING_URI)


def is_remote(uri: str | None = None) -> bool:
    """True when tracking points at a remote server (http/https), not local files."""
    uri = uri or get_tracking_uri()
    return uri.startswith(("http://", "https://"))


def configure_mlflow(experiment: str = DEFAULT_EXPERIMENT):
    """Point MLflow at the configured tracking server and select the experiment.

    Returns the active :class:`mlflow.entities.Experiment`. Imported lazily so
    that modules importing this one don't pay the mlflow import cost unless they
    actually track a run.
    """
    import mlflow

    uri = get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    exp = mlflow.set_experiment(experiment)
    log.info(
        "mlflow.configured",
        tracking_uri=uri,
        remote=is_remote(uri),
        experiment=experiment,
    )
    return exp
