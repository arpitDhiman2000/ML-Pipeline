"""MLflow Model Registry helpers — promotion via aliases.

Why aliases (interview point): a registry accumulates a new version on every
(re)train, so "which one is live?" must be explicit. We pin a ``production``
alias to the chosen version; serving loads ``models:/<name>@production`` and never
hardcodes a version number. Retraining (Sprint 6) re-points the alias to a new
candidate only after it clears the eval gate, and rollback is just moving the
alias back — no redeploy.
"""

from __future__ import annotations

from threat_detection.logging_utils import get_logger

log = get_logger(__name__)

# Canonical registered-model names (single source of truth).
TABULAR_MODEL = "threat-detection-tabular"
TEXT_MODEL = "threat-detection-text"
FUSION_MODEL = "threat-detection-fusion"
ALL_MODELS = (TABULAR_MODEL, TEXT_MODEL, FUSION_MODEL)

PRODUCTION_ALIAS = "production"


def get_client():
    """Return an MLflow client wired to the configured tracking server."""
    from threat_detection.tracking import configure_mlflow

    configure_mlflow()
    import mlflow

    return mlflow.MlflowClient()


def latest_version(client, name: str) -> str | None:
    """Highest version number registered under ``name`` (None if unregistered)."""
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return None
    return str(max(int(v.version) for v in versions))


def production_uri(name: str, alias: str = PRODUCTION_ALIAS) -> str:
    """The alias-pinned model URI serving should load."""
    return f"models:/{name}@{alias}"


def promote(name: str, *, version: str | None = None, alias: str = PRODUCTION_ALIAS) -> str:
    """Point ``alias`` at ``version`` (default: the latest) of ``name``.

    Returns the version that the alias now references.
    """
    client = get_client()
    target = version or latest_version(client, name)
    if target is None:
        raise ValueError(f"No registered versions found for model '{name}'")
    client.set_registered_model_alias(name, alias, target)
    log.info("registry.promoted", model=name, version=target, alias=alias)
    return target
