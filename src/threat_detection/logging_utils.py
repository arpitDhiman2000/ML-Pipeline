"""Structured logging setup.

Why structlog + JSON (interview point): in production the pipeline's logs are
consumed by machines (CloudWatch, Elasticsearch/Kibana), not just humans. JSON
lines with stable keys are queryable; free-text logs are not. We render
human-friendly colored logs in a TTY (local dev) and JSON everywhere else.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", *, force_json: bool | None = None) -> None:
    """Configure stdlib + structlog once at process startup.

    Args:
        level: Root log level name.
        force_json: Force JSON output regardless of TTY (used in CI/containers).
            When ``None``, JSON is used iff stdout is not a TTY.
    """
    use_json = (not sys.stdout.isatty()) if force_json is None else force_json

    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
