"""Prometheus instrumentation for the serving layer.

These counters/histograms are what Grafana (ops) and the Kibana/analyst views
(Sprint 6) chart over time: request volume, latency percentiles, and the live
distribution of threat scores — the first signal that something has drifted.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter("td_requests_total", "Total scoring requests", ["endpoint"])
LATENCY = Histogram(
    "td_request_latency_seconds",
    "Scoring latency in seconds",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
THREATS = Counter("td_threats_total", "Events flagged as threats")
EVENTS_SCORED = Counter("td_events_scored_total", "Total events scored")
THREAT_PROBABILITY = Histogram(
    "td_threat_probability",
    "Distribution of fused threat probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


def record_result(result: dict) -> None:
    """Update score-distribution metrics from one scored event."""
    EVENTS_SCORED.inc()
    THREAT_PROBABILITY.observe(result["threat_probability"])
    if result["is_threat"]:
        THREATS.inc()


def metrics_payload() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
