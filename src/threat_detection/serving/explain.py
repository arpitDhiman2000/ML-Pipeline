"""Fast, per-request explanations for analysts.

An analyst must see *why* an event was flagged — a black-box score is not
actionable. We return:
  * top contributing flow features (importance x standardised magnitude — a fast
    local attribution; SHAP TreeExplainer can be swapped in if a fuller
    attribution is needed, at higher latency), and
  * which known-malicious indicators fired in the payload.
Both are cheap enough to stay well inside the p99 latency budget.
"""

from __future__ import annotations

import re

import numpy as np

from threat_detection.data import schema

# Known-malicious payload signatures surfaced to the analyst.
_INDICATORS: dict[str, re.Pattern] = {
    "sql_injection": re.compile(r"(?i)(union\s+select|drop\s+table|or\s+'1'='1'|--\s*$)"),
    "shell_injection": re.compile(r"(\$\(|;\s*(rm|nc|bash|sh|wget|curl)\b|/bin/sh)"),
    "path_traversal": re.compile(r"\.\./"),
    "xss": re.compile(r"(?i)<script"),
    "credential_access": re.compile(r"/etc/(passwd|shadow)"),
    "encoded_payload": re.compile(r"(?i)powershell\s+-enc|base64\s+-d"),
}


def tabular_top_features(importances: np.ndarray, scaled_row: np.ndarray, k: int) -> list[dict]:
    """Top-k flow features by (global importance x local standardised magnitude)."""
    contribution = importances * np.abs(scaled_row)
    order = np.argsort(contribution)[::-1][:k]
    return [
        {
            "feature": schema.FLOW_FEATURES[i],
            "scaled_value": round(float(scaled_row[i]), 4),
            "importance": round(float(importances[i]), 4),
        }
        for i in order
    ]


def payload_indicators(payload: str | None) -> list[str]:
    """Names of malicious indicators that match the payload."""
    if not payload:
        return []
    return [name for name, pat in _INDICATORS.items() if pat.search(payload)]
