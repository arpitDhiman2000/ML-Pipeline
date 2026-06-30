"""Pydantic request/response schemas — the validated API boundary.

Validating at the boundary (interview point) means a malformed event is rejected
with a clear 422 before it ever reaches a model, and the OpenAPI docs are
generated for free. An event carries flow ``features`` and/or a ``payload``
string; at least one must be present.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Event(BaseModel):
    """One event to score: flow features and/or a command/payload string."""

    features: dict[str, float] | None = Field(
        default=None, description="CICFlowMeter flow features (name -> value)"
    )
    payload: str | None = Field(default=None, description="Raw command/payload string")

    @model_validator(mode="after")
    def _at_least_one_modality(self) -> Event:
        if not self.features and self.payload is None:
            raise ValueError("event must include 'features' and/or 'payload'")
        return self


class Explanation(BaseModel):
    top_features: list[dict] = Field(default_factory=list)
    triggering_indicators: list[str] = Field(default_factory=list)


class ScoreResponse(BaseModel):
    threat_probability: float
    is_threat: bool
    confidence: float
    predicted_class: str
    decision_source: str  # 'fusion' | 'text-only' | 'tabular-only'
    component_scores: dict[str, float]
    explanation: Explanation
    latency_ms: float


class BatchRequest(BaseModel):
    events: list[Event] = Field(..., min_length=1)


class BatchResponse(BaseModel):
    results: list[ScoreResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    decision_threshold: float


class ModelInfoResponse(BaseModel):
    production_versions: dict[str, str | None]
    decision_threshold: float
