"""FastAPI serving application.

Endpoints:
  GET  /health      liveness + whether models are loaded
  GET  /model-info  the @production versions backing this service
  POST /score       score a single event (returns decision + explanation)
  POST /batch       score many events
  GET  /metrics     Prometheus exposition

The model bundle is loaded once at startup (lifespan) and held on app.state, so
per-request work is just transform + predict — keeping p99 within budget.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from threat_detection.logging_utils import configure_logging, get_logger
from threat_detection.serving import metrics
from threat_detection.serving.model_bundle import ModelBundle
from threat_detection.serving.schemas import (
    BatchRequest,
    BatchResponse,
    Event,
    HealthResponse,
    ModelInfoResponse,
    ScoreResponse,
)

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        app.state.bundle = ModelBundle.load()
    except Exception as exc:  # serve in a degraded state rather than crash-loop
        log.error("bundle.load_failed", error=str(exc))
        app.state.bundle = None
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Threat Detection API", version="0.1.0", lifespan=lifespan)

    def _bundle(request: Request) -> ModelBundle:
        bundle = request.app.state.bundle
        if bundle is None:
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail="models not loaded")
        return bundle

    def _score_event(bundle: ModelBundle, event: Event, endpoint: str) -> ScoreResponse:
        start = time.perf_counter()
        result = bundle.score(event.features, event.payload)
        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics.LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000.0)
        metrics.record_result(result)
        return ScoreResponse(latency_ms=round(latency_ms, 3), **result)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        bundle = request.app.state.bundle
        threshold = bundle.config.fusion.decision_threshold if bundle else 0.5
        return HealthResponse(
            status="ok" if bundle else "degraded",
            models_loaded=bundle is not None,
            decision_threshold=threshold,
        )

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        bundle = _bundle(request)
        versions: dict[str, str | None] = {}
        try:  # best-effort: registry may be unreachable in an offline container
            from threat_detection.registry import ALL_MODELS, get_client

            client = get_client()
            for name in ALL_MODELS:
                try:
                    versions[name] = client.get_model_version_by_alias(name, "production").version
                except Exception:
                    versions[name] = None
        except Exception as exc:
            log.warning("model_info.registry_unavailable", error=str(exc))
        return ModelInfoResponse(
            production_versions=versions,
            decision_threshold=bundle.config.fusion.decision_threshold,
        )

    @app.post("/score", response_model=ScoreResponse)
    def score(event: Event, request: Request) -> ScoreResponse:
        metrics.REQUESTS.labels(endpoint="score").inc()
        return _score_event(_bundle(request), event, "score")

    @app.post("/batch", response_model=BatchResponse)
    def batch(payload: BatchRequest, request: Request) -> BatchResponse:
        metrics.REQUESTS.labels(endpoint="batch").inc()
        bundle = _bundle(request)
        results = [_score_event(bundle, ev, "batch") for ev in payload.events]
        return BatchResponse(results=results, count=len(results))

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        body, content_type = metrics.metrics_payload()
        return Response(content=body, media_type=content_type)

    return app


# Module-level app for `uvicorn threat_detection.serving.app:app`.
app = create_app()
