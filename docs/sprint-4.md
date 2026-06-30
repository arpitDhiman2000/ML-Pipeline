# Sprint 4 — Fusion + FastAPI Serving

**Goal:** turn three model scores into one decision, served behind a production
FastAPI service with validation, explanations, metrics, and a latency budget.
This is the backbone of the *"model inference"* test bullet and the deploy story.

## What was built

| Module | Responsibility |
|--------|----------------|
| [`models/fusion.py`](../src/threat_detection/models/fusion.py) | `FusionMetaLearner` — standardise → logistic regression over `[anomaly_score, attack_proba, text_proba]`. `average_baseline` is the naive alternative it must beat. |
| [`training/train_fusion.py`](../src/threat_detection/training/train_fusion.py) | Builds a synthetic **joint** (flow, payload) set correlated by label, trains fusion, compares vs averaging, logs to MLflow, registers `threat-detection-fusion`. |
| [`serving/schemas.py`](../src/threat_detection/serving/schemas.py) | Pydantic request/response — the validated boundary (an `Event` needs `features` and/or `payload`). |
| [`serving/model_bundle.py`](../src/threat_detection/serving/model_bundle.py) | Loads all wrapper artifacts once; `score()` runs preprocess → anomaly → classifier → text → fusion. |
| [`serving/explain.py`](../src/threat_detection/serving/explain.py) | Fast per-request explanation: top flow features + matched malicious indicators (SQLi, shell, traversal, XSS…). |
| [`serving/metrics.py`](../src/threat_detection/serving/metrics.py) | Prometheus counters/histograms: request volume, latency, **threat-probability distribution** (the first drift signal). |
| [`serving/app.py`](../src/threat_detection/serving/app.py) | FastAPI app: `GET /health`, `GET /model-info`, `POST /score`, `POST /batch`, `GET /metrics`. Bundle loaded once at startup (lifespan). |
| [`load/locustfile.py`](../load/locustfile.py) | Locust load test for throughput / p99 latency vs the eval-gate budget. |

## Interview points seeded here
- **Why a learned meta-learner over averaging?** The three scores have different scales/reliabilities; averaging treats them as equal. The meta-learner learns the weighting — `pr_auc_uplift_vs_averaging` proves it (it learns to ignore an uninformative score).
- **Train/serve consistency:** serving reuses the *same* fitted preprocessor + tokenizer + model wrappers — no second code path.
- **Registry as governance:** `@production` aliases say which versions are live (`/model-info`); serving loads the DVC-versioned artifact bytes.
- **Explainability in prod:** every decision returns top features + triggering indicators, so a flag is auditable.
- **Latency budget:** the eval gate's `max_p99_latency_ms` is validated with Locust against the running service.

## Decision routing (graceful degradation)
The fusion meta-learner is trained with all three scores present, so feeding
fabricated "benign" defaults for an absent modality lets it override a real
signal. The bundle therefore routes:
- **both modalities** → fusion meta-learner;
- **payload only** → the text classifier's own decision (so a lone malicious
  payload is never suppressed by absent tabular scores — caught in testing);
- **flow only** → fusion with a neutral text prior (real tabular signals drive it).

The response includes `decision_source` so every verdict is traceable.

## Endpoints
```
GET  /health       -> {status, models_loaded, decision_threshold}
GET  /model-info   -> {production_versions: {model: version}, ...}
POST /score        -> {threat_probability, is_threat, confidence, predicted_class,
                       component_scores, explanation, latency_ms}
POST /batch        -> {results: [...], count}
GET  /metrics      -> Prometheus exposition
```

## Tests added (29 new → **112 total, all green**)
- **fusion**: fit/predict, average baseline, **meta beats averaging**, save/load parity.
- **serving unit**: schema validation (empty event → error), indicator detection, top-feature ranking.
- **serving integration**: trains all 3 models once, drives the real app — `/health`, `/score` (with SQLi indicator), payload-only, empty-event 422, `/batch`, `/metrics`.

## Done when ✓
An end-to-end request scores a live event with explanation under the latency
budget. Next: **Sprint 5** containerises this service and wires CI/CD + AWS.
