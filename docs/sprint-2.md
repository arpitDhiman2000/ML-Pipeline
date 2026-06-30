# Sprint 2 — Tabular Models + Eval Gate

**Goal:** Isolation Forest + XGBoost with **honest, leak-free evaluation**, logged
to MLflow (DagsHub), gated by an automated promotion rule. Models are intentionally
kept lightweight (no heavy tuning) — the focus is the *mechanism* (tracking, gate,
inference tests) that later sprints build on.

## What was built

| Module | Responsibility |
|--------|----------------|
| [`models/anomaly.py`](../src/threat_detection/models/anomaly.py) | `AnomalyDetector` (Isolation Forest). **Contamination tied to observed attack prevalence**, not guessed. Returns `anomaly_score` (higher = more anomalous). The zero-day safety net. |
| [`models/classifier.py`](../src/threat_detection/models/classifier.py) | `AttackClassifier` (XGBoost, 8-class). Per-sample **class-balanced weights** for imbalance; early stopping on val; `attack_proba = 1 − P(BENIGN)` is the single binary threat score. |
| [`evaluation/metrics.py`](../src/threat_detection/evaluation/metrics.py) | PR-AUC, MCC, per-class recall, and **PR-curve threshold selection** at a target recall (the security trade-off encoded). |
| [`evaluation/gate.py`](../src/threat_detection/evaluation/gate.py) | The **eval gate** — promotion rule as data: PR-AUC floor AND worst per-class recall floor AND no PR-AUC regression vs incumbent AND p99-latency budget (enforced from Sprint 4). Reused by CI (Sprint 5) and retraining (Sprint 6). |
| [`evaluation/leakage.py`](../src/threat_detection/evaluation/leakage.py) | `compare_smote_leakage` — proves the leaked (SMOTE-before-split) vs leak-free (SMOTE-inside-CV) PR-AUC gap. |
| [`training/train_tabular.py`](../src/threat_detection/training/train_tabular.py) | Orchestrates train → evaluate → **log to MLflow** → save native artifacts → **register model** → run gate. Registration degrades gracefully if the backend has no registry. |

## Interview points seeded here
- **Why not accuracy?** ~75% benign ⇒ accuracy rewards an "all-benign" predictor. We headline PR-AUC / recall / MCC / per-class recall.
- **Why two models?** XGBoost knows *labeled* attacks; Isolation Forest covers the *unknown* (zero-day). Defensible contamination = observed prevalence.
- **Imbalance without cheating:** class-weighting in XGBoost, and SMOTE only inside CV folds — `leakage.py` shows the inflated score you get if you don't.
- **The gate** is the same rule that will block a bad deploy in CI and a bad retrain in the drift loop — codified once, tested, reused.

## MLflow / DagsHub
`td train-tabular` logs params, the full metric set, the gate decision (`gate.json`),
and both models to your **DagsHub MLflow** (Experiments tab), registering the
classifier as `threat-detection-tabular` (Models tab).

## Tests added (26 new → **77 total, all green**)
- **metrics**: PR-AUC/MCC correctness, per-class recall, threshold meets target recall.
- **gate**: pass/fail on each rule, incumbent regression, latency budget.
- **models**: contamination tie-in, fit/score/predict shapes, `attack_proba ∈ [0,1]`, **save/load inference parity** (anomaly + classifier).
- **leakage**: leaked PR-AUC ≥ leak-free (the inflation).
- **integration**: generate → preprocess → **train end to end**, models persisted, metrics in valid ranges, PR-AUC > chance.

## Done when ✓
`dvc repro` runs `… → train_tabular`; models saved to `artifacts/models/`; every
run logged + gated in MLflow. (Reproducing exact ~82%/76% is a tuning exercise —
`xgboost.tune: true` enables Optuna — deliberately left off per the project's
deploy/monitoring focus.)
