"""Threat Detection ML Pipeline.

A near-real-time network threat detector combining a tabular path
(Isolation Forest + XGBoost) and a text path (LSTM over command/payload
strings), fused into a single decision and wrapped in a production MLOps
backbone (DVC, MLflow, FastAPI, Docker, GitHub Actions, AWS, Evidently).
"""

__version__ = "0.1.0"
