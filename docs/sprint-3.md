# Sprint 3 — LSTM Text Classifier

**Goal:** a sequence model for malicious commands/payloads, justified against a
cheap baseline. Kept lightweight (small BiLSTM, few epochs, CPU-friendly) per the
project's deploy/monitoring focus.

## What was built

| Module | Responsibility |
|--------|----------------|
| [`models/text_lstm.py`](../src/threat_detection/models/text_lstm.py) | `LSTMNet` (embedding → (Bi)LSTM → dropout → 1-logit head) + `TextClassifier` wrapper: encodes via the **frozen Sprint-1 tokenizer**, trains with **class-weighted** `BCEWithLogitsLoss`, early stops on val loss, selects the operating threshold for a target recall, and saves/loads with inference parity. |
| [`models/text_baseline.py`](../src/threat_detection/models/text_baseline.py) | `TfidfBaseline` — char (1–3)-gram TF-IDF + class-balanced logistic regression. The bar the LSTM must clear. |
| [`training/train_text.py`](../src/threat_detection/training/train_text.py) | Loads processed text splits → trains LSTM (per-epoch loss curves to MLflow) → threshold on val → evaluates on test → **bakes off vs the baseline** (F1 uplift) → saves artifact → registers model. |

## Interview points seeded here
- **Why an LSTM, not BoW/TF-IDF?** Order encodes intent in commands/payloads; the LSTM models that sequence, BoW discards it. Proven by the **F1 bake-off** (`f1_uplift_vs_baseline` logged every run).
- **Why char-level tokens?** Payloads are full of hostile, non-dictionary tokens (obfuscated shell, SQLi); a small char vocab generalises and never explodes.
- **Why not BERT?** Short inputs, narrow vocab, 50K/day latency+cost budget — a small LSTM lands close on F1 far cheaper. (The DistilBERT reference is an optional extension; the cheap-baseline bake-off already makes the point.)
- **Imbalance:** `pos_weight` in the loss up-weights the rare malicious class; threshold chosen on the PR curve for a recall target, same security trade-off as the tabular path.

## MLflow / DagsHub
`td train-text` logs hyperparameters, **per-epoch train/val loss curves**, the
LSTM-vs-baseline metrics, and the model to your DagsHub MLflow (Experiments tab),
registering it as `threat-detection-text`.

## Tests added (6 new → **83 total, all green**)
- **net forward shape**, `predict_proba ∈ [0,1]`, **learns malicious > benign** on clear payloads, **save/load inference parity** (incl. threshold), baseline separates classes.
- **integration**: generate → preprocess → **train end to end**, artifact persisted, PR-AUC > 0.6.

## Done when ✓
LSTM trains, beats/levels the BoW baseline on this signal, artifact saved to
`artifacts/models/text_lstm.pt`, every run logged in MLflow. Next: **Sprint 4**
fuses the tabular + text scores behind a FastAPI service.
