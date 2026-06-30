# Sprint 1 — Preprocessing & Feature Engineering

**Goal:** a clean, **leak-free** feature pipeline that is shared *identically* by
training and serving — directly backing the resume bullet *"unit tests covering
feature engineering."*

## What was built

| Module | Responsibility |
|--------|----------------|
| [`features/cleaning.py`](../src/threat_detection/features/cleaning.py) | Row-level cleaning. **Train mode**: Inf→NaN, drop duplicate flows, drop all-NaN rows. **Serve mode**: Inf→NaN only — every live event must be scored and you can't dedupe a single request. Separating row-cleaning from the fitted transform avoids a classic skew bug. |
| [`features/target.py`](../src/threat_detection/features/target.py) | Stable, **schema-derived** label↔index mapping (BENIGN=0 forever) + binary `is_attack` flag. Not a fitted `LabelEncoder`, so class ids never drift between runs. |
| [`features/tabular.py`](../src/threat_detection/features/tabular.py) | `TabularPreprocessor`: `Inf→NaN → median impute → standardize`, fit on **train only**, saved as a single versioned `joblib` artifact. Reindexes incoming columns to canonical order, so reordered/missing columns at inference are handled identically. |
| [`features/text.py`](../src/threat_detection/features/text.py) | `TextTokenizer`: char-level (default) vocab built on **train only**, frozen to a JSON artifact, deterministic, with `<pad>`/`<unk>`, fixed-length encode (pad/truncate). |
| [`features/build.py`](../src/threat_detection/features/build.py) | Orchestrates the `preprocess` DVC stage: clean → **stratified split** → fit-on-train → transform all splits → persist artifacts + processed parquet. |

## The leak-free design (interview gold)

1. **Dedupe before split** so identical flows can't straddle train/test.
2. **Fit imputer/scaler/vocab on the train split only** — the test set never
   influences the median, scale, or vocabulary. (SMOTE-inside-CV comes in Sprint 2.)
3. **One fitted artifact, reloaded at serve time** — there is no second code path
   that re-derives scaling. This is how train/serve skew is eliminated, and it's
   asserted directly in the tests.

## Tests added (28 new; 51 total, all green)

* **Cleaning** — Inf→NaN targets features only; train drops dupes; **serve preserves every row**.
* **Target** — BENIGN=0, encode/decode round-trip, binary flag, unknown-label guard.
* **Tabular** — output shape `(n, 78)`/float32; **no NaN/Inf in output from dirty input**; standardization centers features; **save/load parity (byte-identical transform)**; column-reorder robustness; missing-column imputation.
* **Text** — fixed-length encode, truncation/padding, `<unk>` mapping, determinism, word-level mode, **save/load parity**.
* **Integration** — full stage to disk: artifacts persist, processed splits are finite (leak-free), reloaded preprocessor (serving path) transforms identically, tokenizer reloads.

## Done when ✓
Tests green; `dvc repro` runs `generate_data → preprocess`; the **same** fitted
`preprocessor.joblib` + `tokenizer.json` load and run at inference time.

Processed output (real run): tabular train≈35k / val≈5k / test=10k; text 14k/2k/4k.
