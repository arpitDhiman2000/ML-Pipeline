"""TF-IDF + Logistic Regression baseline for the text bake-off.

The LSTM has to *earn* its place. This cheap char-n-gram baseline is the bar it
must clear on F1 — if a bag-of-character-ngrams model matched the LSTM, the
sequence model wouldn't be justified. Char n-grams (not words) because payloads
are full of hostile, non-dictionary tokens.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfBaseline:
    """Char (1-3)-gram TF-IDF -> class-balanced logistic regression."""

    def __init__(self, *, seed: int = 42):
        self.pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2)),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
                ),
            ]
        )

    def fit(self, texts: list[str], y: np.ndarray) -> TfidfBaseline:
        self.pipeline.fit([str(t) for t in texts], np.asarray(y))
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba([str(t) for t in texts])[:, 1]

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict([str(t) for t in texts]).astype(np.int64)
