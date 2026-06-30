"""Unit test for the SMOTE-leakage demonstration.

The point: applying SMOTE before the CV split leaks synthetic neighbours into
the validation folds and inflates PR-AUC. The leak-free score (SMOTE inside each
fold) is the honest one. On this deliberately noisy dataset, the leaked score
should come out at least as high as the leak-free score — usually higher.
"""

from __future__ import annotations

import numpy as np
import pytest

from threat_detection.evaluation.leakage import compare_smote_leakage

pytestmark = pytest.mark.unit


def test_leaked_score_is_inflated() -> None:
    rng = np.random.default_rng(0)
    n = 500
    # Noisy, imbalanced, only partially separable -> leakage has room to inflate.
    X = rng.normal(size=(n, 6))
    signal = X[:, 0] + X[:, 1] + rng.normal(scale=2.5, size=n)
    y = (signal > 2.0).astype(int)  # ~15-20% positives
    # guarantee both classes are present and minority is non-trivial
    assert 10 < y.sum() < n - 10

    result = compare_smote_leakage(X, y, seed=0, n_splits=5)

    assert 0.0 <= result.leak_free_pr_auc <= 1.0
    assert 0.0 <= result.leaked_pr_auc <= 1.0
    # Leakage inflates (or at worst ties within noise) the cross-validated score.
    assert result.leaked_pr_auc >= result.leak_free_pr_auc - 0.02
