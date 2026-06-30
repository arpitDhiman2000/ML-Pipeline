"""Demonstrate the SMOTE data-leak that inflates scores.

This is the interview story about doing imbalance handling *honestly*:

  * WRONG (leaked): oversample with SMOTE on the FULL dataset, then cross-
    validate. SMOTE synthesises minority points by interpolating neighbours, so
    synthetic copies of validation points leak into the training folds. The
    cross-validated PR-AUC comes out optimistically inflated.

  * RIGHT (leak-free): oversample INSIDE each fold, fitting SMOTE only on that
    fold's training partition. The validation fold stays untouched, so the score
    is honest.

``compare_smote_leakage`` returns both numbers; the gap (leaked - leak_free) is
the lesson. We use a cheap LogisticRegression so the demo runs in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class LeakageComparison:
    leaked_pr_auc: float
    leak_free_pr_auc: float

    @property
    def gap(self) -> float:
        return self.leaked_pr_auc - self.leak_free_pr_auc


def compare_smote_leakage(
    X: np.ndarray,
    y_binary: np.ndarray,
    *,
    seed: int = 42,
    n_splits: int = 5,
) -> LeakageComparison:
    """Cross-validated PR-AUC with SMOTE applied leakily vs leak-free."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    k = min(5, int(y_binary.sum()) - 1)
    k = max(1, k)

    # --- leaked: resample the whole dataset up front, then CV on it ---
    X_res, y_res = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y_binary)
    leaked = cross_val_score(
        LogisticRegression(max_iter=1000),
        X_res,
        y_res,
        cv=cv,
        scoring="average_precision",
    ).mean()

    # --- leak-free: SMOTE lives INSIDE the CV pipeline, fit per training fold ---
    pipe = ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=seed, k_neighbors=k)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    leak_free = cross_val_score(pipe, X, y_binary, cv=cv, scoring="average_precision").mean()

    return LeakageComparison(leaked_pr_auc=float(leaked), leak_free_pr_auc=float(leak_free))
