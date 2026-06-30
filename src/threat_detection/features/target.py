"""Target derivation: multi-class label <-> index, and the binary attack flag.

The label mapping is derived deterministically from :data:`schema.ALL_LABELS`,
not *fitted* from data. This matters: a fitted ``LabelEncoder`` orders classes by
whatever happens to appear in a given training set, so two training runs could
silently assign different integer ids to the same class. A fixed mapping keeps
class id 3 == "PortScan" forever, across every run and every machine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threat_detection.data import schema

# Stable, schema-derived mappings (BENIGN=0, then attack classes in order).
LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(schema.ALL_LABELS)}
INDEX_TO_LABEL: dict[int, str] = {i: label for label, i in LABEL_TO_INDEX.items()}
NUM_CLASSES: int = len(schema.ALL_LABELS)


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Map string labels to their stable integer ids."""
    unknown = set(labels.unique()) - set(LABEL_TO_INDEX)
    if unknown:
        raise ValueError(f"Unknown labels not in schema: {sorted(unknown)}")
    return labels.map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)


def decode_labels(indices: np.ndarray) -> np.ndarray:
    """Map integer ids back to string labels."""
    return np.array([INDEX_TO_LABEL[int(i)] for i in indices], dtype=object)


def to_binary_target(labels: pd.Series) -> np.ndarray:
    """Derive the binary attack flag: BENIGN -> 0, any attack -> 1."""
    return (labels != schema.BENIGN_LABEL).to_numpy(dtype=np.int64)
