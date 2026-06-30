"""Loaders that present raw data through the canonical schema.

The pipeline never reads CSV/parquet directly elsewhere — it goes through these
loaders, which guarantee the canonical column names and label space regardless
of whether the data is synthetic or real CICIDS2017. This is the seam that makes
the project "real-ready": swap the bytes on disk, keep the code.

Real CICIDS2017 quirks handled here:
  * Column names ship with leading spaces (" Destination Port") -> stripped.
  * Labels use granular names ("DoS Hulk", "Web Attack - XSS") -> mapped to the
    coarse classes in :data:`schema.ATTACK_CLASSES`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from threat_detection.data import schema

# Maps granular real CICIDS2017 labels -> our coarse class space. Matching is
# done case-insensitively on a normalised substring, so new variants map too.
_LABEL_KEYWORDS: list[tuple[str, str]] = [
    ("ddos", "DDoS"),
    ("dos", "DoS"),
    ("portscan", "PortScan"),
    ("port scan", "PortScan"),
    ("brute", "Brute Force"),
    ("patator", "Brute Force"),
    ("web attack", "Web Attack"),
    ("xss", "Web Attack"),
    ("sql injection", "Web Attack"),
    ("bot", "Botnet"),
    ("infiltration", "Infiltration"),
    ("heartbleed", "Infiltration"),
    ("benign", schema.BENIGN_LABEL),
]


def normalise_label(raw_label: str) -> str:
    """Map a raw (possibly granular) label to the canonical class space."""
    text = str(raw_label).strip().lower()
    for keyword, canonical in _LABEL_KEYWORDS:
        if keyword in text:
            return canonical
    # Unknown attack variant -> treat as a generic attack rather than dropping.
    return schema.BENIGN_LABEL if text in {"", "nan"} else "DoS"


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and standardise the label column."""
    df = df.rename(columns=lambda c: str(c).strip())
    # Real files name the label column "Label"; tolerate casing variants.
    for cand in ("Label", "label", "LABEL"):
        if cand in df.columns:
            df = df.rename(columns={cand: schema.LABEL_COLUMN})
            break
    return df


def load_tabular(path: Path) -> pd.DataFrame:
    """Load a flow table (parquet or CSV) and return it in canonical form.

    Accepts either the synthetic parquet or a directory/file of real CICIDS2017
    CSVs. Only the canonical feature columns + Label are retained, in order.
    """
    if path.is_dir():
        frames = [pd.read_csv(p, low_memory=False) for p in sorted(path.glob("*.csv"))]
        if not frames:
            raise FileNotFoundError(f"No CSV files found in {path}")
        df = pd.concat(frames, ignore_index=True)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    df = normalise_columns(df)
    if schema.LABEL_COLUMN not in df.columns:
        raise ValueError(f"No label column found in {path}")
    df[schema.LABEL_COLUMN] = df[schema.LABEL_COLUMN].map(normalise_label)

    missing = [c for c in schema.FLOW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} expected feature columns, e.g. {missing[:5]}"
        )
    return df[list(schema.tabular_columns())]


def load_text(path: Path) -> pd.DataFrame:
    """Load the command/payload corpus in canonical form."""
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if schema.TEXT_COLUMN not in df.columns or schema.TEXT_LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"{path} must contain columns '{schema.TEXT_COLUMN}' and '{schema.TEXT_LABEL_COLUMN}'"
        )
    return df[[schema.TEXT_COLUMN, schema.TEXT_LABEL_COLUMN]]
