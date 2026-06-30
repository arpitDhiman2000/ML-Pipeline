"""Canonical schema for the CICIDS2017 flow data and the text corpus.

This is the contract every downstream component agrees on. Defining it once,
here, is what lets the synthetic generator, the real-CSV loader, the
preprocessor, and the tests all speak the same language. If a real CICIDS2017
CSV is dropped in, the loader normalises its columns to exactly these names.

The 78 numeric features are the standard CICFlowMeter outputs shipped with
CICIDS2017 (the same set referenced in the project plan).
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
BENIGN_LABEL: Final = "BENIGN"

ATTACK_CLASSES: Final[tuple[str, ...]] = (
    "DoS",
    "DDoS",
    "PortScan",
    "Brute Force",
    "Web Attack",
    "Botnet",
    "Infiltration",
)

ALL_LABELS: Final[tuple[str, ...]] = (BENIGN_LABEL, *ATTACK_CLASSES)

LABEL_COLUMN: Final = "Label"
# Binary target derived from Label (BENIGN -> 0, any attack -> 1). The tabular
# anomaly/binary path uses this; the multi-class path uses Label directly.
BINARY_TARGET_COLUMN: Final = "is_attack"

# ---------------------------------------------------------------------------
# CICFlowMeter numeric feature columns (the canonical CICIDS2017 set).
# ---------------------------------------------------------------------------
FLOW_FEATURES: Final[tuple[str, ...]] = (
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",  # genuine duplicate column shipped in CICIDS2017 CSVs
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
)

assert len(FLOW_FEATURES) == 78, f"expected 78 flow features, got {len(FLOW_FEATURES)}"

# Columns that are *rates* and therefore prone to Inf/NaN in real CICIDS2017
# (division by a near-zero flow duration). The generator injects bad values
# here so the preprocessor has the real-world cleaning job to do.
RATE_FEATURES: Final[tuple[str, ...]] = (
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
)

# A handful of features carry the actual class signal in the synthetic data, so
# the downstream models have something genuinely learnable. (In real data the
# signal is spread across many features; we concentrate it for fast iteration.)
SIGNAL_FEATURES: Final[tuple[str, ...]] = (
    "Flow Duration",
    "Total Fwd Packets",
    "Flow Bytes/s",
    "SYN Flag Count",
    "Destination Port",
    "Packet Length Mean",
)

# ---------------------------------------------------------------------------
# Text corpus schema (command / payload classifier)
# ---------------------------------------------------------------------------
TEXT_COLUMN: Final = "payload"
TEXT_LABEL_COLUMN: Final = "label"  # 0 = benign, 1 = malicious


def tabular_columns() -> list[str]:
    """Full ordered column list for the raw flow table (features + Label)."""
    return [*FLOW_FEATURES, LABEL_COLUMN]
