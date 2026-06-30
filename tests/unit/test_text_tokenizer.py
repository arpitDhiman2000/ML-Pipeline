"""Unit tests for the command/payload tokenizer."""

from __future__ import annotations

import pytest

from threat_detection.config import TextTokenizerConfig
from threat_detection.features.text import PAD_ID, UNK_ID, TextTokenizer

pytestmark = pytest.mark.unit


@pytest.fixture
def corpus() -> list[str]:
    return [
        "ls -la /home/alice",
        "'; DROP TABLE users; --",
        "cat /etc/passwd",
        "GET /api/v1/orders",
    ]


def test_specials_reserved() -> None:
    tok = TextTokenizer(TextTokenizerConfig(level="char")).fit(["abc"])
    assert tok.token_to_id["<pad>"] == PAD_ID
    assert tok.token_to_id["<unk>"] == UNK_ID


def test_encoding_is_fixed_length(corpus: list[str]) -> None:
    cfg = TextTokenizerConfig(level="char", max_len=32)
    tok = TextTokenizer(cfg).fit(corpus)
    for text in corpus:
        assert len(tok.encode(text)) == 32


def test_truncation_long_input() -> None:
    cfg = TextTokenizerConfig(level="char", max_len=5)
    tok = TextTokenizer(cfg).fit(["abcdefghij"])
    assert len(tok.encode("abcdefghij")) == 5


def test_padding_short_input() -> None:
    cfg = TextTokenizerConfig(level="char", max_len=10)
    tok = TextTokenizer(cfg).fit(["abc"])
    encoded = tok.encode("ab")
    assert encoded[-1] == PAD_ID
    assert len(encoded) == 10


def test_unknown_token_maps_to_unk(corpus: list[str]) -> None:
    cfg = TextTokenizerConfig(level="char", max_len=8, lowercase=True)
    tok = TextTokenizer(cfg).fit(["abc"])
    # 'z' never seen in training -> <unk>
    encoded = tok.encode("z")
    assert encoded[0] == UNK_ID


def test_determinism(corpus: list[str]) -> None:
    cfg = TextTokenizerConfig(level="char")
    a = TextTokenizer(cfg).fit(corpus)
    b = TextTokenizer(cfg).fit(corpus)
    assert a.token_to_id == b.token_to_id


def test_word_level_tokenization() -> None:
    cfg = TextTokenizerConfig(level="word", max_len=10)
    tok = TextTokenizer(cfg).fit(["drop table users", "select from users"])
    assert "users" in tok.token_to_id
    assert "table" in tok.token_to_id


def test_save_load_parity(tmp_path, corpus: list[str]) -> None:
    cfg = TextTokenizerConfig(level="char", max_len=16)
    tok = TextTokenizer(cfg).fit(corpus)
    expected = tok.encode_batch(corpus)

    path = tmp_path / "tokenizer.json"
    tok.save(path)
    reloaded = TextTokenizer.load(path)
    actual = reloaded.encode_batch(corpus)

    assert reloaded.token_to_id == tok.token_to_id
    assert (expected == actual).all()


def test_save_refuses_unfitted(tmp_path) -> None:
    tok = TextTokenizer(TextTokenizerConfig())
    with pytest.raises(RuntimeError, match="unfitted"):
        tok.save(tmp_path / "x.json")
