"""Command/payload tokenizer — fit once on TRAIN, saved as a JSON artifact.

Same parity principle as the tabular side: the vocabulary is built from training
data only and frozen into an artifact. At inference we *never* refit — we load
the exact vocab and encode identically, so a token unseen in training maps to
``<unk>`` rather than silently shifting every id (which would corrupt the LSTM's
learned embeddings).

Char-level by default (the documented choice): payloads contain hostile,
malformed, non-dictionary tokens (obfuscated shell, SQLi), so a small character
vocabulary generalises better than word-level and never explodes in size.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from threat_detection.config import TextTokenizerConfig

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

_WORD_RE = re.compile(r"\w+|[^\w\s]")  # words OR single punctuation chars


class TextTokenizer:
    """Deterministic tokenizer with a frozen vocabulary."""

    def __init__(self, config: TextTokenizerConfig):
        self.config = config
        self.token_to_id: dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        self._fitted = False

    # -- tokenization --------------------------------------------------------
    def _split(self, text: str) -> list[str]:
        if self.config.lowercase:
            text = text.lower()
        if self.config.level == "char":
            return list(text)
        return _WORD_RE.findall(text)

    # -- fit -----------------------------------------------------------------
    def fit(self, texts: list[str]) -> TextTokenizer:
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(self._split(str(t)))

        # Deterministic ordering: by descending frequency, then token, so the
        # vocab (and thus every token id) is reproducible across runs.
        candidates = [
            tok
            for tok, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            if freq >= self.config.min_freq
        ]
        limit = self.config.max_vocab - len(self.token_to_id)  # reserve specials
        for tok in candidates[:limit]:
            self.token_to_id[tok] = len(self.token_to_id)
        self._fitted = True
        return self

    # -- encode --------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Encode one string to a fixed-length list of ids (pad/truncate)."""
        if not self._fitted:
            raise RuntimeError("TextTokenizer.encode called before fit/load")
        ids = [self.token_to_id.get(tok, UNK_ID) for tok in self._split(str(text))]
        max_len = self.config.max_len
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [PAD_ID] * (max_len - len(ids))

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.encode(t) for t in texts], dtype=np.int64)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path) -> Path:
        if not self._fitted:
            raise RuntimeError("Refusing to save an unfitted tokenizer")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                {"config": self.config.model_dump(), "token_to_id": self.token_to_id},
                fh,
                ensure_ascii=False,
                indent=2,
            )
        return path

    @classmethod
    def load(cls, path: Path) -> TextTokenizer:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        obj = cls(TextTokenizerConfig.model_validate(payload["config"]))
        obj.token_to_id = {k: int(v) for k, v in payload["token_to_id"].items()}
        obj._fitted = True
        return obj
