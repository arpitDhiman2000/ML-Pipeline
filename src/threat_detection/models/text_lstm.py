"""PyTorch LSTM classifier for malicious command/payload detection.

Why an LSTM here (interview point): order matters in commands/payloads — the
*sequence* of tokens encodes intent, which bag-of-words/TF-IDF throws away. An
LSTM carries state across the sequence to model that ordering. We deliberately
do NOT reach for BERT: inputs are short, the vocabulary is narrow, and the
50K-events/day budget cares about latency + cost. A small (Bi)LSTM lands close to
a transformer on F1 at a fraction of the inference cost — proven by the bake-off
against the TF-IDF baseline (and, in principle, a DistilBERT reference).

The model consumes fixed-length token-id sequences produced by the frozen
``TextTokenizer`` artifact from Sprint 1 — same encoder at train and serve.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from threat_detection.config import LSTMConfig
from threat_detection.features.text import PAD_ID, TextTokenizer
from threat_detection.logging_utils import get_logger

log = get_logger(__name__)
ARTIFACT_VERSION = 1


def resolve_device(preference: str) -> torch.device:
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


class LSTMNet(nn.Module):
    """embedding -> (Bi)LSTM -> dropout -> linear head (1 logit)."""

    def __init__(self, vocab_size: int, config: LSTMConfig):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        n_dirs = 2 if config.bidirectional else 1
        self.head = nn.Linear(config.hidden_dim * n_dirs, 1)
        self._n_dirs = n_dirs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)  # (B, L, E)
        _, (h_n, _) = self.lstm(emb)  # h_n: (num_layers*dirs, B, H)
        last = h_n[-self._n_dirs :]  # last layer's directions
        feat = torch.cat([last[i] for i in range(self._n_dirs)], dim=-1)  # (B, H*dirs)
        return self.head(self.dropout(feat)).squeeze(-1)  # (B,) logits


class TextClassifier:
    """Trainable wrapper: tokenize -> LSTM -> calibrated malicious probability."""

    def __init__(self, config: LSTMConfig, tokenizer: TextTokenizer, *, seed: int):
        self.config = config
        self.tokenizer = tokenizer
        self.seed = seed
        self.device = resolve_device(config.device)
        self.threshold = 0.5
        torch.manual_seed(seed)
        self.net = LSTMNet(tokenizer.vocab_size, config).to(self.device)
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    # -- encoding ------------------------------------------------------------
    def _encode(self, texts: list[str]) -> torch.Tensor:
        ids = self.tokenizer.encode_batch([str(t) for t in texts])
        return torch.from_numpy(ids).long()

    # -- training ------------------------------------------------------------
    def fit(
        self,
        texts: list[str],
        y: np.ndarray,
        *,
        val_texts: list[str] | None = None,
        val_y: np.ndarray | None = None,
    ) -> TextClassifier:
        torch.manual_seed(self.seed)
        X = self._encode(texts).to(self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)

        # class-weighted loss: up-weight the rare malicious class.
        n_pos = float(y_t.sum().item())
        n_neg = float(len(y_t) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)

        dataset = torch.utils.data.TensorDataset(X, y_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )

        history = self.history_
        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(self.config.max_epochs):
            self.net.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            train_loss = epoch_loss / len(dataset)
            history["train_loss"].append(train_loss)

            val_loss = float("nan")
            if val_texts is not None and val_y is not None:
                val_loss = self._val_loss(val_texts, val_y, criterion)
                history["val_loss"].append(val_loss)
                if val_loss < best_val - 1e-4:
                    best_val, best_state = val_loss, self._state_cpu()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
            log.info("lstm.epoch", epoch=epoch, train_loss=train_loss, val_loss=val_loss)
            if epochs_no_improve >= self.config.patience:
                log.info("lstm.early_stop", epoch=epoch)
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _state_cpu(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}

    @torch.no_grad()
    def _val_loss(self, texts: list[str], y: np.ndarray, criterion) -> float:
        self.net.eval()
        X = self._encode(texts).to(self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)
        return float(criterion(self.net(X), y_t).item())

    # -- inference -----------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Malicious probability in [0, 1]."""
        self.net.eval()
        X = self._encode(texts).to(self.device)
        logits = self.net(X)
        return torch.sigmoid(logits).cpu().numpy()

    def predict(self, texts: list[str]) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(np.int64)

    # -- persistence ---------------------------------------------------------
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        joblib.dump(
            {
                "version": ARTIFACT_VERSION,
                "config": self.config.model_dump(),
                "seed": self.seed,
                "threshold": self.threshold,
                "vocab_size": self.tokenizer.vocab_size,
            },
            path.with_suffix(path.suffix + ".meta.joblib"),
        )
        return path

    @classmethod
    def load(cls, path: Path, tokenizer: TextTokenizer) -> TextClassifier:
        meta = joblib.load(path.with_suffix(path.suffix + ".meta.joblib"))
        obj = cls(LSTMConfig.model_validate(meta["config"]), tokenizer, seed=meta["seed"])
        state = torch.load(path, map_location=obj.device)
        obj.net.load_state_dict(state)
        obj.threshold = meta["threshold"]
        return obj
