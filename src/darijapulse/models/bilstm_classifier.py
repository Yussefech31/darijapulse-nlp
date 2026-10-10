"""PyTorch Bidirectional LSTM model with Attention Pooling for Darija sentiment analysis."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.darijapulse.preprocessing.normalizer import DarijaNormalizer
from src.darijapulse.preprocessing.tokenizer import DarijaTokenizer

from .base import BaseDarijaClassifier

logger = logging.getLogger(__name__)


class BiLSTMNet(nn.Module):
    """Bidirectional LSTM with self-attention pooling for sequence sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Attention pooling weight vector
        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)

        # Attention pooling
        attn_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch_size, hidden_dim * 2)

        logits = self.classifier(context)  # (batch_size, 2)
        return logits


class BiLSTMDarijaClassifier(BaseDarijaClassifier):
    """Deep learning neural classifier trained on normalized word sequences."""

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        max_seq_len: int = 40,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.003,
        device: str = "cpu",
    ):
        super().__init__(name="PyTorch BiLSTM (Attention)")
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device(device)

        self.normalizer = DarijaNormalizer()
        self.tokenizer = DarijaTokenizer(max_vocab_size=6000, max_seq_len=self.max_seq_len)
        self.model: BiLSTMNet | None = None

    def fit(self, texts: list[str], labels: list[int] | np.ndarray) -> BiLSTMDarijaClassifier:
        """Fit tokenizer vocabulary and train BiLSTM neural parameters."""
        norm_texts = [self.normalizer.normalize(t) for t in texts]
        self.tokenizer.fit(norm_texts)

        X_encoded = self.tokenizer.encode_batch(norm_texts)
        y_tensor = torch.tensor(labels, dtype=torch.long)
        X_tensor = torch.tensor(X_encoded, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = BiLSTMNet(
            vocab_size=self.tokenizer.vocab_size(),
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 5 == 0 or epoch == self.epochs:
                logger.info("BiLSTM Epoch %d/%d - Loss: %.4f", epoch, self.epochs, total_loss / len(loader))

        self.is_fitted = True
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Inference probabilities via Softmax over output logits."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict.")

        norm_texts = [self.normalizer.normalize(t) for t in texts]
        X_encoded = self.tokenizer.encode_batch(norm_texts)
        X_tensor = torch.tensor(X_encoded, dtype=torch.long).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict(self, texts: list[str]) -> np.ndarray:
        """Return binary sentiment predictions (argmax of probabilities)."""
        probs = self.predict_proba(texts)
        return np.argmax(probs, axis=1)
