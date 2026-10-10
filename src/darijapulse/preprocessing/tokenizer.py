"""Word-level and character n-gram tokenizer for Moroccan Darija neural sequences."""

from __future__ import annotations

from collections import Counter

import numpy as np


class DarijaTokenizer:
    """Vocabulary indexer and padding processor for neural sequence models."""

    def __init__(self, max_vocab_size: int = 5000, max_seq_len: int = 40):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.word2idx: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word: dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.is_fitted = False

    def fit(self, texts: list[str]) -> DarijaTokenizer:
        """Build frequency-ranked vocabulary dictionary from raw corpus texts."""
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = text.strip().split()
            counter.update(tokens)

        most_common = counter.most_common(self.max_vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.is_fitted = True
        return self

    def encode(self, text: str) -> list[int]:
        """Convert string to integer sequence capped at max_seq_len with zero-padding."""
        tokens = text.strip().split()
        indices = [self.word2idx.get(t, 1) for t in tokens][: self.max_seq_len]
        if len(indices) < self.max_seq_len:
            indices += [0] * (self.max_seq_len - len(indices))
        return indices

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Batch encode list of texts into 2D NumPy integer array (N, max_seq_len)."""
        return np.array([self.encode(t) for t in texts], dtype=np.int64)

    def vocab_size(self) -> int:
        return len(self.word2idx)
