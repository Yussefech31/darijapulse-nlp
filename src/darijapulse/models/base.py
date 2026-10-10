"""Abstract base classifier defining Moroccan Darija inference protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDarijaClassifier(ABC):
    """Protocol for sentiment classification models on Moroccan Darija text."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, texts: list[str], labels: list[int] | np.ndarray) -> BaseDarijaClassifier:
        """Fit model on raw or normalized Darija texts and binary sentiment labels."""
        pass

    @abstractmethod
    def predict(self, texts: list[str]) -> np.ndarray:
        """Return binary sentiment predictions: 1 (Positive) or 0 (Negative)."""
        pass

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return probability matrix of shape (N, 2): [P(Negative), P(Positive)]."""
        pass

    def get_metadata(self) -> dict[str, Any]:
        """Return configuration and model metadata dictionary."""
        return {"name": self.name, "is_fitted": self.is_fitted}
