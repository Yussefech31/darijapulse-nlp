"""Sublinear TF-IDF + Logistic Regression baseline for Moroccan Darija sentiment."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.darijapulse.preprocessing.normalizer import DarijaNormalizer

from .base import BaseDarijaClassifier


class TfidfDarijaClassifier(BaseDarijaClassifier):
    """Word and character n-gram TF-IDF pipeline paired with regularized Logistic Regression."""

    def __init__(self, c_param: float = 1.5, max_features: int = 8000):
        super().__init__(name="TF-IDF + Logistic Regression")
        self.c_param = c_param
        self.max_features = max_features
        self.normalizer = DarijaNormalizer()

        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                    max_features=self.max_features,
                    analyzer="word",
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=self.c_param,
                    class_weight="balanced",
                    random_state=42,
                    max_iter=500,
                ),
            ),
        ])

    def fit(self, texts: list[str], labels: list[int] | np.ndarray) -> TfidfDarijaClassifier:
        """Normalize texts and fit TF-IDF vectorizer + Logistic Regression classifier."""
        normalized_texts = [self.normalizer.normalize(t) for t in texts]
        y = np.asarray(labels, dtype=np.int64)
        self.pipeline.fit(normalized_texts, y)
        self.is_fitted = True
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict binary sentiment labels (0: Negative, 1: Positive)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
        norm_texts = [self.normalizer.normalize(t) for t in texts]
        return self.pipeline.predict(norm_texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict class probability distribution [P(neg), P(pos)]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba.")
        norm_texts = [self.normalizer.normalize(t) for t in texts]
        return self.pipeline.predict_proba(norm_texts)

    def get_top_features(self, n_top: int = 10) -> dict[str, list[tuple[str, float]]]:
        """Extract highest positive and negative sentiment coefficient keywords."""
        if not self.is_fitted:
            return {"positive": [], "negative": []}

        tfidf: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        clf: LogisticRegression = self.pipeline.named_steps["clf"]

        feature_names = tfidf.get_feature_names_out()
        coefs = clf.coef_[0]

        top_pos_idx = np.argsort(coefs)[-n_top:][::-1]
        top_neg_idx = np.argsort(coefs)[:n_top]

        return {
            "positive": [(feature_names[i], round(float(coefs[i]), 4)) for i in top_pos_idx],
            "negative": [(feature_names[i], round(float(coefs[i]), 4)) for i in top_neg_idx],
        }
