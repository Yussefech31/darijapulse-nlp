"""Unit tests for TF-IDF and BiLSTM sentiment models."""

import numpy as np
import pytest

from src.darijapulse.models.bilstm_classifier import BiLSTMDarijaClassifier
from src.darijapulse.models.tfidf_classifier import TfidfDarijaClassifier


@pytest.fixture
def sample_dataset() -> tuple[list[str], list[int]]:
    texts = [
        "منتوج واعر بزاف عجبني شكرا",
        "تبارك الله عليكم خدمة نقية",
        "سيرفيس طوب ووصل في الوقت",
        "سلعة عيانة بزاف ومكايناش جودة",
        "خسرت فلوسي ما ننصح تا حد يشري",
        "خدمة ناقصة وتعطلو عليا سيمانة",
    ]
    labels = [1, 1, 1, 0, 0, 0]
    return texts, labels


def test_tfidf_classifier(sample_dataset):
    texts, labels = sample_dataset
    model = TfidfDarijaClassifier(max_features=500)
    model.fit(texts, labels)

    preds = model.predict(texts)
    assert len(preds) == len(texts)
    assert set(preds).issubset({0, 1})

    probs = model.predict_proba(texts)
    assert probs.shape == (len(texts), 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    top_feats = model.get_top_features(n_top=3)
    assert "positive" in top_feats
    assert "negative" in top_feats


def test_bilstm_classifier(sample_dataset):
    texts, labels = sample_dataset
    model = BiLSTMDarijaClassifier(embed_dim=16, hidden_dim=16, num_layers=1, epochs=3, batch_size=2)
    model.fit(texts, labels)

    preds = model.predict(texts)
    assert len(preds) == len(texts)
    assert set(preds).issubset({0, 1})

    probs = model.predict_proba(texts)
    assert probs.shape == (len(texts), 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)
