"""Unit tests for sentiment classification metrics."""

import numpy as np
import pytest

from src.darijapulse.evaluation.metrics import ClassificationMetrics, compute_sentiment_metrics


def test_compute_sentiment_metrics_perfect():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 1, 0, 0, 1])

    metrics = compute_sentiment_metrics(y_true, y_pred, model_name="Ideal", latency_ms_per_sample=0.5)
    assert isinstance(metrics, ClassificationMetrics)
    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.pos_f1 == 1.0
    assert metrics.neg_f1 == 1.0
    assert metrics.sample_size == 5

    d = metrics.to_dict()
    assert d["accuracy"] == 1.0
    assert d["confusion_matrix"] == [[2, 0], [0, 3]]


def test_sentiment_metrics_imperfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])

    metrics = compute_sentiment_metrics(y_true, y_pred, model_name="Imperfect")
    assert metrics.accuracy == pytest.approx(0.75, abs=1e-3)
    assert 0.0 < metrics.macro_f1 < 1.0
