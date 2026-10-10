"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.darijapulse.api.app import app, initialize_service


@pytest.fixture(scope="module")
def client():
    initialize_service()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert "int8_onnx" in data["models_loaded"]
    assert data["onnx_runtime_available"] is True


def test_normalize_endpoint_arabizi(client):
    payload = {"text": "tbarkellah 3lik akhay service top", "remove_stopwords": False}
    response = client.post("/normalize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_arabizi"] is True
    assert "ع" in data["normalized_text"]


def test_normalize_endpoint_arabic(client):
    payload = {"text": "هاد المنتوج واعر بزاف", "remove_stopwords": False}
    response = client.post("/normalize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_arabizi"] is False


def test_predict_int8_onnx(client):
    payload = {"text": "tbarkellah 3lik service zwin bzaf", "model_name": "int8_onnx"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] in [0, 1]
    assert data["sentiment"] in ["Positive", "Negative"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["is_arabizi"] is True
    assert data["inference_latency_ms"] >= 0.0


def test_predict_tfidf(client):
    payload = {"text": "سلعة عيانة وخايبة بزاف", "model_name": "tfidf"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_label"] in [0, 1]


def test_benchmark_endpoint(client):
    response = client.get("/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 3


def test_explain_endpoint(client):
    response = client.get("/explain")
    assert response.status_code == 200
    data = response.json()
    assert "positive" in data
    assert "negative" in data
