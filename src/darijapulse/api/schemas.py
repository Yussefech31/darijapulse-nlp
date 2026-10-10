"""Pydantic schemas for DarijaPulse NLP REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: list[str]
    onnx_runtime_available: bool


class NormalizationRequest(BaseModel):
    text: str = Field(..., description="Raw Moroccan Darija text in Arabic script or Arabizi")
    remove_stopwords: bool = Field(False, description="Optionally filter common Darija stopwords")


class NormalizationResponse(BaseModel):
    original_text: str
    is_arabizi: bool
    normalized_text: str


class PredictionRequest(BaseModel):
    text: str = Field(..., description="Moroccan Darija comment or review to analyze")
    model_name: str = Field("int8_onnx", description="Model engine: int8_onnx, fp32_onnx, bilstm, tfidf")


class SentimentPrediction(BaseModel):
    predicted_label: int = Field(..., description="1 for Positive, 0 for Negative")
    sentiment: str = Field(..., description="'Positive' or 'Negative'")
    confidence: float = Field(..., description="Winning class probability (0.0 to 1.0)")
    positive_probability: float
    negative_probability: float
    normalized_text: str
    is_arabizi: bool
    inference_latency_ms: float
    model_used: str


class BenchmarkSummaryResponse(BaseModel):
    dataset_name: str
    test_samples: int
    models: list[dict[str, Any]]
    onnx_quantization_speedup: float
