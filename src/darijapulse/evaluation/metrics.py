"""Evaluation metrics and confusion matrix utilities for Darija sentiment models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    """Quantitative performance container for sentiment classification."""

    model_name: str
    accuracy: float
    macro_f1: float
    precision: float
    recall: float
    pos_f1: float
    neg_f1: float
    sample_size: int
    latency_ms_per_sample: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "pos_f1": round(self.pos_f1, 4),
            "neg_f1": round(self.neg_f1, 4),
            "sample_size": self.sample_size,
            "latency_ms_per_sample": round(self.latency_ms_per_sample, 3),
            "confusion_matrix": self.confusion_matrix,
        }


def compute_sentiment_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    model_name: str = "Classifier",
    latency_ms_per_sample: float = 0.0,
) -> ClassificationMetrics:
    """Compute Accuracy, Macro F1, Class F1s, and 2x2 Confusion Matrix."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(yt, yp))
    macro_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    prec = float(precision_score(yt, yp, average="macro", zero_division=0))
    rec = float(recall_score(yt, yp, average="macro", zero_division=0))

    # Binary per-class F1
    pos_f1 = float(f1_score(yt, yp, pos_label=1, zero_division=0))
    neg_f1 = float(f1_score(yt, yp, pos_label=0, zero_division=0))

    cm = confusion_matrix(yt, yp, labels=[0, 1]).tolist()

    return ClassificationMetrics(
        model_name=model_name,
        accuracy=acc,
        macro_f1=macro_f1,
        precision=prec,
        recall=rec,
        pos_f1=pos_f1,
        neg_f1=neg_f1,
        sample_size=len(yt),
        latency_ms_per_sample=latency_ms_per_sample,
        confusion_matrix=cm,
    )
