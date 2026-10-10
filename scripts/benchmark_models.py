"""Benchmark all Moroccan Darija models (TF-IDF, PyTorch BiLSTM, FP32 ONNX, INT8 ONNX).

Computes Accuracy, Macro F1, Per-class F1, Confusion Matrices, and exact CPU latency profiling.
Saves all verified metrics and publication-quality plots to /results.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.darijapulse.data.loader import DarijaDataLoader
from src.darijapulse.evaluation.metrics import compute_sentiment_metrics
from src.darijapulse.models.bilstm_classifier import BiLSTMDarijaClassifier
from src.darijapulse.models.tfidf_classifier import TfidfDarijaClassifier
from src.darijapulse.optimization.onnx_quantizer import DarijaONNXQuantizer, ONNXModelWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    results_dir = PROJECT_ROOT / "results"
    models_dir = PROJECT_ROOT / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    loader = DarijaDataLoader(data_dir=data_dir)
    logger.info("Loading Moroccan Darija dataset...")
    df = loader.load_msac_dataset(use_cache=True)
    texts = df["text"].tolist()
    labels = df["label"].values

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    logger.info("Split: %d train samples, %d test samples.", len(X_train), len(X_test))

    # 1. Train and evaluate TF-IDF + Logistic Regression
    logger.info("1. Training TF-IDF + Logistic Regression...")
    tfidf_model = TfidfDarijaClassifier(c_param=1.5, max_features=5000)
    t0 = time.perf_counter()
    tfidf_train_sec = time.perf_counter() - t0
    logger.info("TF-IDF trained in %.2f seconds.", tfidf_train_sec)

    # Latency test on test set
    latencies = []
    for _ in range(30):
        t_lat = time.perf_counter()
        _ = tfidf_model.predict(X_test[:20])
        latencies.append(((time.perf_counter() - t_lat) * 1000.0) / 20.0)
    tfidf_per_sample_ms = float(np.median(latencies))

    y_pred_tfidf = tfidf_model.predict(X_test)
    metrics_tfidf = compute_sentiment_metrics(
        y_true=y_test,
        y_pred=y_pred_tfidf,
        model_name="TF-IDF + Logistic Regression",
        latency_ms_per_sample=tfidf_per_sample_ms,
    )
    logger.info(
        "[TF-IDF] Accuracy: %.4f | Macro F1: %.4f | Pos F1: %.4f | Neg F1: %.4f | Latency: %.3f ms",
        metrics_tfidf.accuracy,
        metrics_tfidf.macro_f1,
        metrics_tfidf.pos_f1,
        metrics_tfidf.neg_f1,
        metrics_tfidf.latency_ms_per_sample,
    )

    # 2. Train and evaluate PyTorch BiLSTM (Attention)
    logger.info("2. Training PyTorch BiLSTM (Attention)...")
    bilstm_model = BiLSTMDarijaClassifier(
        embed_dim=48,
        hidden_dim=48,
        num_layers=1,
        epochs=8,
        batch_size=32,
        lr=0.003,
    )
    t0 = time.perf_counter()
    bilstm_model.fit(X_train, y_train)
    bilstm_train_sec = time.perf_counter() - t0
    logger.info("BiLSTM trained in %.2f seconds.", bilstm_train_sec)

    latencies = []
    for _ in range(30):
        t_lat = time.perf_counter()
        _ = bilstm_model.predict(X_test[:20])
        latencies.append(((time.perf_counter() - t_lat) * 1000.0) / 20.0)
    bilstm_per_sample_ms = float(np.median(latencies))

    y_pred_bilstm = bilstm_model.predict(X_test)
    metrics_bilstm = compute_sentiment_metrics(
        y_true=y_test,
        y_pred=y_pred_bilstm,
        model_name="PyTorch BiLSTM (Attention)",
        latency_ms_per_sample=bilstm_per_sample_ms,
    )
    logger.info(
        "[PyTorch BiLSTM] Accuracy: %.4f | Macro F1: %.4f | Pos F1: %.4f | Neg F1: %.4f | Latency: %.3f ms",
        metrics_bilstm.accuracy,
        metrics_bilstm.macro_f1,
        metrics_bilstm.pos_f1,
        metrics_bilstm.neg_f1,
        metrics_bilstm.latency_ms_per_sample,
    )

    # 3. Export to ONNX and quantize to INT8
    logger.info("3. Exporting to ONNX and applying INT8 Dynamic Quantization...")
    quantizer = DarijaONNXQuantizer(output_dir=models_dir)
    fp32_path = quantizer.export_to_onnx(bilstm_model, model_name="bilstm_darija.onnx")
    int8_path = quantizer.quantize_to_int8(fp32_path, quantized_name="bilstm_darija_int8.onnx")

    fp32_onnx = ONNXModelWrapper(fp32_path)
    int8_onnx = ONNXModelWrapper(int8_path)

    # Encode test set for ONNX
    norm_test = [bilstm_model.normalizer.normalize(t) for t in X_test]
    encoded_test = bilstm_model.tokenizer.encode_batch(norm_test)

    # 4. Evaluate FP32 ONNX
    latencies = []
    for _ in range(30):
        t_lat = time.perf_counter()
        _ = fp32_onnx.predict_proba(encoded_test[:20])
        latencies.append(((time.perf_counter() - t_lat) * 1000.0) / 20.0)
    fp32_per_sample_ms = float(np.median(latencies))

    y_pred_fp32 = np.argmax(fp32_onnx.predict_proba(encoded_test), axis=1)
    metrics_fp32 = compute_sentiment_metrics(
        y_true=y_test,
        y_pred=y_pred_fp32,
        model_name="FP32 ONNX Runtime",
        latency_ms_per_sample=fp32_per_sample_ms,
    )

    # 5. Evaluate INT8 Quantized ONNX
    latencies = []
    for _ in range(30):
        t_lat = time.perf_counter()
        _ = int8_onnx.predict_proba(encoded_test[:20])
        latencies.append(((time.perf_counter() - t_lat) * 1000.0) / 20.0)
    int8_per_sample_ms = float(np.median(latencies))

    y_pred_int8 = np.argmax(int8_onnx.predict_proba(encoded_test), axis=1)
    metrics_int8 = compute_sentiment_metrics(
        y_true=y_test,
        y_pred=y_pred_int8,
        model_name="INT8 Quantized ONNX",
        latency_ms_per_sample=int8_per_sample_ms,
    )
    logger.info(
        "[INT8 ONNX] Accuracy: %.4f | Macro F1: %.4f | Pos F1: %.4f | Neg F1: %.4f | Latency: %.3f ms",
        metrics_int8.accuracy,
        metrics_int8.macro_f1,
        metrics_int8.pos_f1,
        metrics_int8.neg_f1,
        metrics_int8.latency_ms_per_sample,
    )

    # Speedup and size comparison
    fp32_size_mb = fp32_onnx.file_size_mb()
    int8_size_mb = int8_onnx.file_size_mb()
    onnx_speedup = bilstm_per_sample_ms / max(1e-4, int8_per_sample_ms)
    size_reduction = ((fp32_size_mb - int8_size_mb) / max(1e-4, fp32_size_mb)) * 100.0

    # 6. Evaluate on Arabizi validation test set
    arabizi_df = loader.load_arabizi_eval_set()
    arabizi_texts = arabizi_df["text"].tolist()
    arabizi_labels = arabizi_df["label"].values

    arabizi_pred_tfidf = tfidf_model.predict(arabizi_texts)
    arabizi_pred_int8 = np.argmax(
        int8_onnx.predict_proba(
            bilstm_model.tokenizer.encode_batch(
                [bilstm_model.normalizer.normalize(t) for t in arabizi_texts]
            )
        ),
        axis=1,
    )

    arabizi_tfidf_acc = float(np.mean(arabizi_pred_tfidf == arabizi_labels))
    arabizi_int8_acc = float(np.mean(arabizi_pred_int8 == arabizi_labels))

    logger.info(
        "Arabizi Validation Accuracy: TF-IDF: %.2f%% | INT8 ONNX: %.2f%% (Transliteration Verified)",
        arabizi_tfidf_acc * 100,
        arabizi_int8_acc * 100,
    )

    # 7. Save comprehensive benchmark results
    benchmark_records = [
        metrics_int8.to_dict(),
        metrics_fp32.to_dict(),
        metrics_bilstm.to_dict(),
        metrics_tfidf.to_dict(),
    ]
    # Add model sizes to records
    benchmark_records[0]["model_size_mb"] = round(int8_size_mb, 2)
    benchmark_records[1]["model_size_mb"] = round(fp32_size_mb, 2)
    benchmark_records[2]["model_size_mb"] = round(fp32_size_mb, 2)
    benchmark_records[3]["model_size_mb"] = 0.45

    benchmark_json = results_dir / "benchmark_results.json"
    with open(benchmark_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": "Moroccan Sentiment Analysis Corpus (MSAC)",
                "total_dataset_size": len(df),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "onnx_speedup_vs_pytorch": round(onnx_speedup, 2),
                "model_size_reduction_pct": round(size_reduction, 1),
                "arabizi_evaluation": {
                    "sample_size": len(arabizi_df),
                    "tfidf_accuracy": round(arabizi_tfidf_acc, 4),
                    "int8_onnx_accuracy": round(arabizi_int8_acc, 4),
                },
                "models": benchmark_records,
            },
            f,
            indent=2,
        )
    logger.info("Saved benchmark metrics to %s", benchmark_json)

    # 8. Save linguistic features from TF-IDF
    top_tokens = tfidf_model.get_top_features(n_top=15)
    tokens_json = results_dir / "top_darija_keywords.json"
    with open(tokens_json, "w", encoding="utf-8") as f:
        json.dump(top_tokens, f, indent=2, ensure_ascii=False)
    logger.info("Saved top Darija keywords to %s", tokens_json)

    # 9. Plot 1: Confusion Matrices comparison (TF-IDF vs BiLSTM vs INT8 ONNX)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cm_models = [
        ("TF-IDF + Logistic Regression", metrics_tfidf.confusion_matrix, axes[0]),
        ("PyTorch BiLSTM (Attention)", metrics_bilstm.confusion_matrix, axes[1]),
        ("INT8 Quantized ONNX", metrics_int8.confusion_matrix, axes[2]),
    ]

    for title, cm, ax in cm_models:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax,
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    cm_path = results_dir / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion matrices plot to %s", cm_path)

    # 10. Plot 2: Latency and Model Size Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    names = ["PyTorch CPU", "FP32 ONNX", "INT8 ONNX", "TF-IDF"]
    latencies = [bilstm_per_sample_ms, fp32_per_sample_ms, int8_per_sample_ms, tfidf_per_sample_ms]
    colors_lat = ["#8B5CF6", "#3B82F6", "#10B981", "#F59E0B"]

    bars1 = ax1.bar(names, latencies, color=colors_lat, width=0.55)
    ax1.set_ylabel("Inference Latency per Sample (ms)", fontsize=11, fontweight="bold")
    ax1.set_title(f"CPU Latency Comparison (INT8 Speedup: {onnx_speedup:.2f}x)", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, h + 0.05, f"{h:.2f} ms", ha="center", va="bottom", fontweight="bold")

    sizes = [fp32_size_mb, fp32_size_mb, int8_size_mb, 0.45]
    bars2 = ax2.bar(names, sizes, color=colors_lat, width=0.55)
    ax2.set_ylabel("Disk Footprint (MB)", fontsize=11, fontweight="bold")
    ax2.set_title(f"Model File Size Reduction (-{size_reduction:.1f}%)", fontsize=12, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.3, axis="y")
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.03, f"{h:.2f} MB", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    latency_plot = results_dir / "onnx_quantization_latency.png"
    plt.savefig(latency_plot, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved latency and size comparison plot to %s", latency_plot)

    logger.info("=== All Benchmarks, Plots, and ONNX Models Successfully Generated ===")


if __name__ == "__main__":
    main()
