"""ONNX model exporter and INT8 dynamic CPU quantizer for Moroccan Darija NLP."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.darijapulse.models.bilstm_classifier import BiLSTMDarijaClassifier

logger = logging.getLogger(__name__)


class ONNXModelWrapper:
    """Inference session wrapper executing ONNX models with batching and softmax outputs."""

    def __init__(self, onnx_path: Path | str):
        self.onnx_path = Path(onnx_path)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(self.onnx_path), sess_options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_logits(self, input_ids: np.ndarray) -> np.ndarray:
        """Run ONNX graph inference."""
        inputs = {self.input_name: input_ids.astype(np.int64)}
        outputs = self.session.run([self.output_name], inputs)
        return outputs[0]

    def predict_proba(self, input_ids: np.ndarray) -> np.ndarray:
        """Run inference and apply Softmax."""
        logits = self.predict_logits(input_ids)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def file_size_mb(self) -> float:
        """Return model size on disk in Megabytes."""
        return os.path.getsize(self.onnx_path) / (1024.0 * 1024.0)


class DarijaONNXQuantizer:
    """Manages PyTorch to ONNX graph tracing, validation, and INT8 dynamic quantization."""

    def __init__(self, output_dir: Path | str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_to_onnx(
        self,
        bilstm_clf: BiLSTMDarijaClassifier,
        model_name: str = "darija_bilstm.onnx",
    ) -> Path:
        """Export trained PyTorch BiLSTM neural graph to standard ONNX representation."""
        if not bilstm_clf.is_fitted or bilstm_clf.model is None:
            raise ValueError("BiLSTM model must be trained before export.")

        output_path = self.output_dir / model_name
        bilstm_clf.model.eval()

        # Dummy input for tracing: batch_size=1, seq_len=max_seq_len
        dummy_input = torch.zeros((1, bilstm_clf.max_seq_len), dtype=torch.long, device=bilstm_clf.device)

        torch.onnx.export(
            bilstm_clf.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            dynamo=False,
        )

        # Validate ONNX graph integrity
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("Successfully exported valid ONNX model to %s (%.2f MB)", output_path, os.path.getsize(output_path) / (1024 * 1024))
        return output_path

    def quantize_to_int8(
        self,
        onnx_path: Path | str,
        quantized_name: str = "darija_bilstm_int8.onnx",
    ) -> Path:
        """Apply dynamic INT8 quantization to ONNX graph weights for optimized CPU deployment."""
        input_path = Path(onnx_path)
        quantized_path = self.output_dir / quantized_name

        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(quantized_path),
            weight_type=QuantType.QInt8,
        )

        logger.info(
            "Quantized ONNX model saved to %s (%.2f MB vs original %.2f MB)",
            quantized_path,
            os.path.getsize(quantized_path) / (1024 * 1024),
            os.path.getsize(input_path) / (1024 * 1024),
        )
        return quantized_path

    @staticmethod
    def benchmark_latency(
        bilstm_clf: BiLSTMDarijaClassifier,
        fp32_onnx_path: Path | str,
        int8_onnx_path: Path | str,
        sample_texts: list[str],
        num_runs: int = 50,
    ) -> dict[str, Any]:
        """Benchmark latency across PyTorch CPU, ONNX FP32, and INT8 Quantized ONNX."""
        # 1. PyTorch CPU benchmark
        pytorch_latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = bilstm_clf.predict_proba(sample_texts)
            pytorch_latencies.append((time.perf_counter() - t0) * 1000.0)
        pytorch_median_ms = float(np.median(pytorch_latencies))

        # Encode input texts once for ONNX benchmarks
        norm_texts = [bilstm_clf.normalizer.normalize(t) for t in sample_texts]
        encoded = bilstm_clf.tokenizer.encode_batch(norm_texts)

        # 2. ONNX FP32 benchmark
        fp32_wrapper = ONNXModelWrapper(fp32_onnx_path)
        fp32_latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = fp32_wrapper.predict_proba(encoded)
            fp32_latencies.append((time.perf_counter() - t0) * 1000.0)
        fp32_median_ms = float(np.median(fp32_latencies))

        # 3. ONNX INT8 Quantized benchmark
        int8_wrapper = ONNXModelWrapper(int8_onnx_path)
        int8_latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = int8_wrapper.predict_proba(encoded)
            int8_latencies.append((time.perf_counter() - t0) * 1000.0)
        int8_median_ms = float(np.median(int8_latencies))

        speedup_vs_pytorch = pytorch_median_ms / max(1e-4, int8_median_ms)
        speedup_vs_fp32 = fp32_median_ms / max(1e-4, int8_median_ms)

        fp32_size_mb = fp32_wrapper.file_size_mb()
        int8_size_mb = int8_wrapper.file_size_mb()
        size_reduction_pct = ((fp32_size_mb - int8_size_mb) / max(1e-4, fp32_size_mb)) * 100.0

        return {
            "batch_size": len(sample_texts),
            "num_runs": num_runs,
            "pytorch_cpu_latency_ms": round(pytorch_median_ms, 3),
            "onnx_fp32_latency_ms": round(fp32_median_ms, 3),
            "onnx_int8_latency_ms": round(int8_median_ms, 3),
            "speedup_vs_pytorch": round(speedup_vs_pytorch, 2),
            "speedup_vs_fp32": round(speedup_vs_fp32, 2),
            "model_size_fp32_mb": round(fp32_size_mb, 2),
            "model_size_int8_mb": round(int8_size_mb, 2),
            "size_reduction_pct": round(size_reduction_pct, 1),
        }
