"""Unit tests for ONNX graph export, INT8 quantization, and inference runtime."""

import numpy as np

from src.darijapulse.models.bilstm_classifier import BiLSTMDarijaClassifier
from src.darijapulse.optimization.onnx_quantizer import DarijaONNXQuantizer, ONNXModelWrapper


def test_onnx_export_and_quantization(tmp_path):
    texts = [
        "سيرفيس واعر وزوين بزاف",
        "تبارك الله خدمة ممتازة",
        "سلعة خايبة وعيانة بزاف",
        "ما عجبنيش الحال تعطل بزاف",
    ]
    labels = [1, 1, 0, 0]

    bilstm = BiLSTMDarijaClassifier(embed_dim=16, hidden_dim=16, num_layers=1, epochs=2, batch_size=2)
    bilstm.fit(texts, labels)

    quantizer = DarijaONNXQuantizer(output_dir=tmp_path)
    fp32_path = quantizer.export_to_onnx(bilstm, model_name="test_fp32.onnx")
    assert fp32_path.exists()
    assert fp32_path.stat().st_size > 0

    int8_path = quantizer.quantize_to_int8(fp32_path, quantized_name="test_int8.onnx")
    assert int8_path.exists()
    assert int8_path.stat().st_size > 0
    assert int8_path.stat().st_size < fp32_path.stat().st_size

    # Test ONNX inference wrapper
    int8_wrapper = ONNXModelWrapper(int8_path)
    encoded = bilstm.tokenizer.encode_batch(["سيرفيس زوين"])
    probs = int8_wrapper.predict_proba(encoded)

    assert probs.shape == (1, 2)
    assert np.allclose(np.sum(probs), 1.0, atol=1e-4)
    assert int8_wrapper.file_size_mb() > 0.0
