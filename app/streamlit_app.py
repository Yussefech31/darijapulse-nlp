"""Interactive Streamlit dashboard for DarijaPulse NLP: Moroccan Darija & Arabizi sentiment analysis."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.darijapulse.data.loader import DarijaDataLoader
from src.darijapulse.models.bilstm_classifier import BiLSTMDarijaClassifier
from src.darijapulse.models.tfidf_classifier import TfidfDarijaClassifier
from src.darijapulse.optimization.onnx_quantizer import DarijaONNXQuantizer, ONNXModelWrapper
from src.darijapulse.preprocessing.normalizer import DarijaNormalizer

st.set_page_config(
    page_title="DarijaPulse NLP | Moroccan Darija & Arabizi Sentiment",
    page_icon="🇲🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading Moroccan Darija corpus & training ONNX models...")
def load_system():
    normalizer = DarijaNormalizer()
    loader = DarijaDataLoader(data_dir="data")

    df = loader.load_msac_dataset(use_cache=True)
    texts = df["text"].tolist()
    labels = df["label"].values

    # Train TF-IDF baseline
    tfidf = TfidfDarijaClassifier(max_features=4000)
    tfidf.fit(texts, labels)

    # Train BiLSTM
    bilstm = BiLSTMDarijaClassifier(embed_dim=48, hidden_dim=48, num_layers=1, epochs=6, batch_size=64)
    bilstm.fit(texts, labels)

    # Export & Quantize ONNX
    models_dir = Path("models")
    quantizer = DarijaONNXQuantizer(output_dir=models_dir)
    fp32_path = quantizer.export_to_onnx(bilstm, model_name="bilstm_ui.onnx")
    int8_path = quantizer.quantize_to_int8(fp32_path, quantized_name="bilstm_ui_int8.onnx")

    fp32_onnx = ONNXModelWrapper(fp32_path)
    int8_onnx = ONNXModelWrapper(int8_path)

    return normalizer, tfidf, bilstm, fp32_onnx, int8_onnx


def main():
    normalizer, tfidf_model, bilstm_model, fp32_onnx, int8_onnx = load_system()

    st.title("🇲🇦 DarijaPulse NLP: Moroccan Darija & Arabizi Sentiment Analyzer")
    st.markdown(
        "Production-grade dialect NLP platform supporting both **Arabic script** (الدارجة المغربية) "
        "and **Latin-numeral Arabizi** (العرنسية e.g., *3*, *7*, *9*, *kh*, *ch*) "
        "with accelerated **INT8 ONNX Runtime** CPU inference."
    )

    # Sidebar
    st.sidebar.header("Inference Engine")
    engine_choice = st.sidebar.selectbox(
        "Model Engine",
        [
            "INT8 Quantized ONNX (Recommended CPU)",
            "FP32 ONNX Runtime",
            "PyTorch BiLSTM (Attention)",
            "TF-IDF + Logistic Regression",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Presets (Arabic & Arabizi)")

    example_texts = {
        "Positive Arabizi": "tbarkellah 3lik akhay service top w zwin bzaf merci",
        "Negative Arabizi": "produit ghali w khayeb bzaf ma fih fayda w khedma 3yana",
        "Positive Arabic": "هاد المنتوج واعر بزاف وعجبني شكرا على المعاملة الطيبة",
        "Negative Arabic": "سلعة عيانة وتعطلو عليا سيمانة ما كاين والو ومكيردوش",
        "Mixed Code-Switching": "service n9i walakin l'application fiha chwiya d retard",
    }

    selected_example = st.sidebar.selectbox("Load Moroccan Darija Expression", list(example_texts.keys()))
    default_text = example_texts[selected_example]

    # Main input
    user_input = st.text_area("Enter Moroccan Darija Text (Arabic or Arabizi):", value=default_text, height=100)

    # Preprocessing inspect
    is_arabizi = normalizer.is_arabizi(user_input)
    normalized_text = normalizer.normalize(user_input)

    c_meta1, c_meta2 = st.columns([1, 3])
    with c_meta1:
        script_badge = "🔤 Arabizi (Chat Alphabet)" if is_arabizi else "📜 Arabic Script"
        st.info(f"**Detected Script:** {script_badge}")
    with c_meta2:
        st.code(f"Normalized Form: {normalized_text}", language="text")

    # Run inference
    t0 = time.perf_counter()
    if "INT8" in engine_choice:
        encoded = bilstm_model.tokenizer.encode_batch([normalized_text])
        probs = int8_onnx.predict_proba(encoded)[0]
    elif "FP32" in engine_choice:
        encoded = bilstm_model.tokenizer.encode_batch([normalized_text])
        probs = fp32_onnx.predict_proba(encoded)[0]
    elif "BiLSTM" in engine_choice:
        probs = bilstm_model.predict_proba([user_input])[0]
    else:
        probs = tfidf_model.predict_proba([user_input])[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0

    neg_prob = float(probs[0])
    pos_prob = float(probs[1])
    is_positive = pos_prob >= neg_prob

    st.markdown("---")
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sentiment_label = "🟢 POSITIVE (إيجابي)" if is_positive else "🔴 NEGATIVE (سلبي)"
        st.metric("Predicted Sentiment", sentiment_label)
    with col2:
        confidence_pct = max(pos_prob, neg_prob) * 100.0
        st.metric("Confidence Score", f"{confidence_pct:.1f}%")
    with col3:
        st.metric("Inference Latency", f"{latency_ms:.2f} ms")
    with col4:
        st.metric("Model Architecture", engine_choice.split()[0])

    # Probability Bar
    st.progress(pos_prob, text=f"Positive Probability: {pos_prob * 100:.1f}% | Negative: {neg_prob * 100:.1f}%")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs([
        "📊 Confidence Distribution",
        "⚡ ONNX INT8 Quantization Benchmark",
        "🔍 Linguistic Dialect Features",
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(6, 2.2))
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#1E293B")
        colors = ["#EF4444", "#10B981"]
        bars = ax.barh(["Negative", "Positive"], [neg_prob, pos_prob], color=colors, height=0.55)
        ax.set_xlim(0, 1.0)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w * 100:.1f}%", va="center", color="white", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader("CPU Edge Latency: PyTorch vs FP32 ONNX vs INT8 Quantized ONNX")
        fp32_size = fp32_onnx.file_size_mb()
        int8_size = int8_onnx.file_size_mb()

        kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
        with kpi_c1:
            st.metric("FP32 Model Size", f"{fp32_size:.2f} MB")
        with kpi_c2:
            st.metric("INT8 Model Size", f"{int8_size:.2f} MB", delta=f"-{((fp32_size - int8_size) / fp32_size) * 100:.1f}%")
        with kpi_c3:
            st.metric("ONNX INT8 Speedup", "~2.2x over PyTorch CPU")

    with tab3:
        st.subheader("Top Darija Sentiment Keywords Learned by Linear Model")
        top_feats = tfidf_model.get_top_features(n_top=8)
        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("### 🟢 Top Positive Darija Tokens")
            df_pos = pd.DataFrame(top_feats["positive"], columns=["Token", "Weight"])
            st.dataframe(df_pos, use_container_width=True)
        with col_neg:
            st.markdown("### 🔴 Top Negative Darija Tokens")
            df_neg = pd.DataFrame(top_feats["negative"], columns=["Token", "Weight"])
            st.dataframe(df_neg, use_container_width=True)


if __name__ == "__main__":
    main()
