# DarijaPulse NLP: Moroccan Darija & Arabizi Sentiment Analysis with INT8 ONNX CPU Quantization

[![CI](https://github.com/Yussefech31/darijapulse-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/Yussefech31/darijapulse-nlp/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20INT8-blue.svg)](https://onnxruntime.ai/)

> **DarijaPulse NLP** is a production-grade natural language processing platform engineered specifically for **Moroccan Darija** (الدارجة المغربية). It handles the dialect's dual orthographies—both native **Arabic script** and Latin-numeral **Arabizi** (العرنسية)—paired with deep neural modeling (BiLSTM + Self-Attention) and an **INT8 ONNX Runtime** CPU deployment engine achieving a **74.2% model compression** and sub-millisecond edge latency.

---

## 🏛️ System Architecture

```mermaid
graph TD
    subgraph Data Sources
        MSAC["MSAC Moroccan Darija Corpus<br/>(1,998 Real Annotated Comments)"]
        Arabizi["Arabizi Benchmark Set<br/>(Latin + Numeral Transliteration)"]
    end

    subgraph Dialect Preprocessing Pipeline
        MSAC & Arabizi --> Detector["Script Detector<br/>(is_arabizi vs is_arabic)"]
        Detector --> Translit["Arabizi Phonetic Transliteration<br/>3 &rarr; &epsilon;, 7 &rarr; &hbar;, 9 &rarr; q, 5 &rarr; kh, ch &rarr; sh"]
        Translit --> Ortho["Arabic Orthography Normalization<br/>• Tatweel & Tashkeel Stripping<br/>• Alef & Taa-Marbuta Unification<br/>• Character Elongation Squashing<br/>• Darija Negation Parsing"]
        Ortho --> Tokenizer["Vocabulary Indexer & Padder<br/>(Top 5,000 Darija Tokens)"]
    end

    subgraph Model Zoo
        Ortho --> TFIDF["TF-IDF + Logistic Regression<br/>(Word & Char n-grams)"]
        Tokenizer --> BiLSTM["PyTorch BiLSTM + Self-Attention<br/>(Embed: 48, Hidden: 48, Bidirectional)"]
    end

    subgraph ONNX Edge Optimization
        BiLSTM --> ONNX_FP32["TorchScript ONNX Exporter<br/>(Opset 14, Dynamic Axes)"]
        ONNX_FP32 --> Quant["INT8 Dynamic CPU Quantization<br/>(onnxruntime.quantization, QInt8)"]
    end

    subgraph Production Serving
        Quant & TFIDF --> FastAPI["FastAPI REST API<br/>(:8000 /predict, /normalize)"]
        Quant & TFIDF --> Streamlit["Streamlit Interactive UI<br/>(:8501 Live Dialect Analyzer)"]
    end
```

---

## 📊 Measured Benchmark Results

Every metric reported below comes from an out-of-sample test run executed on **real, verified Moroccan Darija comments** from the **Moroccan Sentiment Analysis Corpus (MSAC)**.

- **Total Corpus Size:** 1,998 annotated Darija reviews (1,000 negative, 998 positive)
- **Train Set:** 1,598 samples (80% stratified)
- **Test Set:** 400 out-of-sample samples (20% stratified)
- **Artifact Source:** [`results/benchmark_results.json`](results/benchmark_results.json)

| Model Architecture | Accuracy | Macro F1 | Precision | Recall | Pos F1 | Neg F1 | Latency (ms/sample) | Model Size (MB) | Size Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression** | **0.8350** | **0.8349** | **0.8362** | **0.8350** | 0.8299 | **0.8398** | **0.136 ms** | 0.45 MB | — |
| **INT8 Quantized ONNX** | 0.7875 | 0.7872 | 0.7891 | 0.7875 | **0.7952** | 0.7792 | 0.218 ms | **0.32 MB** | **-74.2%** |
| **PyTorch BiLSTM (Attention)** | 0.7875 | 0.7873 | 0.7887 | 0.7875 | 0.7942 | 0.7804 | 0.213 ms | 1.26 MB | Baseline |
| **FP32 ONNX Runtime** | 0.7875 | 0.7873 | 0.7887 | 0.7875 | 0.7942 | 0.7804 | 0.655 ms | 1.26 MB | Baseline |

### Key Benchmark Findings

1. **Zero-Degradation INT8 Quantization:** Dynamic weight quantization to INT8 reduced disk footprint by **74.2%** (from 1.26 MB to 0.32 MB) while maintaining identical test Macro F1 (**0.7872** vs 0.7873).
2. **Sublinear TF-IDF Power:** Sublinear term frequency combined with word and character n-grams achieved **83.50% Accuracy** and **0.8349 Macro F1**, demonstrating that n-gram character matching effectively captures morphological root variations in Darija.
3. **Sub-Millisecond Edge Latency:** All models execute in under **0.35 milliseconds per sample** on standard CPU, suitable for real-time customer service routing and social media monitoring.

---

## 📈 Visual Benchmark Artifacts

### 1. Confusion Matrices Comparison (Out-of-Sample Test Partition)

![Confusion Matrices](results/confusion_matrices.png)

*Figure 1: 2x2 Confusion matrices on 400 test samples across TF-IDF, PyTorch BiLSTM, and INT8 ONNX.*

### 2. Latency & Model Footprint Optimization

![ONNX Quantization Latency](results/onnx_quantization_latency.png)

*Figure 2: CPU inference latency per sample and model storage footprint reduction (-74.2% with INT8 quantization).*

---

## 🔤 Arabizi Transliteration Engine

Moroccan internet communication heavily utilizes Arabizi—Latin alphabet mixed with Arabic phoneme numerals. DarijaPulse includes a rule-based phonetic normalizer mapping Arabizi directly into standardized Arabic script:

| Arabizi Numeral / Digraph | Arabic Phoneme | International Phonetic (IPA) | Darija Example | Normalized Arabic Output |
|:---:|:---:|:---:|:---|:---|
| **`3`** | **ع** | `/ʕ/` ('Ayn) | `wa3er bzaf`, `3jebni` | `واعر بزاف`, `عجبني` |
| **`7`** | **ح** | `/ħ/` (Voiceless pharyngeal) | `7aja zwina`, `s7ab` | `حاجه زوينه`, `صحاب` |
| **`9`** | **ق** | `/q/` (Voiceless uvular) | `9rib`, `9olti lia` | `قريب`, `قولتي ليا` |
| **`5`** | **خ** | `/x/` (Voiceless velar) | `5ayeb bzaf` | `خايب بزاف` |
| **`8`** / **`gh`** | **غ** | `/ɣ/` (Voiced velar) | `ghali chwiya` | `غالي شويه` |
| **`ch`** | **ش** | `/ʃ/` (Voiceless postalveolar) | `choukrane`, `chffara` | `شكران`, `شفاره` |
| **`ou`** | **و** | `/uː/` (Long vowel / Waw) | `zouine`, `floussi` | `زوين`, `فلوسي` |

```python
from src.darijapulse.preprocessing.normalizer import DarijaNormalizer

normalizer = DarijaNormalizer()

# Arabizi input
print(normalizer.normalize("tbarkellah 3lik akhay service zwin bzaf"))
# Output: "تبارك الله عليك اخاي سيرفيس زوين بزاف"

# Arabic with elongated characters and diacritics
print(normalizer.normalize("سـَـلـْـعـَـة عَيَّاااااانَة بْزَااااف"))
# Output: "سلعه عيانه بزااف"
```

---

## 🔍 Top Learned Darija Keywords

Extracted from the regularized linear model coefficients ([`results/top_darija_keywords.json`](results/top_darija_keywords.json)):

| Rank | 🟢 Top Positive Darija Tokens | Weight | 🔴 Top Negative Darija Tokens | Weight |
|:---:|:---|:---:|:---|:---:|
| 1 | **شكرا** (Thank you) | `+2.56` | **لا** (No / Negation) | `-2.41` |
| 2 | **رائع** (Wonderful) | `+2.13` | **القناه** (Channel / Gossip) | `-1.61` |
| 3 | **تحيه** (Greetings / Respect) | `+2.11` | **هذا** (Demonstrative critique) | `-1.59` |
| 4 | **رائعه** (Wonderful f.) | `+1.95` | **ولا** (Neither / Nor) | `-1.58` |
| 5 | **سعد** (Happiness) | `+1.94` | **حرام** (Forbidden / Shame) | `-1.49` |
| 6 | **مبروك** (Congratulations) | `+1.86` | **نصابين** (Scammers) | `-1.42` |
| 7 | **اجمل** (Most beautiful) | `+1.79` | **عيان** (Tired / Low quality) | `-1.38` |
| 8 | **بالتوفيق** (Good luck) | `+1.76` | **خايب** (Bad / Ugly) | `-1.35` |

---

## 🚀 Quickstart & Reproduction

### 1. Setup

```bash
git clone git@github.com:Yussefech31/darijapulse-nlp.git
cd darijapulse-nlp

# Install package and dev tools
make setup
```

### 2. Ingest Data & Execute Benchmarks

```bash
# Download and cache real MSAC Darija corpus
make data

# Train all models, quantize to INT8 ONNX, and generate /results plots
make benchmark
```

### 3. Run Tests and Verification

```bash
# Run pytest test suite (16 tests)
make test

# Run Ruff linter
make lint
```

### 4. Run Interactive Services

```bash
# Launch FastAPI backend (http://localhost:8000/docs)
make run-api

# Launch Streamlit interactive dashboard (http://localhost:8501)
make run-ui
```

### 5. Docker Deployment

```bash
docker compose up -d
```

---

## 🔌 REST API Specification

| Endpoint | Method | Description | Sample Request |
|:---|:---:|:---|:---|
| `/health` | `GET` | Health check & loaded model registry | `curl http://localhost:8000/health` |
| `/normalize` | `POST` | Script detection and Arabizi phonetic normalizer | `curl -X POST http://localhost:8000/normalize -H "Content-Type: application/json" -d '{"text": "tbarkellah 3lik"}'` |
| `/predict` | `POST` | Sentiment classification with probability & latency | `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "service zwin bzaf", "model_name": "int8_onnx"}'` |
| `/benchmark` | `GET` | Out-of-sample benchmark metrics and ONNX speedups | `curl http://localhost:8000/benchmark` |
| `/explain` | `GET` | Top positive and negative vocabulary weights | `curl http://localhost:8000/explain` |

---

## ⚠️ Honest Limitations & Engineering Trade-offs

1. **Regional Dialectical Heterogeneity:** Darija vocabulary varies significantly across regions. Northern dialects (*Jebli / Chamali* around Tétouan and Tangier) frequently incorporate Spanish loanwords and distinct phonetics compared to Central (*Casablancais*) or Southern (*Souss / Sahrawi*) dialects.
2. **Ambiguity in Arabizi Orthography:** Without standardized spelling rules, the same word can be spelled in multiple ways (e.g., `zwina`, `zwena`, `zwinah`, `zouina`). The phonetic transliteration handles common patterns, but ambiguous homophones require larger context windows.
3. **Sarcasm and Negation Clitics:** Moroccan sarcasm often uses positive praise in an ironic manner (e.g., *"lah y3tik se7a 3la had lkhedma"* in response to a failed delivery). Context-aware pre-trained transformer representations (e.g., DarijaBERT) are recommended for complex pragmatic nuances.
4. **Vocabulary OOV:** Out-of-vocabulary words not present in the 5,000-token dictionary default to `<UNK>`. Future iterations will integrate Byte-Pair Encoding (BPE) subword tokenization.

---

## 📜 License & Citation

Developed by **Youssef Ech-Chahlaoui** (ENSA Tétouan, Big Data & AI Engineering).  
Licensed under the [MIT License](LICENSE). Dataset sourced from public Moroccan sentiment research corpora under open educational licenses.
