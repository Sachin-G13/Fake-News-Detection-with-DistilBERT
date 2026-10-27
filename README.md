# Fake News Detection with DistilBERT

Project developed for the **Serve-Smart AI/ML Hackathon**.

This repository contains a PyTorch natural language processing pipeline that fine-tunes DistilBERT (67M parameters) for title-only fake news detection. The model achieves **98.26% test accuracy** and **0.9987 ROC-AUC** while delivering **~60ms CPU inference** per headline via 64-token truncation.

---

## Key Highlights

| Metric / Parameter | Value | Details |
| :--- | :--- | :--- |
| **Model** | DistilBERT (67M params) | 6 transformer layers, 12 attention heads, 768 hidden dimension |
| **Inference Latency** | ~60ms on CPU | 64-token truncation minimizes attention overhead for real-time inference |
| **Training Pipeline** | 25.6k articles | Leakage-free PyTorch pipeline; purged 4,348 overlapping headlines |
| **Optimization** | FP16 Mixed Precision | `torch.autocast` on GPU with AdamW (`lr=2e-5`) and linear warmup |
| **Test Accuracy** | 98.26% | Evaluated on 8,267 unseen headlines |
| **Test ROC-AUC** | 0.9987 | High discrimination across probability threshold distributions |
| **Precision / Recall / F1** | 97.55% / 98.87% / 0.9820 | Low false positive and false negative error rates |
| **Deployment** | Streamlit App | Interactive UI for single-headline detection and batch CSV/TSV processing |

---

## Data Pipeline and Leakage Prevention

To prevent artificial metric inflation from syndicated wire headlines shared across splits, the pipeline enforces strict deduplication:

1. **Intra-Train Deduplication**: Removed 2,851 duplicate headlines within the training corpus.
2. **Cross-Split Overlap Removal**: Removed 1,497 headlines present in the test set from the training set.
3. **Total Overlaps Purged**: **4,348 overlapping headlines** ($2,851 + 1,497 = 4,348$).
4. **Final Splits**:
   - **Train**: 20,521 articles (from 25,652 leakage-free articles after 80/20 stratified split)
   - **Validation**: 5,131 articles
   - **Test**: 8,267 unseen headlines

---

## Model Architecture and Inference

- **DistilBERT Backbone**: Retains 97% of BERT-base language understanding with 40% fewer parameters (67M vs. 110M) and 60% faster inference.
- **64-Token Truncation**: News titles contain high-density lexical cues. Bounding sequence length to 64 tokens covers complete headlines while reducing $O(L^2)$ attention computations, achieving ~60ms single-core CPU latency.
- **FP16 Mixed Precision**: PyTorch AMP (`torch.autocast` + `GradScaler`) enabled over 2.5x training speedup on GPU without convergence degradation.

---

## Benchmark Results

Evaluation on **8,267 unseen test headlines**:

```text
              precision    recall  f1-score   support

    Fake (0)     0.9894    0.9769    0.9831      4284
    Real (1)     0.9755    0.9887    0.9820      3983

    accuracy                         0.9826      8267
   macro avg     0.9824    0.9828    0.9826      8267
weighted avg     0.9827    0.9826    0.9826      8267
```

- **Accuracy**: 98.26%
- **ROC-AUC**: 0.9987 (probability-based)
- **CPU Latency**: ~60ms per headline (Intel Core i7)

---

## Streamlit Web Application

The Streamlit app (`app.py`) provides:
- **Single Headline Detector**: Instant classification (Real vs. Fake), confidence percentage, latency timer, and WordPiece subword tokenization visualizer.
- **Batch CSV/TSV Processor**: Upload files for bulk classification with dynamic column selection, batching (size 64), progress tracking, and CSV export.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Sachin-G13/Fake-News-Detection-with-DistilBERT.git
cd Fake-News-Detection-with-DistilBERT
pip install -r requirements.txt
```

### 2. Run Web Application

```bash
streamlit run app.py
```

### 3. Training Notebook

Run the complete pipeline notebook:

```bash
jupyter notebook fake_news_detection_distilbert.ipynb
```

---

## Repository Structure

```text
.
├── app.py                             # Streamlit web application
├── fake_news_detection_distilbert.ipynb # Training, deduplication & evaluation notebook
├── add_save_checkpoint.py             # Checkpoint injection script
├── sample_test_headlines.tsv          # Sample test data for batch processing
├── requirements.txt                   # Project dependencies
├── .gitattributes                     # Git LFS configuration (distilbert_weights.pt)
├── .gitignore                         # Build and cache exclusions
├── LICENSE                            # MIT License
└── README.md                          # Project documentation
```

---

## License

This project is licensed under the [MIT License](LICENSE).
