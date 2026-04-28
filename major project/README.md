# Malicious URL Detection — Multi-Stream Hybrid Deep Learning

A multi-stream hybrid deep learning system for classifying URLs, developed over five iterative phases. The final model combines a CNN-BiGRU-Attention character encoder, a brand-aware lexical MLP, and a novel **Gated Brand Cross-Attention** mechanism, achieving **99.15% binary accuracy** (surpassing Khan et al. 99.08%) and **98.69% four-class accuracy** on the Kaggle benchmark.

## Project Overview

This project implements a malicious URL detection system that bridges character-level deep learning with hand-crafted domain-specific features. Over five phases, the architecture evolved from a dual-input baseline to a three-stream model:

- **Branch A (Sequence)**: Character-level CNN → BiGRU → Self-Attention → GlobalAvgPool (128-d)
- **Branch B (Lexical)**: 27 heuristic features (23 baseline + 4 brand-aware) through a Dense MLP (32-d)
- **Gated Brand Cross-Attention** (Phase 5): Brand features query the BiGRU sequence via cross-attention, gated by a learned sigmoid that silences the stream on non-brand URLs (128-d)
- **Classification Head**: 3-stream concatenation (288-d) → Dense → Sigmoid (binary) or Softmax (4-class)

## Phase Evolution

| Phase | Key Change | 4-Class Acc. | Binary Acc. |
|-------|-----------|-------------|------------|
| **Phase 1** | Baseline dual-input (CNN-BiGRU-Attn + 23-feat MLP) | 98.30% | 98.58%* |
| **Phase 2** | + Frozen DistilBERT (triple-input, 66.6M params) | 98.45% | — |
| **Phase 3** | − DistilBERT, + 4 brand features (27 total) | 98.37% | — |
| **Phase 4** | Data augmentation (641K → 1.39M URLs) | 98.58% | 98.86%* |
| **Phase 5** | + Gated Brand Cross-Attention (3-stream, 140K params) | **98.69%** | **99.15%** |

*\*Binary accuracy from collapsed 4-class predictions, not retrained as binary.*

## Project Structure

```
major project/
├── phase1/                       # Baseline dual-input model
├── phase2/                       # + DistilBERT branch
├── phase3/                       # + Brand-aware features
├── phase4/                       # + Data augmentation (1.39M URLs)
│   └── data_pipeline/            # Augmented dataset preparation
├── phase5/                       # + Gated Brand Cross-Attention (final)
│   ├── src/
│   │   ├── train.py              # Training (--stage 1|2|3)
│   │   ├── evaluate.py           # Evaluation (--stage 1|3)
│   │   ├── model_builder.py      # 3-stream architecture with brand cross-attn
│   │   ├── data_loader.py        # Binary / malicious-only / multiclass modes
│   │   ├── feature_engineering.py # 27 lexical features
│   │   ├── brand_features.py     # Brand impersonation detection features
│   │   ├── stress_test.py        # 16-URL brand impersonation test
│   │   ├── threshold_sweep.py    # Binary decision threshold optimization
│   │   ├── error_analysis.py     # Cross-model error comparison
│   │   ├── text_processing.py    # Character-level tokenization
│   │   └── utils.py              # Shared utilities
│   ├── artifacts/                # Saved models & results (gitignored)
│   ├── config.yaml               # All hyperparameters
│   └── requirements.txt
├── specs/                        # Spec-Kit specifications
├── spec-kit/                     # Spec-Kit framework
├── datasets/                     # Training data (gitignored)
├── PHASE_COMPARISON.md           # Cross-phase results comparison
└── README.md                     # This file
```

## Setup Instructions

### 1. Prerequisites

- **Python 3.10** (required for TensorFlow 2.10 GPU on Windows)
- **CUDA 11.2 + cuDNN 8.1** (for GPU acceleration)
- **NVIDIA GPU** with ≥6GB VRAM (tested on RTX 4060)

### 2. Install Dependencies

```powershell
pip install -r phase5/requirements.txt
```

### 3. Datasets

Two datasets are required:

| Dataset | Path | Size | Source |
|---------|------|------|--------|
| Kaggle `malicious_phish.csv` | `datasets/malicious_phish.csv` | 641K URLs | Kaggle |
| Augmented merged dataset | `phase4/data_pipeline/processed/merged_dataset.csv` | 1.39M URLs | Kaggle + GitHub Phishing.Database + Synthetic |

### 4. Verify GPU

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Usage (Phase 5 — Final Model)

### Training

```powershell
cd "major project/phase5"

# Stage 1: Binary classifier (benign vs malicious) — beats Khan et al.
python src/train.py --stage 1

# Stage 3: Full 4-class classifier (benign, defacement, malware, phishing)
python src/train.py --stage 3

# Stage 2: Malicious sub-classifier (defacement, malware, phishing only)
python src/train.py --stage 2
```

### Evaluation

```powershell
# Binary evaluation on Kaggle test set (comparison with Khan et al. 99.08%)
python src/evaluate.py --stage 1

# 4-class evaluation on Kaggle test set
python src/evaluate.py --stage 3
```

### Stress Test (Brand Impersonation)

```powershell
python src/stress_test.py
```

Tests 16 curated URLs (8 legitimate brand, 8 impersonation phishing) to evaluate brand impersonation detection.

### Threshold Sweep

```powershell
python src/threshold_sweep.py
```

## Final Results (Phase 5, Kaggle Test Set, 96,168 samples)

### Binary Classification (Stage 1)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **99.15%** |
| Khan et al. Benchmark | 99.08% |
| False Positives (benign → malicious) | 274 |
| False Negatives (malicious → benign) | 541 |

### 4-Class Classification (Stage 3)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Benign | 0.9889 | 0.9955 | 0.9922 | 64,212 |
| Defacement | 0.9966 | 0.9977 | 0.9972 | 14,296 |
| Malware | 0.9961 | 0.9462 | 0.9705 | 3,547 |
| Phishing | 0.9651 | 0.9472 | 0.9561 | 14,113 |
| **Overall** | | | | **98.69%** |

- **Phishing→Benign misclassification:** 704/14,113 (4.99%) — satisfies ≤5% safety threshold

### Brand Impersonation Stress Test

| Metric | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|--------|---------|---------|---------|---------|
| Impersonation detected (of 8) | 0 | 0 | 0 | **0** |
| Benign correct (of 8) | 6 | 7 | 7 | 4 |

The zero-day brand impersonation vulnerability persists across all phases — a fundamental limitation of purely string-based URL classification.

## Model Architecture (Phase 5)

```
Input A (168 chars)              Input B (27 features)
       ↓                               ↓
  Embedding(332→32)              Dense(64, ReLU)
       ↓                               ↓
  Conv1D(128, k=3)               Dropout(0.3)
       ↓                               ↓
  MaxPooling(2)                  Dense(32, ReLU) → Branch B (32-d)
       ↓                               ↓
  BiGRU(64) ─── Key/Value ───→  Brand Slice (last 4 features)
       ↓                               ↓           ↓
  Self-Attention                 Query Proj    Sigmoid Gate
       ↓                          (4→128)       (4→128)
  GlobalAvgPool                      ↓              ↓
       ↓                        Cross-Attention  ── × ──→ Gated Brand Ctx (128-d)
  Branch A (128-d)                                  
       ↓                                            ↓
       └──────────── Concatenate (128 + 128 + 32 = 288-d) ──────────┘
                                    ↓
                            Dense(128, ReLU)
                                    ↓
                             Dropout(0.5)
                                    ↓
                    Sigmoid (binary) │ Softmax (4-class)
```

## Configuration (Phase 5)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embedding_dim` | 32 | Character embedding dimension |
| `cnn_filters` | 128 | Conv1D filters in Branch A |
| `gru_units` | 64 | BiGRU units (128-d output bidirectional) |
| `n_brand_features` | 4 | Brand features for cross-attention |
| `learning_rate` | 0.001 | Adam optimizer initial LR |
| `batch_size` | 256 | Training batch size |
| `early_stopping_patience` | 15 | Epochs past best before stopping |
| `reduce_lr_patience` | 5 | Epochs before LR reduction |
| `random_seed` | 42 | Seed for reproducibility |

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| TensorFlow | 2.10.1 |
| NumPy | <2.0 (pinned for TF 2.10) |
| CUDA | 11.2 |
| cuDNN | 8.1 |
| GPU | NVIDIA GeForce RTX 4060 |

## Troubleshooting

**GPU not detected:**
```powershell
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Out of memory:** Reduce `batch_size` in `config.yaml` (try 128 or 64).

**Slow feature extraction:** The Levenshtein distance computation in `brand_features.py` takes ~15–20 minutes on 1M URLs. This is a known bottleneck.

**Lambda layer warning on model load:** The `brand_slice` Lambda layer generates a deserialization warning. Functionally harmless.

## License

This project is developed for academic research purposes (EEE6553 — PhD programme).
