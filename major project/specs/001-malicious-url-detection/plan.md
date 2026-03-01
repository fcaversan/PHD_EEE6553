# Implementation Plan: Hybrid Malicious URL Detection Model

**Branch**: `001-malicious-url-detection` | **Date**: 2026-03-01 | **Spec**: [spec.md](./spec.md)  
**Input**: Feature specification from `/specs/001-malicious-url-detection/spec.md`

---

## Summary

Build a dual-input deep learning model using the Keras Functional API that classifies URLs into four categories (Benign, Defacement, Phishing, Malware). Branch A processes raw URL character sequences through Embedding → 1D-CNN → Bi-GRU → Attention; Branch B processes 23 deterministic lexical heuristic features through a dense network. Outputs are merged at a classification head with a 4-class softmax. The primary research goal is to outperform baseline single-branch models with particular focus on reducing Phishing → Benign misclassification.

---

## Technical Context

**Language/Version**: Python 3.10+  
**Primary Dependencies**: TensorFlow 2.x / Keras, NumPy, Pandas, scikit-learn (StandardScaler), Seaborn, Matplotlib, PyYAML, joblib  
**Storage**: Local filesystem — raw CSV dataset, saved `.keras` model, pickled scaler, saved tokenizer JSON, `config.yaml`  
**Testing**: pytest — unit tests for feature extractor, tokenizer, model builder (shape checks)  
**Target Platform**: Windows/Linux workstation with NVIDIA GPU (CUDA 11.x+, cuDNN 8.x+)  
**Project Type**: Research ML pipeline (training + inference scripts)  
**Performance Goals**: ≥ 95% overall accuracy; Phishing F1 ≥ 0.93; full training run ≤ 2 hours on GPU  
**Constraints**: `max_sequence_length` must be data-driven (95th percentile); no hardcoded hyperparameters; scaler fit on training split only  
**Scale/Scope**: 651,191 URL records, 4 classes, single-machine training

---

## Constitution Check

| Gate | Status | Notes |
|------|--------|-------|
| Dual-branch architecture integrity | ✅ | Branches only merge at `Concatenate()` classification head |
| Deterministic preprocessing | ✅ | `max_sequence_length` from 95th percentile; scaler saved for inference |
| No hardcoded hyperparameters | ✅ | All params in `config.yaml` |
| Reproducibility | ✅ | Seeds set for numpy, tensorflow, random, PYTHONHASHSEED |
| Modular code structure | ✅ | One file per responsibility (see Project Structure) |
| Phishing vs Benign evaluation | ✅ | Confusion matrix + classification report mandatory every run |
| GPU check at startup | ✅ | `tf.config.list_physical_devices('GPU')` logged before training |

---

## Project Structure

### Documentation (this feature)

```text
specs/001-malicious-url-detection/
├── spec.md              ✅ Created
├── plan.md              ✅ This file
└── tasks.md             → Created by /speckit.tasks
```

### Source Code (repository root)

```text
src/
├── feature_engineering.py   # 23 lexical heuristic feature extractors + StandardScaler
├── text_processing.py       # Keras char-level tokenizer + 95th-pct padding
├── model_builder.py         # Dual-branch Keras Functional API model definition
├── data_loader.py           # CSV loading, label encoding, train/val/test splitting
├── train.py                 # Training loop: compile, fit, callbacks, save artifacts
├── evaluate.py              # Classification report + Seaborn confusion matrix
├── classify_url.py          # Single-URL inference script (load artifacts + predict)
└── utils.py                 # Shared: seed setting, GPU check, logging helpers

config.yaml                  # All hyperparameters and file paths
requirements.txt             # Pinned dependencies

artifacts/                   # (gitignored) Runtime outputs
├── model.keras
├── scaler.pkl
├── tokenizer.json
└── results/
    ├── classification_report.txt
    └── confusion_matrix.png

tests/
├── test_feature_engineering.py
├── test_text_processing.py
└── test_model_builder.py
```

---

## Phase Breakdown

### Phase 0 — Environment & Data Setup
- Verify GPU availability and TensorFlow installation
- Download Kaggle dataset (`sid321axn/malicious-urls-dataset`, 651,191 records)
- Explore class distribution and URL length statistics
- Determine `max_sequence_length` from 95th percentile of URL lengths
- Create `config.yaml` with all default hyperparameters

### Phase 1 — Data Pipeline Implementation
- Implement `feature_engineering.py`: 23 heuristic extractors, StandardScaler, save scaler
- Implement `text_processing.py`: char-level Keras Tokenizer, 95th-pct padding, save tokenizer
- Implement `data_loader.py`: load CSV, one-hot encode labels, split 70/15/15 train/val/test
- Unit test both pipelines with sample data

### Phase 2 — Model Construction
- Implement `model_builder.py`:
  - Branch A: Embedding(vocab_size, 32) → Conv1D(128, 3, relu, same) → MaxPooling1D(2) → Bidirectional(GRU(64, return_sequences=True)) → Attention → flatten
  - Branch B: Dense(64, relu) → Dropout(0.3) → Dense(32, relu)
  - Head: Concatenate → Dense(128, relu) → Dropout(0.5) → Dense(4, softmax)
- Run model `.summary()` and verify input/output shapes
- Unit test each branch independently for shape correctness

### Phase 3 — Training
- Implement `train.py`:
  - Set all random seeds
  - Check GPU
  - Load data via `data_loader.py`
  - Build preprocessing via Phase 1 modules
  - Build model via `model_builder.py`
  - Compile: Adam(lr=0.001), CategoricalCrossentropy, [accuracy, Precision(), Recall()]
  - Callbacks: EarlyStopping(val_loss, patience=5, restore_best_weights=True), ReduceLROnPlateau(val_loss, factor=0.5, patience=3)
  - Save model, log training curves

### Phase 4 — Evaluation & Baseline Comparison
- Implement `evaluate.py`: classification report, Seaborn confusion matrix
- Run baseline comparisons: single-branch CNN only, single-branch heuristic only
- Analyze Phishing vs. Benign misclassification in confusion matrix
- Run 3 independent trials, report mean ± std for all metrics

### Phase 5 — Inference & Cleanup
- Implement `classify_url.py` for single-URL inference
- Verify full artifact reload (model + scaler + tokenizer)
- Final documentation pass and requirements.txt pin

---

## Key Hyperparameters (config.yaml defaults)

| Parameter | Value | Defined in |
|-----------|-------|------------|
| `embedding_dim` | 32 | Branch A — Embedding |
| `cnn_filters` | 128 | Branch A — Conv1D |
| `cnn_kernel_size` | 3 | Branch A — Conv1D |
| `gru_units` | 64 | Branch A — Bi-GRU |
| `dense_b_units_1` | 64 | Branch B — Dense 1 |
| `dense_b_units_2` | 32 | Branch B — Dense 2 |
| `dropout_b` | 0.3 | Branch B — Dropout |
| `dense_head_units` | 128 | Head — Dense |
| `dropout_head` | 0.5 | Head — Dropout |
| `num_classes` | 4 | Output layer |
| `learning_rate` | 0.001 | Adam optimizer |
| `batch_size` | 256 | Training |
| `early_stopping_patience` | 5 | EarlyStopping |
| `reduce_lr_patience` | 3 | ReduceLROnPlateau |
| `reduce_lr_factor` | 0.5 | ReduceLROnPlateau |
| `random_seed` | 42 | All stochastic ops |
| `test_split` | 0.15 | data_loader |
| `val_split` | 0.15 | data_loader |
