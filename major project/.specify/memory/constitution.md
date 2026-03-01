# Hybrid Malicious URL Detection Model — Project Constitution

## 1. System Objective

Develop a dual-input deep learning architecture for classifying URLs into four categories (**Benign**, **Defacement**, **Phishing**, **Malware**). The system combines automated character-level feature extraction (1D-CNN + Bi-GRU + Attention) with deterministic lexical feature engineering (23 manual heuristics) to maximize classification accuracy and robustness against obfuscation.

- **Data Source**: Kaggle `sid321axn/malicious-urls-dataset` (651,191 records)
- **Framework**: TensorFlow 2.x / Keras Functional API
- **Classification**: 4-class softmax output

---

## Core Principles

### I. Dual-Branch Architecture Integrity (NON-NEGOTIABLE)
The model must maintain two distinct input branches that are only merged at the classification head:
- **Branch A (Deep Learning)**: Raw URL → Character-level Embedding → 1D-CNN → Bi-GRU → Attention → Context Vector
- **Branch B (Heuristics)**: Raw URL → 23 Lexical Features → Dense Network → Feature Vector
- **Classification Head**: Concatenation → Dense(128) → Dropout(0.5) → Softmax(4)
- No cross-contamination between branches before the merge layer
- Each branch must be independently testable and evaluable

### II. Data Pipeline Determinism
All preprocessing must produce identical outputs for identical inputs:
- **Feature Engineering** (`feature_engineering.py`): Deterministic extraction of 23 heuristic features using regex/string parsing
- **Text Processing** (`text_processing.py`): Character-level tokenization with `Keras Tokenizer(char_level=True)`
- `max_sequence_length` based on 95th percentile of URL lengths (not arbitrary)
- `StandardScaler` applied to heuristic features — scaler must be saved for inference
- `pad_sequences` with `padding='post'` — consistent padding direction

### III. Reproducibility (NON-NEGOTIABLE)
Every experiment must be fully reproducible:
- Set random seeds: `numpy`, `tensorflow`, `random`, `os.environ['PYTHONHASHSEED']`
- Lock all dependencies in `requirements.txt` with exact versions
- Save and version train/validation/test splits
- Store all hyperparameters in configuration files (YAML preferred)
- Save model checkpoints with versioned naming

### IV. Modular Code Organization
The codebase must follow strict separation of concerns:
```
major project/src/
├── feature_engineering.py   # 23 lexical feature extractors
├── text_processing.py       # Character tokenizer & padding
├── model_builder.py         # Keras Functional API model definition
├── train.py                 # Training loop, callbacks, logging
├── evaluate.py              # Metrics, classification report, confusion matrix
├── config.yaml              # All hyperparameters & paths
├── data_loader.py           # Dataset loading & splitting
└── utils.py                 # Shared utilities
```
- No hardcoded hyperparameters in source code — all values from `config.yaml`
- PEP 8 compliance, docstrings on all public functions
- Type hints for function signatures

### V. Rigorous Evaluation Against Baseline Research
Evaluation must specifically address weaknesses identified in baseline studies:
- Generate **Classification Report**: Precision, Recall, F1-Score per class
- Generate **Confusion Matrix** using Seaborn — specifically analyze **Phishing vs. Benign misclassification rate**
- Report **multiple metrics**: Accuracy, Precision, Recall, F1 (macro & weighted), AUC-ROC
- Compare against baseline models (single-branch CNN, single-branch heuristic, classical ML)
- Run minimum 3 trials and report mean ± std deviation

### VI. Training Discipline
Strict training protocol to prevent overfitting and ensure convergence:
- **Optimizer**: Adam with `lr=0.001` (handles varying gradient scales between branches)
- **Loss**: `CategoricalCrossentropy` (one-hot encoded labels required)
- **Metrics**: `['accuracy', Precision(), Recall()]`
- **Early Stopping**: `monitor='val_loss'`, `patience=5`, `restore_best_weights=True`
- **ReduceLROnPlateau**: `monitor='val_loss'`, `factor=0.5`, `patience=3`
- No manual learning rate overrides without documented justification
- Training curves (loss + accuracy) must be logged and visualized every run

### VII. GPU & Performance Optimization
Maximize computational efficiency on available hardware:
- CUDA/cuDNN must be verified before training (`tf.config.list_physical_devices('GPU')`)
- Use `tf.data` pipelines or generator-based loading for the 651K dataset
- Batch size tuned to GPU memory (start at 256, adjust based on OOM)
- Profile training time per epoch and document hardware specs
- Mixed precision (`tf.keras.mixed_precision`) when beneficial

---

## 2. Technical Specifications

### 2.1 Feature Engineering (23 Heuristic Features)

| # | Feature | Type | Extraction |
|---|---------|------|------------|
| 1 | `url_length` | int | `len(url)` |
| 2 | `hostname_length` | int | Length of hostname from parsed URL |
| 3 | `path_length` | int | Length of path from parsed URL |
| 4 | `count_dots` | int | Count of `.` characters |
| 5 | `count_hyphens` | int | Count of `-` characters |
| 6 | `count_at` | int | Count of `@` characters |
| 7 | `count_question` | int | Count of `?` characters |
| 8 | `count_ampersand` | int | Count of `&` characters |
| 9 | `count_equals` | int | Count of `=` characters |
| 10 | `count_underscore` | int | Count of `_` characters |
| 11 | `count_tilde` | int | Count of `~` characters |
| 12 | `count_percent` | int | Count of `%` characters |
| 13 | `count_asterisk` | int | Count of `*` characters |
| 14 | `count_colon` | int | Count of `:` characters |
| 15 | `count_www` | int | Count of `www` substrings |
| 16 | `count_https` | int | Count of `https` substrings |
| 17 | `count_http` | int | Count of `http` substrings |
| 18 | `count_digits` | int | Count of digit characters |
| 19 | `count_letters` | int | Count of letter characters |
| 20 | `count_directories` | int | Count of `/` in path |
| 21 | `use_of_ip` | bool→int | Regex match for IP address pattern |
| 22 | `shortening_service` | bool→int | Regex match for known URL shorteners |
| 23 | (reserved) | — | Additional feature as needed |

- All features output as `float32`
- Output scaled with `StandardScaler` — fitted on training set only

### 2.2 Branch A: Sequence Processing

| Layer | Type | Parameters |
|-------|------|------------|
| Input | Input | `shape=(max_sequence_length,)` |
| Embedding | Embedding | `vocab_size → embedding_dim=32` |
| Conv1D | 1D-CNN | `filters=128, kernel_size=3, activation='relu', padding='same'` |
| MaxPool | MaxPooling1D | `pool_size=2` |
| Bi-GRU | Bidirectional GRU | `units=64, return_sequences=True` |
| Attention | Attention | Custom or Keras-native, outputs flattened context vector |

### 2.3 Branch B: Heuristic Processing

| Layer | Type | Parameters |
|-------|------|------------|
| Input | Input | `shape=(23,)` |
| Dense 1 | Dense | `units=64, activation='relu'` |
| Dropout | Dropout | `rate=0.3` |
| Dense 2 | Dense | `units=32, activation='relu'` |

### 2.4 Classification Head

| Layer | Type | Parameters |
|-------|------|------------|
| Merge | Concatenate | Branch A output + Branch B output |
| Dense | Dense | `units=128, activation='relu'` |
| Dropout | Dropout | `rate=0.5` |
| Output | Dense | `units=4, activation='softmax'` |

---

## 3. Development Workflow

### Phase 1: Data & Features
1. Download and explore the Kaggle dataset (651,191 URLs)
2. Implement `feature_engineering.py` — extract & validate all 23 features
3. Implement `text_processing.py` — character tokenizer with 95th percentile padding
4. Analyze class distribution (Benign vs. Defacement vs. Phishing vs. Malware)

### Phase 2: Model Construction
5. Implement `model_builder.py` — dual-branch Keras Functional API model
6. Unit test each branch independently
7. Verify input/output shapes through the full graph

### Phase 3: Training & Tuning
8. Train with callbacks (EarlyStopping, ReduceLROnPlateau)
9. Log training curves (loss, accuracy, precision, recall)
10. Hyperparameter tuning (embedding_dim, GRU units, dropout rates)

### Phase 4: Evaluation & Analysis
11. Classification report with per-class metrics
12. Confusion matrix — focus on Phishing vs. Benign misclassification
13. Compare against baselines (single-branch, classical ML)
14. Statistical significance across multiple runs

---

## 4. Quality Gates

| Gate | Requirement |
|------|-------------|
| Data | No NaNs, no duplicate URLs in splits, class distribution documented |
| Features | All 23 features produce valid float32 values on full dataset |
| Model | Compiles without errors, summary matches SDD architecture exactly |
| Training | EarlyStopping fires or max epochs reached, no NaN losses |
| Evaluation | Classification report + confusion matrix generated for every run |
| Code | PEP 8 compliant, all functions documented, config externalized |

---

## 5. Governance

- This constitution is the **single source of truth** for all architecture and design decisions
- Any deviation from the SDD must be documented with justification and approved
- Amendments increment the version number and update the amendment date
- All code reviews must verify compliance with the architecture and training specifications

**Version**: 1.0.0 | **Ratified**: 2026-03-01 | **Last Amended**: 2026-03-01
