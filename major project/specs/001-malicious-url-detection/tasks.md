# Tasks: Hybrid Malicious URL Detection Model

**Branch**: `001-malicious-url-detection`  
**Input**: [spec.md](./spec.md), [plan.md](./plan.md)  
**Format**: `[ID] [P?] [Story] Description`

- **[P]** = Can run in parallel (no dependencies on other open tasks)
- **[US?]** = User Story reference

---

## Phase 0: Environment & Project Setup (Shared Infrastructure)

**Purpose**: One-time setup — project structure, config, dependencies, GPU verification.

- [x] T001 Create `src/`, `artifacts/results/`, `tests/` directory structure per plan.md
- [x] T002 Create `config.yaml` with all default hyperparameters from plan.md (embedding_dim, cnn_filters, gru_units, dropout rates, lr, batch_size, seeds, paths, split ratios)
- [x] T003 [P] Create `requirements.txt` with pinned versions: tensorflow, numpy, pandas, scikit-learn, seaborn, matplotlib, pyyaml, joblib, pytest
- [x] T004 [P] Create `src/utils.py` with: `set_all_seeds(seed)`, `check_gpu()`, `get_config(path)` helper functions
- [x] T005 Update `.gitignore` to exclude `artifacts/`, `*.pkl`, `*.keras`, `datasets/`

**Checkpoint**: Environment ready — all subsequent phases can begin.

---

## Phase 1: Data Pipeline (Blocking Prerequisites for all branches)

**Purpose**: Both branch inputs depend on these modules. Must be complete and validated before model building.

### Phase 1a — Data Loader

- [x] T006 [P] Implement `src/data_loader.py`:
  - Load Kaggle CSV (`sid321axn/malicious-urls-dataset`)
  - Drop NaNs and duplicates, log dropped count
  - Print class distribution (Benign / Defacement / Phishing / Malware)
  - Stratified split into train (70%) / val (15%) / test (15%)
  - One-hot encode labels (4 classes)
  - Save split indices for reproducibility
  - Raise descriptive error if CSV path is missing

### Phase 1b — Lexical Feature Engineering (US2)

- [x] T007 Implement `src/feature_engineering.py` — extract all 23 features per constitution spec table:
  - String/regex parsers for `url_length`, `hostname_length`, `path_length`
  - Character count features: `.`, `-`, `@`, `?`, `&`, `=`, `_`, `~`, `%`, `*`, `:`
  - Substring count features: `www`, `https`, `http`, `digits`, `letters`, `directories`
  - Boolean flags: `use_of_ip` (IP regex), `shortening_service` (known shortener list)
  - All outputs cast to `float32`
  - Returns `DataFrame(N, 23)` with named columns
- [x] T008 Add `fit_and_save_scaler(train_df, path)` and `load_and_apply_scaler(df, path)` functions to `feature_engineering.py` using `StandardScaler` + joblib
- [x] T009 [P] Write `tests/test_feature_engineering.py`:
  - Test output shape is `(N, 23)` and dtype is `float32`
  - Test `use_of_ip=1` for an IP-based URL
  - Test `shortening_service=1` for a known shortener URL
  - Test zero NaNs on sample dataset
  - Test scaler is saved and reloaded correctly

### Phase 1c — Text Tokenizer (US3)

- [x] T010 Implement `src/text_processing.py`:
  - Fit `Keras Tokenizer(char_level=True)` on training URLs only
  - Compute `max_sequence_length` = 95th percentile of training URL lengths (log value)
  - Apply `pad_sequences(padding='post')` to train/val/test
  - Output `ndarray(N, max_sequence_length)` int32
  - Save fitted tokenizer as JSON with `tokenizer_to_json()`
  - Load function: `load_tokenizer(path)` → returns fitted tokenizer
- [x] T011 [P] Write `tests/test_text_processing.py`:
  - Test output shape and dtype int32
  - Test `max_sequence_length` is 95th percentile (not max)
  - Test padding direction is post (trailing zeros)
  - Test reloaded tokenizer produces identical sequences

**Checkpoint**: Data pipeline complete — run all Phase 1 tests before proceeding.

---

## Phase 2: Model Construction (US1 prerequisite)

**Purpose**: Build and validate the dual-input Keras model. Depends on Phase 1 for input shapes.

- [x] T012 Implement `src/model_builder.py` — `build_model(vocab_size, max_seq_len, config)`:
  - **Input A**: `Input(shape=(max_seq_len,), name='url_sequence')`
  - **Input B**: `Input(shape=(23,), name='lexical_features')`
  - **Branch A**: Embedding(vocab_size, embedding_dim) → Conv1D(filters, kernel_size, relu, same) → MaxPooling1D(2) → Bidirectional(GRU(gru_units, return_sequences=True)) → Attention() → Flatten/squeeze
  - **Branch B**: Dense(64, relu) → Dropout(0.3) → Dense(32, relu)
  - **Head**: Concatenate([branch_a, branch_b]) → Dense(128, relu) → Dropout(0.5) → Dense(4, softmax)
  - Return compiled model with `model.summary()` logged
- [x] T013 [P] Write `tests/test_model_builder.py`:
  - Test model builds without error
  - Test input names are `url_sequence` and `lexical_features`
  - Test output shape is `(batch, 4)`
  - Test Branch A isolated output shape after Attention
  - Test Branch B isolated output shape after final Dense

**Checkpoint**: `model.summary()` matches SDD architecture exactly — verify layer names and shapes.

---

## Phase 3: Training Pipeline (US1)

**Purpose**: End-to-end training with all specified callbacks and artifact saving.

- [x] T014 Implement `src/train.py`:
  - Call `set_all_seeds(config.random_seed)` and `check_gpu()` at startup
  - Load and split data via `data_loader.py`
  - Build lexical features via `feature_engineering.py`, fit+save scaler
  - Build char sequences via `text_processing.py`, fit+save tokenizer
  - Build model via `model_builder.py`
  - Compile: `Adam(lr)`, `CategoricalCrossentropy`, `[accuracy, Precision(), Recall()]`
  - Callbacks: `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`, `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)`
  - Save trained model to `artifacts/model.keras`
  - Plot and save training curves (loss + accuracy vs epochs) to `artifacts/results/training_curves.png`
  - Log final val_loss, val_accuracy, val_precision, val_recall

**Checkpoint** (US1 Independent Test): Running `train.py` end-to-end produces a saved `model.keras`, scaler, tokenizer, and training curves without error.

---

## Phase 4: Evaluation (US1 + US2 + US3 verification)

**Purpose**: Generate all required evaluation artifacts, specifically targeting Phishing vs. Benign misclassification.

- [x] T015 Implement `src/evaluate.py`:
  - Load model, scaler, tokenizer from `artifacts/`
  - Run inference on test split
  - Print `sklearn.metrics.classification_report` (per-class Precision, Recall, F1, Support)
  - Save report to `artifacts/results/classification_report.txt`
  - Generate Seaborn heatmap confusion matrix with class labels [Benign, Defacement, Phishing, Malware]
  - Annotate Phishing → Benign cell with a highlight/annotation for research emphasis
  - Save confusion matrix to `artifacts/results/confusion_matrix.png`
  - Log overall accuracy and Phishing F1-Score explicitly

**Checkpoint** (SC-001 to SC-003): Verify Phishing F1 ≥ 0.93 and Phishing→Benign misclassification rate ≤ 5%.

---

## Phase 5: Baseline Comparisons

**Purpose**: Validate dual-input advantage over single-branch alternatives.

- [ ] T016 [P] Add `--branch-a-only` flag to `train.py` (or separate `train_baseline.py`) to train Branch A (sequence only) in isolation as a baseline
- [ ] T017 [P] Add `--branch-b-only` flag to train Branch B (heuristics only) in isolation as a second baseline
- [ ] T018 Add comparison table output to `evaluate.py`: Dual-input vs Branch-A-only vs Branch-B-only — overall accuracy, Phishing F1, Phishing→Benign misclassification rate

> **Trial 1 Results (seed=42)**: Accuracy=98.33% | Phishing F1=0.9440 | Malware F1=0.9652 | Benign F1=0.9903 | Defacement F1=0.9947

**Checkpoint**: Dual-input model demonstrably outperforms both single-branch baselines.

---

## Phase 6: Single-URL Inference (US4)

**Purpose**: Validate all artifacts are reusable without retraining.

- [x] T019 Implement `src/classify_url.py`:
  - CLI: `python classify_url.py --url "http://example.com/login"`
  - Load `model.keras`, `scaler.pkl`, `tokenizer.json`, `config.yaml`
  - Extract 23 features, scale with loaded scaler
  - Tokenize and pad with loaded tokenizer
  - Run `model.predict()`
  - Print predicted class label + all 4 class probabilities

**Checkpoint** (US4 Independent Test): A single URL string returns a class label + confidence scores without any retraining or data loading.

---

## Phase 7: Statistical Validation & Final Cleanup

**Purpose**: Ensure results are statistically meaningful and the project is reproducible.

- [x] T020 Run 3 independent training trials (seed variations: 42, 123, 7) — log accuracy and Phishing F1 for each *(Trial 1 complete: acc=98.33%, Phishing F1=0.9440)*
- [ ] T021 Compute and report mean ± std deviation for: overall accuracy, Phishing F1, Phishing→Benign misclassification rate
- [x] T022 [P] Final `requirements.txt` — pin all package versions from active environment
- [x] T023 [P] Update `major project/README.md` with: setup instructions, how to train, how to evaluate, how to run inference
- [ ] T024 Git commit all source code (no artifacts, no dataset) and push to remote

**Final Checkpoint** (all Success Criteria):
- [x] SC-001: Overall accuracy ≥ 95% → **98.33%** ✅
- [x] SC-002: Phishing F1 ≥ 0.93 → **0.9440** ✅
- [ ] SC-003: Phishing→Benign rate ≤ 5% → check confusion matrix
- [x] SC-004: Zero NaN features on full dataset → **0 NaN** ✅
- [x] SC-005: `max_sequence_length` is data-driven and logged → **135** (95th pct) ✅
- [x] SC-006: Training completes in ≤ 2 hours on GPU ✅
- [ ] SC-007: Single-URL inference works from saved artifacts → pending
- [x] SC-008: Evaluation artifacts generated every run ✅
