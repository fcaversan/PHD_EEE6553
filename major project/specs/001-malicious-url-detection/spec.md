# Feature Specification: Hybrid Malicious URL Detection Model

**Feature Branch**: `001-malicious-url-detection`  
**Created**: 2026-03-01  
**Status**: Draft  
**Input**: Dual-input deep learning classifier combining character-level sequence processing with deterministic lexical feature engineering for 4-class URL classification.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Train and evaluate the full dual-input model (Priority: P1)

A researcher loads the Kaggle dataset (651,191 URLs), runs the full pipeline end-to-end (feature extraction → tokenization → training → evaluation), and receives a classification report and confusion matrix.

**Why this priority**: This is the core deliverable. Everything else either enables or extends it.

**Independent Test**: Running `train.py` against the dataset produces a saved model file and prints a classification report showing per-class Precision, Recall, and F1-Score for all four classes.

**Acceptance Scenarios**:

1. **Given** the raw Kaggle CSV is present, **When** `train.py` is executed, **Then** the model trains without error, EarlyStopping fires within 20 epochs, and the saved model file `.keras` is created.
2. **Given** a trained model, **When** `evaluate.py` is run on the test split, **Then** a classification report and a Seaborn confusion matrix are generated showing per-class metrics.
3. **Given** the training run, **When** the model loss is monitored, **Then** `ReduceLROnPlateau` reduces the learning rate on plateau and `EarlyStopping` restores best weights.

---

### User Story 2 — Extract and validate 23 lexical heuristic features (Priority: P2)

A researcher runs `feature_engineering.py` against the full URL dataset and receives a validated, scaled `(N, 23)` float32 matrix with no NaN values and correct feature names.

**Why this priority**: Branch B depends entirely on this module; it must be independently verified before model training.

**Independent Test**: `feature_engineering.py` can be run standalone on a sample CSV and outputs a DataFrame of shape `(N, 23)` with documented column names, all float32, zero NaN values.

**Acceptance Scenarios**:

1. **Given** a pandas Series of raw URL strings, **When** the feature extractor is called, **Then** it returns a DataFrame of shape `(N, 23)` with columns matching the 23 defined features.
2. **Given** a URL containing an IP address (e.g., `http://192.168.1.1/login`), **When** extracted, **Then** `use_of_ip = 1`.
3. **Given** a URL using a shortening service (e.g., `http://bit.ly/abc`), **When** extracted, **Then** `shortening_service = 1`.
4. **Given** the full training dataset, **When** `StandardScaler` is fit and applied, **Then** the scaler is saved to disk for reuse at inference time.

---

### User Story 3 — Tokenize raw URLs into padded character sequences (Priority: P2)

A researcher runs `text_processing.py` and receives a `(N, max_sequence_length)` int32 NumPy array with `max_sequence_length` computed from the 95th percentile of URL lengths.

**Why this priority**: Branch A input shape depends on this; `max_sequence_length` must be computed from data, not hardcoded.

**Independent Test**: `text_processing.py` can be run standalone on a sample Series, returning a NumPy array with correct dtype and shape, and saving the fitted tokenizer and `max_sequence_length` for inference reuse.

**Acceptance Scenarios**:

1. **Given** a pandas Series of raw URLs, **When** the tokenizer is fit and applied, **Then** the output array has shape `(N, max_sequence_length)` and dtype `int32`.
2. **Given** the dataset, **When** `max_sequence_length` is computed, **Then** it equals the 95th percentile of URL character lengths (not the maximum).
3. **Given** a URL shorter than `max_sequence_length`, **When** padded, **Then** padding is applied **post** (trailing zeros).
4. **Given** a fitted tokenizer, **When** saved and reloaded, **Then** it produces identical sequences for the same inputs.

---

### User Story 4 — Classify a single new URL at inference time (Priority: P3)

A user provides a single raw URL string and receives a predicted class (Benign / Defacement / Phishing / Malware) with confidence scores, using a previously saved model, scaler, and tokenizer.

**Why this priority**: Validates the full pipeline is re-usable at inference without retraining.

**Independent Test**: A script `classify_url.py` accepts a URL string argument and returns the predicted class label and softmax probability vector.

**Acceptance Scenarios**:

1. **Given** a saved `.keras` model, scaler, and tokenizer, **When** a raw URL is passed to `classify_url.py`, **Then** it returns the predicted class and the four class probability scores.
2. **Given** a known phishing URL, **When** classified, **Then** the model returns `Phishing` as the top class.

---

### Edge Cases

- What happens when a URL is empty or None? → Feature extractor and tokenizer must return zero-vectors / zero-sequence without raising exceptions.
- What happens when a URL exceeds `max_sequence_length`? → Characters beyond the limit are silently truncated (standard Keras truncation behavior with `padding='post'`).
- What happens when `use_of_ip` regex encounters an IPv6 address? → Document expected behavior; flag as known limitation if not supported.
- How does the system handle severe class imbalance? → Document class distribution; consider class weights in `model.fit()` if Benign dominates.
- What if the Kaggle dataset is missing or corrupted? → `data_loader.py` must raise a clear, descriptive error message with the expected file path.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept two inputs simultaneously: raw URL string (character sequence) and 23 lexical float features.
- **FR-002**: `feature_engineering.py` MUST extract exactly 23 features as defined in the SDD and constitution, returning float32 values.
- **FR-003**: `feature_engineering.py` MUST fit a `StandardScaler` on training data only, and save it for inference reuse.
- **FR-004**: `text_processing.py` MUST use `Keras Tokenizer(char_level=True)` and compute `max_sequence_length` from the **95th percentile** of URL lengths.
- **FR-005**: `text_processing.py` MUST apply `pad_sequences` with `padding='post'` and save the fitted tokenizer.
- **FR-006**: Branch A MUST include layers in this exact order: Embedding → Conv1D(128, kernel=3, relu, same) → MaxPooling1D(2) → Bidirectional(GRU(64, return_sequences=True)) → Attention.
- **FR-007**: Branch B MUST include layers in this exact order: Dense(64, relu) → Dropout(0.3) → Dense(32, relu).
- **FR-008**: Merge layer MUST be `Concatenate()` of Branch A attention output and Branch B final dense output.
- **FR-009**: Classification head MUST be: Dense(128, relu) → Dropout(0.5) → Dense(4, softmax).
- **FR-010**: Training MUST use Adam(lr=0.001), `CategoricalCrossentropy` loss, and metrics `[accuracy, Precision(), Recall()]`.
- **FR-011**: Training MUST include both `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)` and `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)`.
- **FR-012**: Evaluation MUST generate a per-class classification report and a Seaborn confusion matrix specifically highlighting Phishing vs. Benign misclassifications.
- **FR-013**: All hyperparameters (embedding_dim, filters, GRU units, dropout rates, batch size, learning rate) MUST be stored in `config.yaml`, not hardcoded.
- **FR-014**: Random seeds MUST be set for `numpy`, `tensorflow`, `random`, and `PYTHONHASHSEED` before any data operations.

### Key Entities

- **URL Record**: Raw string + label (one of: Benign, Defacement, Phishing, Malware).
- **LexicalFeatureMatrix**: `DataFrame(N, 23)` float32, scaled with a saved `StandardScaler`.
- **CharSequenceMatrix**: `ndarray(N, max_sequence_length)` int32, from a saved `Tokenizer`.
- **DualInputModel**: Keras Functional API model with two named inputs (`url_sequence`, `lexical_features`) and one softmax output.
- **TrainingRun**: One complete fit() call, producing training curves, saved model `.keras`, and evaluation artifacts.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The trained dual-input model achieves **≥ 95% overall accuracy** on the held-out test set.
- **SC-002**: **F1-Score for Phishing class ≥ 0.93** — directly addressing the primary weakness identified in baseline research.
- **SC-003**: **Phishing → Benign misclassification rate ≤ 5%** as visible in the confusion matrix.
- **SC-004**: All 23 lexical features are present in the output matrix with **zero NaN values** across the full 651,191-record dataset.
- **SC-005**: `max_sequence_length` is derived from data (95th percentile), not hardcoded; value is logged at runtime.
- **SC-006**: Full training run (including data loading) completes in **≤ 2 hours** on a GPU-equipped machine.
- **SC-007**: Model and all preprocessing artifacts (`.keras`, scaler, tokenizer, config) are saved such that inference on a new URL is possible **without retraining**.
- **SC-008**: Evaluation output includes classification report + confusion matrix for **every training run** without manual intervention.
