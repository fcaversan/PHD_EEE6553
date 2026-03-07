# Phase 1 — Technical Fact Sheet: Dual-Input Hybrid URL Detection (Character CNN-BiGRU-Attention + Lexical MLP)

---

## 1. Objective

Design and evaluate a dual-input hybrid neural network for multi-class malicious URL detection. The model combines **character-level sequence processing** (Branch A) with **hand-crafted lexical heuristic features** (Branch B) to classify URLs into four categories: benign, defacement, malware, and phishing.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | Kaggle `sid321axn/malicious-urls-dataset` (`malicious_phish.csv`) |
| Raw samples | 651,191 |
| After deduplication + NaN removal | **641,119** |
| Classes | benign, defacement, malware, phishing |
| Input | Raw URL strings (no scheme normalization applied to text) |

**Class distribution:**

| Class | Count | Percentage |
|---|---|---|
| benign | 428,080 | 66.77% |
| defacement | 95,308 | 14.87% |
| phishing | 94,086 | 14.68% |
| malware | 23,645 | 3.69% |

**Stratified split (random_seed=42):**

| Split | Samples | Ratio |
|---|---|---|
| Train | 448,783 | 70% |
| Validation | 96,168 | 15% |
| Test | 96,168 | 15% |

The dataset exhibits class imbalance: benign URLs dominate at ~67%, while malware is the minority class at ~3.7%. Stratified splitting preserves these proportions across all splits.

---

## 3. Preprocessing Pipeline

### 3.1 Branch A — Character-Level Tokenization

URLs are treated as sequences of raw characters — no word segmentation, no semantic parsing.

- **Tokenizer:** Keras `Tokenizer(char_level=True, lower=False, oov_token='<OOV>')`
- Fitted on training URLs only
- **Vocabulary size:** 330 unique characters (including `<OOV>` and padding index)
- **Max sequence length:** 134 (95th percentile of training URL character lengths)
- **Padding:** Post-padded with zeros to `max_sequence_length`
- **Truncation:** Post-truncated if URL exceeds 134 characters
- **Data type:** int32

**URL length statistics (training set):**
- Mean: ~59.8 characters
- 95th percentile: 134 characters

### 3.2 Branch B — 23 Lexical Heuristic Features

Hand-crafted numerical features extracted deterministically from each raw URL string:

**Length features (3):**
1. `url_length` — total character count
2. `hostname_length` — length of the netloc/hostname component
3. `path_length` — length of the path component

**Character count features (11):**
4. `count_dots` — number of `.` characters
5. `count_hyphens` — number of `-` characters
6. `count_at` — number of `@` characters
7. `count_question` — number of `?` characters
8. `count_ampersand` — number of `&` characters
9. `count_equals` — number of `=` characters
10. `count_underscore` — number of `_` characters
11. `count_tilde` — number of `~` characters
12. `count_percent` — number of `%` characters
13. `count_asterisk` — number of `*` characters
14. `count_colon` — number of `:` characters

**Substring count features (3):**
15. `count_www` — occurrences of "www" (case-insensitive)
16. `count_https` — occurrences of "https" (case-insensitive)
17. `count_http` — occurrences of "http" (case-insensitive)

**Character type count features (2):**
18. `count_digits` — total digit characters
19. `count_letters` — total alphabetic characters

**Path feature (1):**
20. `count_directories` — number of `/` in the path component

**Boolean features (2):**
21. `use_of_ip` — binary: 1 if URL contains an IPv4 address pattern (`\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b`)
22. `shortening_service` — binary: 1 if URL matches any of 10 known URL shorteners (bit.ly, goo.gl, tinyurl.com, ow.ly, t.co, short.link, tiny.cc, is.gd, buff.ly, adf.ly)

**Placeholder (1):**
23. `reserved_feature` — always 0 (reserved for future expansion)

**URL parsing note:** URLs lacking a scheme (e.g., `github.com/path`) have `http://` temporarily prepended before `urlparse()` to ensure correct hostname/path splitting. This does not affect the raw URL used for character tokenization.

**Normalization:** `StandardScaler` fitted on training features only, then applied to validation and test sets.

---

## 4. Model Architecture

**Dual-input hybrid neural network** with 2 input tensors:

### Branch A — Character Sequence Processing (CNN-BiGRU-Attention)

| Layer | Configuration | Output Shape |
|---|---|---|
| Input | `(134,)` int32 | `(batch, 134)` |
| Embedding | 330 → 32 dims | `(batch, 134, 32)` |
| Conv1D | 128 filters, kernel=3, ReLU, padding='same' | `(batch, 134, 128)` |
| MaxPooling1D | pool_size=2 | `(batch, 67, 128)` |
| Bidirectional GRU | 64 units, return_sequences=True | `(batch, 67, 128)` |
| Attention | Self-attention `([x, x])` | `(batch, 67, 128)` |
| GlobalAveragePooling1D | — | `(batch, 128)` |

**Branch A rationale:**
- **Embedding (32-dim):** Maps each character index to a learnable dense vector, capturing character-level similarities
- **Conv1D (128 filters, kernel=3):** Extracts local n-gram patterns (e.g., "www", ".com", "http") with receptive field of 3 characters
- **MaxPooling1D (pool=2):** Downsamples temporal dimension by 2×, retaining strongest activations
- **Bidirectional GRU (64 units):** Captures sequential dependencies in both forward and backward directions across the URL
- **Self-Attention:** Allows the model to weight important positions (e.g., domain vs. path segments) differently
- **GlobalAveragePooling1D:** Produces a fixed 128-dimensional vector regardless of input length

### Branch B — Lexical Heuristic MLP

| Layer | Configuration | Output Shape |
|---|---|---|
| Input | `(23,)` float32 | `(batch, 23)` |
| Dense | 64 units, ReLU | `(batch, 64)` |
| Dropout | rate=0.3 | `(batch, 64)` |
| Dense | 32 units, ReLU | `(batch, 32)` |

**Branch B rationale:**
- Simple 2-layer MLP maps the 23 engineered features into a compact 32-dimensional representation
- Dropout (0.3) between layers prevents overfitting on hand-crafted features
- Designed to be lightweight — the bulk of the learning capacity is in Branch A

### Classification Head

| Layer | Configuration | Output Shape |
|---|---|---|
| Concatenate | A(128) + B(32) = **160** | `(batch, 160)` |
| Dense | 128 units, ReLU | `(batch, 128)` |
| Dropout | rate=0.5 | `(batch, 128)` |
| Dense (output) | 4 units, Softmax | `(batch, 4)` |

### Parameter Count

| Category | Count |
|---|---|
| Total parameters | ~221,000 |
| All parameters are trainable | Yes |

---

## 5. Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 (initial) |
| Batch size | 256 |
| Max epochs | 50 |
| Early stopping | patience=5, monitor=val_loss, restore_best_weights=True |
| ReduceLROnPlateau | patience=3, factor=0.5, min_lr=1e-7 |
| Loss function | Categorical Crossentropy |
| Metrics | Accuracy, Precision, Recall |
| Random seed | 42 (canonical), also 123 and 7 for multi-trial evaluation |
| Labels | One-hot encoded (4 classes) |
| Model inputs | `[X_char_sequences, X_lexical_features_scaled]` |

---

## 6. Training History (Canonical Trial — seed=42)

Training converged with early stopping. Best epoch was epoch 10 based on lowest `val_loss`.

- **Best epoch:** 10
- **Final training accuracy:** ~99.0%
- **Best validation accuracy:** ~98.3%
- **Early stopping triggered** after patience=5 epochs without val_loss improvement
- **ReduceLROnPlateau** triggered during later epochs, reducing LR from 0.001 → 0.0005

---

## 7. Test Set Evaluation Results

### 7.1 Canonical Trial (seed=42)

**Test set:** 96,168 samples (15% holdout, stratified)

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| benign | 0.9897 | 0.9912 | 0.9904 | 64,212 |
| defacement | 0.9925 | 0.9981 | 0.9953 | 14,296 |
| malware | 0.9900 | 0.9510 | 0.9703 | 3,547 |
| phishing | 0.9478 | 0.9395 | 0.9436 | 14,113 |

| Metric | Value |
|---|---|
| **Overall Accuracy** | **98.30%** |
| Macro Avg Precision | 0.9800 |
| Macro Avg Recall | 0.9700 |
| Macro Avg F1 | 0.9749 |
| Weighted Avg F1 | 0.9830 |

### 7.2 Multi-Trial Results (3 seeds for statistical robustness)

To assess the stability and reproducibility of results, the full pipeline (data splitting, tokenizer fitting, scaler fitting, model training, and evaluation) was repeated with three different random seeds.

#### Per-Trial Results

| Metric | Seed=42 | Seed=123 | Seed=7 |
|---|---|---|---|
| **Accuracy** | 98.30% | 98.32% | 98.28% |
| Phishing Precision | 0.9478 | 0.9484 | 0.9470 |
| Phishing Recall | 0.9395 | 0.9401 | 0.9390 |
| **Phishing F1** | 0.9436 | 0.9442 | 0.9430 |
| Benign F1 | 0.9904 | 0.9906 | 0.9902 |
| Defacement F1 | 0.9953 | 0.9954 | 0.9951 |
| Malware F1 | 0.9703 | 0.9708 | 0.9698 |

#### Mean ± Std Across 3 Trials

| Metric | Mean ± Std |
|---|---|
| **Accuracy** | **98.30% ± 0.017%** |
| **Phishing F1** | **0.9436 ± 0.0006** |
| Phishing Precision | 0.9477 ± 0.0006 |
| Phishing Recall | 0.9395 ± 0.0005 |
| Benign F1 | 0.9904 ± 0.0002 |
| Defacement F1 | 0.9953 ± 0.0001 |
| Malware F1 | 0.9703 ± 0.0004 |

**Key finding:** The model demonstrates excellent stability across seeds, with accuracy variation of only ±0.017 percentage points and phishing F1 variation of ±0.0006. This confirms that results are not artifacts of a particular random initialization.

---

## 8. Confusion Matrix Analysis (seed=42)

The most critical misclassification pattern is **Phishing → Benign** (false negatives for phishing), as these represent malicious URLs that escape detection.

- Phishing class had the lowest F1-score (0.9436) among all classes
- Approximately 5.5% of phishing URLs were misclassified as benign
- Malware class has the second-lowest F1 (0.9703), partly due to its small sample size (3.69% of data)
- Defacement class achieved near-perfect detection (F1 = 0.9953)

---

## 9. Software & Hardware Environment

| Component | Version / Specification |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.10.1 (last native Windows GPU build) |
| NumPy | 1.23.5 |
| scikit-learn | Standard (for train_test_split, StandardScaler, metrics) |
| Keras | Bundled with TensorFlow 2.10.1 |
| GPU | NVIDIA GeForce RTX 4060 (5,447 MB allocated) |
| CUDA | 11.2 |
| cuDNN | 8.1 |
| OS | Windows 10/11 |

---

## 10. Artifacts Produced

| Artifact | Path | Description |
|---|---|---|
| Trained model | `artifacts/model.keras` | Keras model (best epoch weights) |
| Character tokenizer | `artifacts/tokenizer.json` | Fitted Keras Tokenizer (JSON) |
| Feature scaler | `artifacts/scaler.pkl` | Fitted StandardScaler (joblib) |
| Metadata | `artifacts/metadata.json` | max_sequence_length, vocab_size, label_classes |
| Training log | `artifacts/training_log.txt` | Full training console output |
| Training curves | `artifacts/results/training_curves.png` | Loss + accuracy vs. epoch |
| Classification report | `artifacts/results/trial_*/classification_report.txt` | Per-trial reports |
| Confusion matrix | `artifacts/results/trial_*/confusion_matrix.png` | Per-trial confusion matrices |

---

## 11. Key Design Decisions

1. **Character-level tokenization** (not word-level): URLs don't follow natural language syntax; character n-grams capture domain structure, TLD patterns, and suspicious character sequences more effectively
2. **Dual-input architecture**: Combining learned representations (Branch A) with domain-expert features (Branch B) outperforms either approach alone
3. **95th percentile for max_sequence_length**: Balances coverage (95% of URLs fit without truncation) against computational cost of very long sequences
4. **Self-attention after BiGRU**: Allows positional weighting — the model can learn that certain URL regions (e.g., domain vs. path) are more discriminative for classification
5. **GlobalAveragePooling1D** (not Flatten): Produces translation-invariant features and is more robust to length variations
6. **Stratified splitting**: Preserves class distribution across train/val/test, critical given the 67%/15%/15%/4% imbalance
7. **StandardScaler on training data only**: Prevents data leakage from validation/test sets

---

## 12. Limitations

1. **No semantic understanding**: The model processes individual characters, not words. It cannot understand that "paypal" in `paypal-security-center.com` is a brand name being impersonated
2. **Dataset-bound generalization**: The model learns patterns from the training data distribution; well-crafted phishing URLs that resemble the structural patterns of benign URLs may evade detection
3. **Class imbalance**: Malware class (3.69%) is underrepresented, leading to lower recall (0.9510) compared to majority classes
4. **Static features**: The 23 lexical features are hand-crafted and fixed — no automatic feature discovery for Branch B
5. **No external signals**: No domain reputation, WHOIS data, certificate information, or page content analysis — classification is purely URL-string-based
