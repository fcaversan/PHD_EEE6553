# Phase 2 — Technical Fact Sheet: Triple-Input Hybrid URL Detection with DistilBERT Semantic Branch

---

## 1. Motivation

Phase 1 achieved 98.30% accuracy with a dual-input model (character CNN-BiGRU-Attention + lexical MLP), but it processed URLs purely at the **character level** — no understanding of word semantics. Phase 2 hypothesized that adding a pretrained language model branch could capture **word-level semantic signals** (e.g., suspicious words like "verify", "security", "recover" in phishing domains), improving phishing detection.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | Kaggle `sid321axn/malicious-urls-dataset` (`malicious_phish.csv`) |
| Raw samples | 651,191 |
| After deduplication | **641,119** |
| Classes | benign, defacement, malware, phishing |

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

---

## 3. Preprocessing Pipeline

### 3.1 Branch A — Character-Level Tokenization

- Each URL treated as a sequence of individual characters
- Keras `Tokenizer(char_level=False, oov_token='<UNK>')` fitted on character lists
- **Vocabulary size:** 271 unique characters
- **Max sequence length:** 135 (95th percentile of URL lengths; mean = 59.8)
- Padding: post-padded with zeros to `max_sequence_length`
- Data type: int32

### 3.2 Branch B — 23 Lexical Heuristic Features

Identical to Phase 1. Hand-crafted numerical features extracted from each URL:

1. `url_length`, `hostname_length`, `path_length`
2. Character counts: dots, hyphens, `@`, `?`, `&`, `=`, `_`, `~`, `%`, `*`, `:`
3. Substring counts: `www`, `https`, `http`
4. `count_digits`, `count_letters`, `count_directories`
5. `use_of_ip` (binary: IP address in URL)
6. `shortening_service` (binary: matches 10 known shorteners — bit.ly, goo.gl, tinyurl.com, etc.)
7. `reserved_feature` (always 0, placeholder)

- **Normalization:** `StandardScaler` fitted on training set, applied to val/test
- Feature means range: 0.00–59.79; feature std range: 0.05–45.00

### 3.3 Branch C — DistilBERT Subword Tokenization (NEW in Phase 2)

**Step 1 — URL-to-words conversion** (`url_to_words()`):
1. Strip scheme (`http://`, `https://`)
2. Split on URL structural delimiters: `. - / _ ? = & % + @ # : , ; ~ !`
3. Filter tokens < 2 characters and purely numeric tokens
4. Lowercase all tokens
5. Join with spaces

Examples:
- `secure-paypal-login.verify.tk/account/update` → `"secure paypal login verify tk account update"`
- `docs.python.org/3/library/urllib.parse.html` → `"docs python org library urllib parse html"`

**Step 2 — BERT WordPiece encoding** (`bert_encode_urls()`):
- Tokenizer: `AutoTokenizer.from_pretrained("distilbert-base-uncased")`
- `max_length`: 64 (with padding and truncation)
- Special tokens: `[CLS]` and `[SEP]` added automatically
- Outputs: `input_ids` (int32) and `attention_mask` (int32), both shape `(N, 64)`

---

## 4. Model Architecture

**Triple-input hybrid neural network** with 4 input tensors:

### Branch A — Character Sequence Processing
| Layer | Configuration | Output Shape |
|---|---|---|
| Input | `(135,)` int32 | `(batch, 135)` |
| Embedding | 271 → 32 dims | `(batch, 135, 32)` |
| Conv1D | 128 filters, kernel=3, ReLU, padding='same' | `(batch, 135, 128)` |
| MaxPooling1D | pool_size=2 | `(batch, 67, 128)` |
| Bidirectional GRU | 64 units, return_sequences=True | `(batch, 67, 128)` |
| Attention | Self-attention `([x, x])` | `(batch, 67, 128)` |
| GlobalAveragePooling1D | — | `(batch, 128)` |

### Branch B — Lexical Heuristics
| Layer | Configuration | Output Shape |
|---|---|---|
| Input | `(23,)` float32 | `(batch, 23)` |
| Dense | 64 units, ReLU | `(batch, 64)` |
| Dropout | rate=0.3 | `(batch, 64)` |
| Dense | 32 units, ReLU | `(batch, 32)` |

### Branch C — Semantic (DistilBERT) — NEW
| Layer | Configuration | Output Shape |
|---|---|---|
| Input (ids) | `(64,)` int32 | `(batch, 64)` |
| Input (mask) | `(64,)` int32 | `(batch, 64)` |
| DistilBERT | `distilbert-base-uncased`, **frozen** | `(batch, 64, 768)` |
| CLS extraction | `[:, 0, :]` | `(batch, 768)` |
| Dense (projection) | 128 units, ReLU | `(batch, 128)` |
| Dropout | rate=0.2 | `(batch, 128)` |

### Classification Head
| Layer | Configuration | Output Shape |
|---|---|---|
| Concatenate | A(128) + B(32) + C(128) = **288** | `(batch, 288)` |
| Dense | 128 units, ReLU | `(batch, 128)` |
| Dropout | rate=0.5 | `(batch, 128)` |
| Dense (output) | 4 units, Softmax | `(batch, 4)` |

### Parameter Count
| Category | Count |
|---|---|
| DistilBERT parameters (frozen) | ~66,362,880 |
| Trainable parameters | ~221,000 |
| Total parameters | ~66,584,000 |

---

## 5. Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 (initial) |
| Batch size | 128 |
| Max epochs | 50 |
| Early stopping | patience=5, monitor=val_loss, restore_best_weights=True |
| ReduceLROnPlateau | patience=3, factor=0.5, min_lr=1e-7 |
| Loss function | Categorical Crossentropy |
| Metrics | Accuracy, Precision, Recall |
| Random seed | 42 |
| Labels | One-hot encoded (4 classes) |
| DistilBERT | Frozen (not fine-tuned) |

---

## 6. Training History (16 Epochs, Early Stopped)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|---|---|---|---|---|---|
| 1 | 0.1918 | 93.61% | 0.1081 | 96.47% | 0.001 |
| 2 | 0.1059 | 96.71% | 0.0803 | 97.38% | 0.001 |
| 3 | 0.0828 | 97.44% | 0.0719 | 97.69% | 0.001 |
| 4 | 0.0709 | 97.79% | 0.0661 | 97.87% | 0.001 |
| 5 | 0.0633 | 98.02% | 0.0609 | 98.09% | 0.001 |
| 6 | 0.0571 | 98.20% | 0.0584 | 98.15% | 0.001 |
| 7 | 0.0527 | 98.35% | 0.0580 | 98.18% | 0.001 |
| 8 | 0.0485 | 98.46% | 0.0551 | 98.29% | 0.001 |
| 9 | 0.0453 | 98.57% | 0.0570 | 98.23% | 0.001 |
| 10 | 0.0423 | 98.64% | 0.0563 | 98.28% | 0.001 |
| **11** | **0.0393** | **98.74%** | **0.0543** | **98.32%** | **0.001** |
| 12 | 0.0371 | 98.81% | 0.0582 | 98.30% | 0.001 |
| 13 | 0.0348 | 98.85% | 0.0593 | 98.24% | 0.001 |
| 14 | 0.0330 | 98.91% | 0.0632 | 98.28% | 0.001 |
| 15 | 0.0252 | 99.18% | 0.0623 | 98.37% | 5e-4 |
| 16 | 0.0220 | 99.28% | 0.0685 | 98.37% | 5e-4 |

- **Best epoch:** 11 (lowest val_loss = 0.0543, val_accuracy = 98.32%)
- **Early stopping triggered at epoch 16** (patience=5 after epoch 11)
- **ReduceLROnPlateau** reduced LR from 0.001 → 0.0005 after epoch 14
- Weights restored from epoch 11

---

## 7. Test Set Evaluation Results

**Test set:** 96,168 samples (15% holdout, stratified)

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| benign | 0.9894 | 0.9923 | 0.9908 | 64,212 |
| defacement | 0.9931 | 0.9979 | 0.9955 | 14,296 |
| malware | 0.9888 | 0.9487 | 0.9683 | 3,547 |
| phishing | 0.9523 | 0.9448 | **0.9485** | 14,113 |

| Metric | Value |
|---|---|
| **Overall Accuracy** | **98.45%** |
| Macro Avg Precision | 0.9809 |
| Macro Avg Recall | 0.9709 |
| Macro Avg F1 | 0.9758 |
| Weighted Avg F1 | 0.9845 |

---

## 8. Phase 1 vs Phase 2 Comparison

| Metric | Phase 1 (seed=42) | Phase 2 (seed=42) | Delta |
|---|---|---|---|
| **Overall Accuracy** | 98.30% | **98.45%** | **+0.15 pp** |
| **Phishing F1** | 0.9436 | **0.9485** | **+0.0049** |
| Phishing Precision | 0.9478 | 0.9523 | +0.0045 |
| Phishing Recall | 0.9395 | 0.9448 | +0.0053 |
| Malware F1 | 0.9703 | 0.9683 | -0.0020 |
| Defacement F1 | 0.9953 | 0.9955 | +0.0002 |
| Benign F1 | 0.9904 | 0.9908 | +0.0004 |
| Architecture | Dual-input (A+B) | Triple-input (A+B+C) | — |
| Concat dimension | 160 | 288 | +128 |
| Batch size | 256 | 128 | Halved (GPU memory) |
| Epochs to best | 10 | 11 | +1 |
| Trainable params | ~221K | ~221K | Comparable |
| Total params | ~221K | ~66.6M | +66.4M (frozen BERT) |

**Key improvement:** Phishing F1 increased from 0.9436 → 0.9485, driven by gains in both precision (+0.0045) and recall (+0.0053). Overall accuracy improved by 0.15 percentage points.

---

## 9. Software & Hardware Environment

| Component | Version / Specification |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.10.1 (last native Windows GPU build) |
| NumPy | 1.23.5 |
| HuggingFace transformers | 4.44.2 (pinned <5.0 for TF support) |
| tokenizers | 0.19.1 |
| scikit-learn | (standard) |
| GPU | NVIDIA GeForce RTX 4060 (5,447 MB allocated) |
| CUDA | 11.2 |
| cuDNN | 8.1 |
| OS | Windows 10/11 |
| Training time per epoch | ~780–810 seconds (~13 min) |
| Total training time | ~16 epochs × ~13 min ≈ **3.5 hours** |

---

## 10. Key Technical Notes

- **DistilBERT was completely frozen** — only the projection Dense(128) layer on top of the CLS token was trainable, keeping trainable parameter count similar to Phase 1
- **URL-to-words preprocessing** was critical: raw URLs were split on structural delimiters to produce meaningful subword tokens for BERT (e.g., `paypal-security.com` → `"paypal security com"`)
- Model loading requires `custom_objects={'TFDistilBertModel': TFDistilBertModel}` due to Keras not recognizing the HuggingFace layer natively
- `PYTHONUTF8=1` environment variable required on Windows to handle Unicode characters in console output

---

## 11. Limitations Observed

An external stress test with 16 curated URLs (8 legitimate, 8 impersonation-phishing) revealed:
- **0/8 phishing impersonation URLs were correctly detected** (all classified as benign with high confidence)
- **2/8 legitimate URLs were false positives** (microsoft.com and github.com incorrectly flagged as phishing)
- The frozen BERT branch did not meaningfully improve detection of well-crafted impersonation domains — likely because (a) the training labels are domain-reputation-based, not semantics-based, and (b) BERT was frozen, preventing phishing-specific fine-tuning
