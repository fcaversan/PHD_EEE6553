# Phase 4 — Technical Fact Sheet: Enhanced Data Pipeline with External Phishing + Synthetic Impersonation URLs

---

## 1. Objective

Extend the Phase 3 dual-input hybrid URL detection model by **augmenting the training dataset** with (a) ~732K real-world phishing URLs from the GitHub Phishing.Database, and (b) ~20K synthetically generated brand-impersonation phishing URLs. The goal is to improve phishing recall — particularly for brand-impersonation URLs — while preserving detection accuracy on benign, defacement, and malware classes.

The model architecture is unchanged from Phase 3: **Character CNN-BiGRU-Attention (Branch A) + 27-feature Brand-Aware Lexical MLP (Branch B)**, with a 4-class softmax classification head.

---

## 2. What Changed from Phase 3

| Aspect | Phase 3 | Phase 4 |
|---|---|---|
| Training data | Kaggle only (641,119 URLs) | Kaggle + GitHub Phishing.Database + Synthetic (1,389,956 URLs) |
| Phishing class % | ~14.7% | **60.7%** |
| Data pipeline | Single CSV load | Multi-source merge pipeline |
| New scripts | — | `download_phishing_urls.py`, `generate_synthetic_urls.py`, `merge_datasets.py` |
| Vocab size | 330 | **331** |
| Max sequence length | 134 | **168** (95th percentile of merged data) |
| Architecture | Unchanged | Unchanged |
| Feature count | 27 (23 baseline + 4 brand) | Unchanged |

---

## 3. Dataset

### 3.1 Data Sources

| Source | Label | Count | Percentage | Description |
|---|---|---|---|---|
| Kaggle `malicious_phish.csv` | kaggle | 638,865 | 46.0% | Original multi-class dataset (benign, defacement, malware, phishing) |
| GitHub Phishing.Database | phishing_db | 731,092 | 52.6% | Real-world active phishing URLs (MIT licensed, updated hourly) |
| Synthetic generator | synthetic | 19,999 | 1.4% | Brand-impersonation phishing URLs (7 strategies) |
| **Total** | | **1,389,956** | 100% | |

### 3.2 External Phishing Data Source

| Property | Value |
|---|---|
| Primary source | GitHub Phishing.Database (MIT license) |
| URL | `https://raw.githubusercontent.com/Phishing-Database/Phishing.Database/master/phishing-links-ACTIVE.txt` |
| Coverage | ~780K+ verified active phishing links |
| Update frequency | Every few hours |
| Source tag in CSV | `phishing_database` |
| Secondary source | OpenPhish Community Feed (`https://openphish.com/feed.txt`) |
| Secondary coverage | ~500 recently-detected (15-min updates) |
| Download tool | `data_pipeline/download_phishing_urls.py` |
| Output file | `data_pipeline/processed/external_phishing.csv` |

### 3.3 Synthetic Phishing Generation

20,000 brand-impersonation phishing URLs generated via 7 weighted strategies:

| Strategy | Weight | Description | Example |
|---|---|---|---|
| Subdomain abuse | 25% | Brand as subdomain of unrelated domain | `paypal.com.secure-login.xyz/verify` |
| Typosquatting | 20% | Misspelled brand via 5 mutations (swap, double, drop, insert, replace) | `paypla.com/login` |
| Homoglyph substitution | 10% | Visually similar character replacement (a→@, e→3, o→0, l→1, s→$, g→9, t→7) | `g00gle.com/auth` |
| Hyphenation | 20% | Trust-word hyphens around brand name | `paypal-secure-login.xyz` |
| Path injection | 10% | Brand in URL path only, not domain | `evil-site.xyz/paypal/login` |
| IP-based | 5% | IP address with brand in path | `http://192.168.1.100:8080/paypal/secure` |
| Long subdomain chain | 10% | Many subdomains to obscure real domain | `paypal.com.secure.verify.account.evil.xyz/login` |

Generation tool: `data_pipeline/generate_synthetic_urls.py`

### 3.4 Merged Class Distribution

| Class | Count | Percentage |
|---|---|---|
| phishing | 843,513 | 60.7% |
| benign | 427,341 | 30.7% |
| defacement | 95,456 | 6.9% |
| malware | 23,646 | 1.7% |

### 3.5 Merge Pipeline

- **Deduplication**: Case-insensitive URL match with trailing `/` stripped; first occurrence kept
- **Shuffle**: Deterministic with seed=42
- **Output**: `data_pipeline/processed/merged_dataset.csv` (columns: `url`, `type`, `source`)
- **CLI**: `python merge_datasets.py [--kaggle-only] [--external <path>] [--synthetic-count N] [--seed 42]`

### 3.6 Stratified Split (random_seed=42)

| Split | Samples | Ratio |
|---|---|---|
| Train | 972,968 | 70% |
| Validation | 208,494 | 15% |
| Test | 208,494 | 15% |

---

## 4. Preprocessing Pipeline

### 4.1 Branch A — Character-Level Tokenization

| Property | Value |
|---|---|
| Tokenizer | Keras `Tokenizer(char_level=True, lower=False, oov_token='<OOV>')` |
| Fitted on | Training URLs only |
| Vocabulary size | 331 unique characters (including `<OOV>` and padding index 0) |
| Max sequence length | 168 (95th percentile of training URL character lengths) |
| Padding | Post-padded with zeros |
| Truncation | Post-truncated if URL exceeds 168 characters |
| Data type | int32 |

### 4.2 Branch B — 27 Lexical Heuristic Features

**23 baseline features** (unchanged from Phase 1):

| # | Feature | Description |
|---|---|---|
| 1 | `url_length` | Total character count |
| 2 | `hostname_length` | Length of hostname component |
| 3 | `path_length` | Length of path component |
| 4 | `count_dots` | Number of `.` characters |
| 5 | `count_hyphens` | Number of `-` characters |
| 6 | `count_at` | Number of `@` characters |
| 7 | `count_question` | Number of `?` characters |
| 8 | `count_ampersand` | Number of `&` characters |
| 9 | `count_equals` | Number of `=` characters |
| 10 | `count_underscore` | Number of `_` characters |
| 11 | `count_tilde` | Number of `~` characters |
| 12 | `count_percent` | Number of `%` characters |
| 13 | `count_asterisk` | Number of `*` characters |
| 14 | `count_colon` | Number of `:` characters |
| 15 | `count_www` | Occurrences of "www" (case-insensitive) |
| 16 | `count_https` | Occurrences of "https" (case-insensitive) |
| 17 | `count_http` | Occurrences of "http" (case-insensitive) |
| 18 | `count_digits` | Total digit (0–9) characters |
| 19 | `count_letters` | Total alphabetic characters |
| 20 | `count_directories` | Number of `/` in path |
| 21 | `use_of_ip` | Binary: IPv4 pattern detected |
| 22 | `shortening_service` | Binary: URL shortener detected |
| 23 | `reserved_feature` | Always 0 (placeholder for compatibility) |

**4 brand-aware impersonation features** (introduced in Phase 3, retained here):

| # | Feature | Description |
|---|---|---|
| 24 | `brand_in_domain` | 1.0 if hostname contains a known brand but registered domain ≠ official domain |
| 25 | `brand_count` | Count of distinct brand tokens appearing in full URL |
| 26 | `trust_word_in_domain` | 1.0 if hostname contains a trust-word token (e.g., "secure", "verify", "login") |
| 27 | `min_brand_edit_distance` | Normalized Levenshtein distance between registered domain and closest known brand (lower = more brand-like) |

**Brand dictionary**: 48 brands monitored (Google, PayPal, Apple, Microsoft, Amazon, Netflix, Facebook, Instagram, Twitter, LinkedIn, Dropbox, Adobe, GitHub, Yahoo, Outlook, Office, Chase, Wells Fargo, Citibank, Bank of America, HSBC, Barclays, NatWest, Steam, Spotify, Discord, Twitch, eBay, Alibaba, Walmart, Target, FedEx, UPS, DHL, USPS, IRS, NHS, Vodafone, AT&T, Verizon, Samsung, Huawei, NVIDIA, Intel, Cisco, Oracle, Salesforce, Zoom, DocuSign, Coinbase, Binance, Blockchain, MetaMask)

**Trust words**: 16 tokens — `secure`, `verify`, `login`, `account`, `update`, `auth`, `signin`, `confirm`, `recover`, `support`, `password`, `billing`, `payment`, `validate`, `credential`, `alert`, `urgent`

**URL shorteners detected**: `bit.ly`, `goo.gl`, `tinyurl.com`, `ow.ly`, `t.co`, `short.link`, `tiny.cc`, `is.gd`, `buff.ly`, `adf.ly`

**Normalization**: `StandardScaler` fitted on training features only, then applied to validation and test sets.

---

## 5. Model Architecture

**Unchanged from Phase 3.** Dual-input hybrid neural network with 2 input tensors.

### Branch A — Character Sequence Processing (CNN-BiGRU-Attention)

| Layer | Configuration | Output Shape |
|---|---|---|
| Input | `(168,)` int32 | `(batch, 168)` |
| Embedding | 331 → 32 dims | `(batch, 168, 32)` |
| Conv1D | 128 filters, kernel=3, ReLU, padding='same' | `(batch, 168, 128)` |
| MaxPooling1D | pool_size=2 | `(batch, 84, 128)` |
| Bidirectional GRU | 64 units, return_sequences=True | `(batch, 84, 128)` |
| Attention | Self-attention `([x, x])` | `(batch, 84, 128)` |
| GlobalAveragePooling1D | — | `(batch, 128)` |

### Branch B — Lexical Heuristic MLP

| Layer | Configuration | Output Shape |
|---|---|---|
| Input | `(27,)` float32 | `(batch, 27)` |
| Dense | 64 units, ReLU | `(batch, 64)` |
| Dropout | rate=0.3 | `(batch, 64)` |
| Dense | 32 units, ReLU | `(batch, 32)` |

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
| Total parameters | ~122,500 |
| All parameters are trainable | Yes |

---

## 6. Training Configuration

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
| Random seed | 42 |
| Labels | One-hot encoded (4 classes) |
| Model inputs | `[X_char_sequences, X_lexical_features_scaled]` |

---

## 7. Training History

Training ran for 13 epochs with early stopping. Best epoch was epoch 8 based on lowest `val_loss`.

- **Best epoch:** 8 (of 13)
- **Best validation accuracy:** ~99.11%
- **Early stopping triggered** after 5 additional epochs without val_loss improvement
- **GPU**: NVIDIA RTX 4060 (5,447 MB allocated)

---

## 8. Test Set Evaluation Results

### 8.1 Augmented (Merged) Test Set

**Test set:** 208,494 samples (15% holdout, stratified from merged dataset — 60.7% phishing)

#### Classification Report (Augmented Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| benign | 0.9854 | 0.9936 | 0.9895 | 64,101 |
| defacement | 0.9902 | 0.9914 | 0.9908 | 14,290 |
| malware | 0.9895 | 0.9270 | 0.9572 | 3,546 |
| phishing | 0.9941 | 0.9916 | 0.9928 | 126,557 |

| Metric | Value |
|---|---|
| **Overall Accuracy (Augmented)** | **99.11%** |
| Macro Avg Precision | 0.9898 |
| Macro Avg Recall | 0.9759 |
| Macro Avg F1 | 0.9826 |
| Weighted Avg F1 | 0.9911 |

### 8.2 Kaggle-Only Test Set Evaluation (Apples-to-Apples Cross-Phase Comparison)

To enable a fair comparison with Phases 1–3, the Phase 4 trained model was also evaluated on the **original Kaggle-only test set** (96,168 samples, 70/15/15 stratified split, seed=42) — the same test set used in all prior phases.

**Test set:** 96,168 samples (original Kaggle distribution: 66.77% benign, 14.87% defacement, 3.69% malware, 14.68% phishing)

#### Classification Report (Kaggle-Only Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| benign | 0.9882 | 0.9949 | 0.9915 | 64,212 |
| defacement | 0.9987 | 0.9953 | 0.9970 | 14,296 |
| malware | 0.9961 | 0.9414 | 0.9680 | 3,547 |
| phishing | 0.9588 | 0.9459 | 0.9523 | 14,113 |

| Metric | Value |
|---|---|
| **Overall Accuracy** | **98.58%** |
| Macro Avg Precision | 0.9854 |
| Macro Avg Recall | 0.9694 |
| Macro Avg F1 | 0.9772 |
| Weighted Avg F1 | 0.9857 |

### 8.3 Cross-Phase Comparison (All on Kaggle-Only Test Set)

| Metric | Phase 1 | Phase 2 | Phase 3 | **Phase 4** |
|---|---|---|---|---|
| **Overall Accuracy** | 98.30% | 98.45% | 98.37% | **98.58%** |
| **Phishing F1** | 0.9436 | 0.9485 | 0.9455 | **0.9523** |
| Phishing Precision | 0.9478 | 0.9523 | 0.9541 | **0.9588** |
| Phishing Recall | 0.9395 | 0.9448 | 0.9372 | **0.9459** |
| Benign F1 | 0.9904 | 0.9908 | 0.9905 | **0.9915** |
| Defacement F1 | 0.9953 | 0.9955 | 0.9946 | **0.9970** |
| Malware F1 | 0.9703 | 0.9683 | 0.9672 | 0.9680 |

**Key findings:**
- On an identical test distribution, Phase 4's data augmentation yields the best results across **all** phases: +0.28pp accuracy and +0.0087 phishing F1 vs Phase 1.
- Unlike the augmented test set results (§8.1), the Kaggle-only evaluation shows **no regressions** in minority class performance — defacement F1 actually improved to 0.9970 (best ever), and malware F1 (0.9680) is comparable to Phase 1.
- The in-distribution augmented test set metrics (99.11% accuracy, 0.9928 phishing F1) reflect performance on the shifted 60.7%-phishing distribution. The Kaggle-only results (98.58%, 0.9523) provide the fair cross-phase comparison.

---

## 9. External Stress Test — Brand Impersonation URLs

A curated set of 16 hand-crafted URLs (8 legitimate, 8 brand-impersonation phishing) was tested post-training to assess out-of-distribution generalization:

### Results

| # | URL | Tag | Prediction | Confidence | Correct? |
|---|---|---|---|---|---|
| 1 | `accounts.google.com/signin/v2/` | benign | benign | 52.68% | ✓ |
| 2 | `secure-login-google.com/auth/session` | phishing | benign | 95.03% | ✗ |
| 3 | `paypal.com/myaccount/summary` | benign | benign | 99.88% | ✓ |
| 4 | `paypal-security-center.com/verify/account` | phishing | benign | 99.98% | ✗ |
| 5 | `www.microsoft.com/en-us/security` | benign | phishing | 99.94% | ✗ |
| 6 | `microsoftonline-authentication.com/login` | phishing | benign | 99.31% | ✗ |
| 7 | `github.com/login` | benign | benign | 79.94% | ✓ |
| 8 | `github-secure-auth.com/session/recover` | phishing | benign | 51.83% | ✗ |
| 9 | `appleid.apple.com/` | benign | benign | 99.99% | ✓ |
| 10 | `appleid-verify-now.com/icloud/recovery` | phishing | benign | 98.73% | ✗ |
| 11 | `www.dropbox.com/login` | benign | benign | 61.74% | ✓ |
| 12 | `dropbox-file-share-secure.com/open` | phishing | benign | 72.45% | ✗ |
| 13 | `www.netflix.com/login` | benign | benign | 83.82% | ✓ |
| 14 | `netflix-account-security-center.com/signin` | phishing | benign | 75.95% | ✗ |
| 15 | `portal.office.com/` | benign | benign | 100.00% | ✓ |
| 16 | `office365-credential-check.com/owa/auth` | phishing | benign | 99.65% | ✗ |

### Summary

- **Legitimate URLs**: 7/8 correct (1 false positive on `microsoft.com/en-us/security`)
- **Impersonation phishing URLs**: **0/8 detected** (all predicted benign with high confidence)
- **Conclusion**: The model achieves 99.11% in-distribution test accuracy but fails on novel brand-impersonation phishing URLs that don't appear in the training distribution

---

## 10. Confusion Matrix Analysis

The most critical misclassification pattern is **Phishing → Benign** (false negatives for phishing), as these represent malicious URLs that escape detection.

- Phishing class now dominates at 60.7% of data, up from 14.7%
- In-distribution phishing recall improved substantially (0.9916 vs. 0.9395)
- Malware class remains the minority at 1.7%, with the lowest recall (0.9270)
- Despite high in-distribution metrics, the external stress test reveals the model has not learned brand-impersonation patterns

---

## 11. Software & Hardware Environment

| Component | Version / Specification |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.10.1 (last native Windows GPU build) |
| NumPy | ≥1.20.0, <1.24.0 |
| pandas | ≥2.0.0, <3.0.0 |
| scikit-learn | ≥1.3.0, <2.0.0 |
| seaborn | ≥0.12.0, <1.0.0 |
| matplotlib | ≥3.7.0, <4.0.0 |
| PyYAML | ≥6.0, <7.0 |
| joblib | ≥1.3.0, <2.0.0 |
| GPU | NVIDIA GeForce RTX 4060 (5,447 MB allocated) |
| CUDA | 11.2 |
| cuDNN | 8.1 |
| OS | Windows 10/11 |

---

## 12. Artifacts Produced

| Artifact | Path | Description |
|---|---|---|
| Trained model | `artifacts/model.keras` | Keras model (best epoch 8 weights) |
| Character tokenizer | `artifacts/tokenizer.json` | Fitted Keras Tokenizer (JSON, 331 vocab) |
| Feature scaler | `artifacts/scaler.pkl` | Fitted StandardScaler (joblib, 27 features) |
| Metadata | `artifacts/metadata.json` | max_sequence_length=168, vocab_size=331, n_lexical_features=27 |
| Training curves | `artifacts/results/training_curves.png` | Loss + accuracy vs. epoch |
| Classification report | `artifacts/results/classification_report.txt` | Per-class P/R/F1, overall accuracy (augmented test set) |
| Kaggle-only report | `artifacts/results/kaggle_only_classification_report.txt` | Per-class P/R/F1, overall accuracy (original Kaggle test set) |
| Confusion matrix | `artifacts/results/confusion_matrix.png` | 4×4 confusion matrix |

### Data Pipeline Artifacts

| Artifact | Path | Description |
|---|---|---|
| Raw phishing URLs | `data_pipeline/raw/phishing-links-ACTIVE.txt` | 732,011 raw URLs from GitHub Phishing.Database |
| External phishing CSV | `data_pipeline/processed/external_phishing.csv` | Parsed + deduplicated external phishing URLs |
| Synthetic phishing CSV | `data_pipeline/processed/synthetic_phishing.csv` | 20,000 generated brand-impersonation URLs |
| Merged dataset | `data_pipeline/processed/merged_dataset.csv` | 1,389,956 merged + deduplicated URLs |

---

## 13. Data Pipeline Scripts

| Script | Purpose | CLI |
|---|---|---|
| `download_phishing_urls.py` | Download phishing URLs from GitHub Phishing.Database + OpenPhish | `python download_phishing_urls.py [--skip-openphish] [--from-file <path>]` |
| `generate_synthetic_urls.py` | Generate brand-impersonation phishing URLs (7 strategies) | `python generate_synthetic_urls.py --count 20000 [--seed 42]` |
| `merge_datasets.py` | Merge Kaggle + external + synthetic into unified CSV | `python merge_datasets.py [--kaggle-only] [--external <path>]` |

---

## 14. Key Design Decisions (Phase 4 Specific)

1. **GitHub Phishing.Database as primary external source**: MIT licensed, no registration required, ~780K+ URLs, updated hourly. Chosen after PhishTank registration was found to be "temporarily disabled"
2. **Preserved Phase 3 architecture**: No architectural changes — isolates the effect of data augmentation on model performance
3. **7-strategy synthetic generation**: Covers subdomain abuse, typosquatting, homoglyphs, hyphenation, path injection, IP-based, and long subdomain chains to maximize impersonation pattern diversity
4. **48-brand dictionary**: Covers tech, finance, social media, logistics, government, gaming, and crypto sectors
5. **Multi-source merge with deduplication**: Case-insensitive URL dedup prevents train/test leakage across sources
6. **Scheme stripping at inference**: `classify_url.py` strips `http://`/`https://` before character tokenization to align with training distribution (Kaggle benign URLs lack schemes)

---

## 15. Limitations & Known Issues

1. **Brand impersonation blind spot**: Despite 732K+ real phishing URLs, 20K synthetic impersonation URLs, and 4 brand-aware features, the model classifies all 8 hand-crafted impersonation URLs as benign with high confidence. The model has not learned to generalize brand-impersonation patterns beyond the training distribution
2. **Class imbalance inverted**: Phishing now dominates at 60.7%, which may bias the model toward learning phishing-specific surface patterns (e.g., URL length, character distributions from the GitHub data) rather than structural impersonation signals
3. **Malware class erosion**: Malware recall dropped from 0.9510 (Phase 1) to 0.9270 (Phase 4), consistent with its shrinking proportion (3.69% → 1.7%) in the merged dataset
4. **Scheme handling inconsistency**: Training data mixes URLs with and without schemes. The `_strip_scheme()` function at inference removes schemes, but Branch B feature extraction (e.g., `count_http`, `count_https`) may still encode scheme presence differently across sources
5. **Single trial**: Unlike Phase 1's 3-seed evaluation, Phase 4 reports a single seed=42 trial. Statistical robustness is not established
6. **No class weighting**: Training uses unweighted categorical crossentropy despite the 60.7%/30.7%/6.9%/1.7% class imbalance
7. **Levenshtein distance performance**: Pure-Python edit distance computation on ~1M URLs is computationally expensive during feature extraction
