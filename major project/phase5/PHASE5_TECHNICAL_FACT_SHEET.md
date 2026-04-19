# Phase 5 — Technical Fact Sheet: Hierarchical Binary Classification with Gated Brand Cross-Attention

---

## 1. Objective

Reframe the classification task as **hierarchical two-stage detection**: Stage 1 performs binary classification (benign vs malicious) to maximize detection accuracy against the Khan et al. benchmark (99.08%), while introducing a novel **Gated Brand Cross-Attention** mechanism designed to address the zero-day brand-impersonation vulnerability identified across Phases 2–4.

The training dataset is unchanged from Phase 4 (1,389,956 URLs). The key contribution is architectural: a third attention stream that cross-attends the 4 brand-aware features over the BiGRU sequence, gated by a learned sigmoid to avoid interference on non-brand URLs.

---

## 2. What Changed from Phase 4

| Aspect | Phase 4 | Phase 5 Stage 1 |
|---|---|---|
| Classification task | 4-class (benign, defacement, malware, phishing) | **Binary (benign vs malicious)** |
| Output activation | Softmax (4 units) | **Sigmoid (1 unit)** |
| Loss function | Categorical Crossentropy | **Binary Crossentropy** |
| Architecture | 2-stream (Branch A + Branch B) | **3-stream (Branch A + Brand Cross-Attention + Branch B)** |
| Concat dimensions | 160 (128 + 32) | **288 (128 + 128 + 32)** |
| New layers | — | `brand_slice`, `brand_query_proj`, `brand_query_reshape`, `brand_cross_attention`, `brand_context_flat`, `brand_gate`, `brand_context_gated` |
| Early stopping patience | 5 | **15** |
| ReduceLR patience | 3 | **5** |
| Vocab size | 331 | **332** |
| Benchmark target | Kaggle 4-class accuracy | **Khan et al. 99.08% binary accuracy** |
| Kaggle binary accuracy | 98.86% (collapsed from 4-class) | **99.15%** |

---

## 3. Dataset

### 3.1 Data Source

Identical to Phase 4 — the merged augmented dataset is reused without modification.

| Source | Count | Percentage |
|---|---|---|
| Kaggle `malicious_phish.csv` | 638,865 | 46.0% |
| GitHub Phishing.Database | 731,092 | 52.6% |
| Synthetic impersonation | 19,999 | 1.4% |
| **Total** | **1,389,956** | 100% |

### 3.2 Binary Label Mapping

| Original Class | Binary Label |
|---|---|
| benign | **benign** (0) |
| defacement | **malicious** (1) |
| malware | **malicious** (1) |
| phishing | **malicious** (1) |

### 3.3 Binary Class Distribution

| Class | Count | Percentage |
|---|---|---|
| malicious | 962,615 | 69.26% |
| benign | 427,341 | 30.74% |

### 3.4 Stratified Split (seed=42)

| Split | Samples | Ratio |
|---|---|---|
| Train | 972,968 | 70% |
| Validation | 208,494 | 15% |
| Test | 208,494 | 15% |

---

## 4. Preprocessing Pipeline

Unchanged from Phase 4.

### 4.1 Branch A — Character-Level Tokenization

| Property | Value |
|---|---|
| Tokenizer | Keras `Tokenizer(char_level=True, lower=False, oov_token='<OOV>')` |
| Fitted on | Training URLs only |
| Vocabulary size | 332 |
| Max sequence length | 168 (95th percentile) |
| Padding | Post-padded with zeros |
| Data type | int32 |

### 4.2 Branch B — 27 Lexical Heuristic Features

Identical to Phase 4: 23 baseline + 4 brand-aware features. Normalized with `StandardScaler` fitted on training data only.

The 4 brand-aware features (indices 24–27) serve dual purpose in Phase 5:
1. **Branch B input**: Fed through the Dense(64)→Dense(32) MLP as in Phase 4
2. **Brand Cross-Attention query**: Sliced and projected as the cross-attention query (new in Phase 5)

---

## 5. Model Architecture

### 5.1 Key Innovation: Gated Brand Cross-Attention

The core architectural contribution of Phase 5. Motivation: across Phases 2–4, the model scored **0/8** on hand-crafted brand-impersonation phishing URLs. The 4 brand features fire correctly on these URLs (e.g., `brand_in_domain=1` for `paypal-security-center.com`) but get overwhelmed by Branch A's high benign confidence in the concatenation layer.

**Design**: The 4 brand features are projected into a query vector that cross-attends over the BiGRU character sequence, producing a brand-conditioned context vector. A sigmoid gate learned from the raw brand features modulates this context — when brand signals are zero (non-brand URLs), the gate approaches 0 and the brand stream contributes nothing, preserving baseline accuracy.

### 5.2 Branch A — Character Sequence Processing (CNN-BiGRU-Attention)

| Layer | Configuration | Output Shape |
|---|---|---|
| Input | `(168,)` int32 | `(batch, 168)` |
| Embedding | 332 → 32 dims | `(batch, 168, 32)` |
| Conv1D | 128 filters, kernel=3, ReLU, padding='same' | `(batch, 168, 128)` |
| MaxPooling1D | pool_size=2 | `(batch, 84, 128)` |
| Bidirectional GRU | 64 units, return_sequences=True | `(batch, 84, 128)` |
| Self-Attention | `([x, x])` | `(batch, 84, 128)` |
| GlobalAveragePooling1D | — | `(batch, 128)` |

The BiGRU output (before self-attention) is also tapped as the **key/value** for the brand cross-attention.

### 5.3 Branch B — Lexical Heuristic MLP

| Layer | Configuration | Output Shape |
|---|---|---|
| Input | `(27,)` float32 | `(batch, 27)` |
| Dense | 64 units, ReLU | `(batch, 64)` |
| Dropout | rate=0.3 | `(batch, 64)` |
| Dense | 32 units, ReLU | `(batch, 32)` |

### 5.4 Gated Brand Cross-Attention (New)

| Layer | Configuration | Output Shape |
|---|---|---|
| Lambda (brand_slice) | Slice last 4 features from 27-d input | `(batch, 4)` |
| Dense (brand_query_proj) | 128 units, ReLU | `(batch, 128)` |
| Reshape (brand_query) | → `(1, 128)` | `(batch, 1, 128)` |
| Attention (cross) | Query=brand_query, Key/Value=BiGRU `(84, 128)` | `(batch, 1, 128)` |
| Reshape (flatten) | → `(128,)` | `(batch, 128)` |
| Dense (brand_gate) | 128 units, **Sigmoid** (from raw 4-d brand feats) | `(batch, 128)` |
| Multiply (gated) | context × gate | `(batch, 128)` |

**Gate behaviour**:
- When `brand_in_domain=0`, `brand_count=0`, `trust_word_in_domain=0`, `min_brand_edit_distance≈1.0`: gate outputs ≈ 0 → brand stream is silenced
- When brand features fire (impersonation signal): gate opens → brand-conditioned sequence context flows into the classification head

### 5.5 Classification Head

| Layer | Configuration | Output Shape |
|---|---|---|
| Concatenate | A(128) + BrandCtx(128) + B(32) = **288** | `(batch, 288)` |
| Dense | 128 units, ReLU | `(batch, 128)` |
| Dropout | rate=0.5 | `(batch, 128)` |
| Dense (output) | 1 unit, **Sigmoid** | `(batch, 1)` |

### 5.6 Parameter Count

Increased from ~122.5K (Phase 4) due to the brand cross-attention layers:
- brand_query_proj: 4 × 128 + 128 = 640
- brand_gate: 4 × 128 + 128 = 640
- Attention (cross): internal parameters
- Head dense_head: now 288 × 128 + 128 = 36,992 (was 160 × 128 + 128 = 20,608)

---

## 6. Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 (initial) |
| Batch size | 256 |
| Max epochs | 50 |
| Early stopping | patience=**15**, monitor=val_loss, restore_best_weights=True |
| ReduceLROnPlateau | patience=**5**, factor=0.5, min_lr=1e-7 |
| Loss function | **Binary Crossentropy** |
| Metrics | Accuracy, Precision, Recall |
| Random seed | 42 |
| Labels | Single column P(malicious) ∈ {0, 1} |
| Model inputs | `[X_char_sequences, X_lexical_features_scaled]` |

**Rationale for increased patience**: The previous Stage 1 run (without brand attention) peaked at epoch 8 of 23. With the new brand cross-attention branch adding parameters, more epochs are needed for the gate to calibrate.

---

## 7. Training History

| Property | Value |
|---|---|
| Total epochs | 25 |
| Best epoch | **10** |
| Early stopping triggered | Epoch 25 (15 epochs past best) |
| Final learning rate | 1.25e-4 (reduced 3× from initial 1e-3) |
| GPU | NVIDIA RTX 4060 (5,447 MB allocated) |
| Time per epoch | ~92s |

### Per-Epoch Progression (Key Milestones)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.0110 | 99.62% | — | — |
| 2 | 0.0096 | 99.65% | — | — |
| 10 (best) | — | — | **0.0191** | **99.42%** |
| 25 (stopped) | 0.00039 | 99.99% | 0.0654 | 99.42% |

Best validation results (epoch 10):

| Metric | Value |
|---|---|
| val_loss | 0.0191 |
| val_accuracy | 99.42% |
| val_precision | 99.66% |
| val_recall | 99.50% |

---

## 8. Test Set Evaluation Results

### 8.1 Kaggle-Only Binary Test Set (Primary Benchmark)

**Test set:** 96,168 samples from original Kaggle dataset (70/15/15 stratified split, seed=42). Binary mapping: benign=0, all others=1.

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| benign | 0.9916 | 0.9957 | 0.9937 | 64,212 |
| malicious | 0.9914 | 0.9831 | 0.9872 | 31,956 |

| Metric | Value |
|---|---|
| **Overall Binary Accuracy** | **99.15%** |
| Macro Avg Precision | 0.9915 |
| Macro Avg Recall | 0.9894 |
| Macro Avg F1 | 0.9904 |
| Weighted Avg F1 | 0.9915 |

#### Confusion Matrix

|  | Pred Benign | Pred Malicious |
|---|---|---|
| **Actual Benign** | 63,938 | 274 |
| **Actual Malicious** | 541 | 31,415 |

- **False Positives** (benign → malicious): 274
- **False Negatives** (malicious → benign): 541

### 8.2 Comparison with Khan et al. Benchmark

| Model | Binary Accuracy | Difference |
|---|---|---|
| Khan et al. (2024) | 99.08% | — |
| Phase 4 (collapsed 4-class) | 98.86% | −0.22pp |
| Phase 5 Stage 1 (no brand attn) | 98.92% | −0.16pp |
| Phase 5 Stage 1 (threshold 0.44) | 98.93% | −0.15pp |
| **Phase 5 Stage 1 (brand cross-attn)** | **99.15%** | **+0.07pp** |

### 8.3 Augmented (Merged) Binary Test Set

**Test set:** 208,494 samples (15% holdout from merged dataset)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| benign | 0.9872 | 0.9924 | 0.9898 | 64,101 |
| malicious | 0.9966 | 0.9943 | 0.9955 | 144,393 |

| Metric | Value |
|---|---|
| **Augmented Binary Accuracy** | **99.37%** |

### 8.4 FP/FN Reduction vs Previous Stage 1

| Metric | No Brand Attn | Brand Cross-Attn | Change |
|---|---|---|---|
| False Positives | 381 | **274** | −28.1% |
| False Negatives | 648 | **541** | −16.5% |
| Total errors | 1,029 | **815** | −20.8% |

---

## 9. External Stress Test — Brand Impersonation URLs

Same 16 curated URLs (8 legitimate, 8 brand-impersonation phishing) used across Phases 2–4.

### Results

| # | URL | Tag | Prediction | Confidence | Correct? |
|---|---|---|---|---|---|
| 1 | `accounts.google.com/signin/v2/` | benign | malicious | 72.19% | No |
| 2 | `secure-login-google.com/auth/session` | phishing | benign | 99.50% | No |
| 3 | `paypal.com/myaccount/summary` | benign | benign | 99.99% | Yes |
| 4 | `paypal-security-center.com/verify/account` | phishing | benign | 99.99% | No |
| 5 | `www.microsoft.com/en-us/security` | benign | malicious | 100.00% | No |
| 6 | `microsoftonline-authentication.com/login` | phishing | benign | 100.00% | No |
| 7 | `github.com/login` | benign | malicious | 52.49% | No |
| 8 | `github-secure-auth.com/session/recover` | phishing | benign | 98.12% | No |
| 9 | `appleid.apple.com/` | benign | benign | 96.14% | Yes |
| 10 | `appleid-verify-now.com/icloud/recovery` | phishing | benign | 99.89% | No |
| 11 | `www.dropbox.com/login` | benign | malicious | 84.43% | No |
| 12 | `dropbox-file-share-secure.com/open` | phishing | benign | 99.96% | No |
| 13 | `www.netflix.com/login` | benign | benign | 99.09% | Yes |
| 14 | `netflix-account-security-center.com/signin` | phishing | benign | 99.85% | No |
| 15 | `portal.office.com/` | benign | benign | 100.00% | Yes |
| 16 | `office365-credential-check.com/owa/auth` | phishing | benign | 100.00% | No |

### Summary

- **Legitimate URLs**: 4/8 correct (3 new FPs vs Phase 4's 1 FP — more aggressive model)
- **Impersonation phishing URLs**: **0/8 detected** (unchanged from Phases 2–4)

### Cross-Phase Stress Test Comparison

| Metric | Phase 2 | Phase 3 | Phase 4 | **Phase 5** |
|---|---|---|---|---|
| Impersonation detected (of 8) | 0/8 | 0/8 | 0/8 | **0/8** |
| Benign correct (of 8) | 6/8 | 7/8 | 7/8 | **4/8** |

### Analysis

The Gated Brand Cross-Attention improved in-distribution accuracy (+0.23pp on Kaggle binary) but failed to solve out-of-distribution brand impersonation. The model became more aggressive overall (more FPs on legitimate brand URLs like `microsoft.com`, `github.com`, `dropbox.com`) without learning to distinguish legitimate brand domains from impersonation domains. This confirms that the zero-day impersonation vulnerability is a **domain identity** problem, not a pattern-matching problem — the model cannot learn from training data alone that `paypal.com` is legitimate while `paypal-security-center.com` is not.

---

## 10. Threshold Sweep Analysis

A pre-brand-attention Stage 1 model was swept across thresholds 0.10–0.90:

| Threshold | Accuracy | FP | FN |
|---|---|---|---|
| 0.50 (default) | 98.92% | 381 | 648 |
| 0.44 (optimal) | 98.93% | 407 | 622 |

The accuracy curve was extremely flat between 0.35–0.55, indicating the model's decision boundary is well-calibrated and threshold tuning offers negligible improvement.

---

## 11. Software & Hardware Environment

| Component | Version / Specification |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.10.1 (last native Windows GPU build) |
| NumPy | <2.0 (pinned for TF 2.10 compatibility) |
| pandas | ≥2.0.0 |
| scikit-learn | ≥1.3.0 |
| GPU | NVIDIA GeForce RTX 4060 (5,447 MB allocated) |
| CUDA | 11.2 |
| cuDNN | 8.1 |
| OS | Windows 10/11 |

---

## 12. Artifacts Produced

| Artifact | Path | Description |
|---|---|---|
| Stage 1 model | `artifacts/stage1_model.keras` | Best epoch 10 weights (binary, brand cross-attention) |
| Tokenizer | `artifacts/stage1_tokenizer.json` | Fitted character tokenizer (vocab=332) |
| Scaler | `artifacts/stage1_scaler.pkl` | Fitted StandardScaler (27 features) |
| Metadata | `artifacts/stage1_metadata.json` | max_seq=168, vocab=332, n_features=27 |
| Training curves | `artifacts/results/stage1_training_curves.png` | Loss + accuracy vs epoch |
| Kaggle report | `artifacts/results/stage1_kaggle_binary_report.txt` | Classification report + confusion matrix |

---

## 13. Scripts

| Script | Purpose | CLI |
|---|---|---|
| `train.py` | Train Stage 1 or Stage 2 model | `python src/train.py --stage 1` |
| `evaluate.py` | Evaluate on Kaggle + augmented test sets | `python src/evaluate.py` |
| `stress_test.py` | 16-URL brand impersonation stress test | `python src/stress_test.py` |
| `threshold_sweep.py` | Sweep binary decision thresholds | `python src/threshold_sweep.py` |
| `error_analysis.py` | Cross-model error comparison (P4 vs P5) | `python src/error_analysis.py` |

---

## 14. Key Design Decisions (Phase 5 Specific)

1. **Binary reformulation**: Collapsing defacement/malware/phishing → malicious enables direct comparison with Khan et al. (2024) and simplifies the detection task
2. **Gated Brand Cross-Attention**: Novel mechanism where brand features query the BiGRU sequence via cross-attention, modulated by a sigmoid gate. The gate is critical — without it, the brand stream would interfere on non-brand URLs and degrade accuracy
3. **3-stream merge**: Expanding concatenation from 160-d to 288-d gives the classification head more capacity to integrate brand-conditioned information alongside general sequence and lexical features
4. **Increased patience (15 epochs)**: The brand cross-attention adds parameters that need more training to calibrate the sigmoid gate
5. **BiGRU tap before self-attention**: The cross-attention queries the BiGRU output *before* self-attention, preserving positional character information that might be averaged out after self-attention

---

## 15. Limitations & Known Issues

1. **Zero-day impersonation vulnerability persists**: Despite the novel Gated Brand Cross-Attention, the model scores 0/8 on hand-crafted impersonation URLs. The fundamental issue is that the model cannot learn domain identity from URL strings alone — `paypal-security-center.com` is structurally valid and indistinguishable from legitimate domains without an external reputation signal
2. **Increased benign FP rate on stress test**: Legitimate brand URLs (microsoft.com, github.com, dropbox.com) are now misclassified at higher rates (4/8 wrong vs 1/8 in Phase 4), suggesting the brand cross-attention made the model more aggressive on brand-containing URLs without learning the correct direction
3. **Lambda layer serialization warning**: The `brand_slice` Lambda layer generates a deserialization warning when loading the model. Functionally harmless but could be replaced with a custom layer for cleaner serialization
4. **Stage 2 not yet trained**: The hierarchical pipeline is designed for 2-stage classification (binary → 3-class sub-classification on malicious URLs), but only Stage 1 has been trained
5. **Single trial**: Results from seed=42 only; statistical robustness not established
6. **Levenshtein performance**: Pure-Python edit distance on 972K training URLs takes ~15–20 minutes during feature extraction

---

## 16. Open Problem & Future Direction

The zero-day impersonation vulnerability is confirmed as a **domain identity problem**, not a pattern-matching problem. Five phases of increasingly sophisticated approaches — architecture changes (BERT), feature engineering (brand-aware), data augmentation (732K+ real phishing + 20K synthetic), and now attention mechanisms (Gated Brand Cross-Attention) — have all failed to detect novel impersonation domains.

The most promising remaining approach is a **hybrid rule-based override**: if `brand_in_domain == 1` (hostname contains a known brand but registered domain ≠ official domain), flag as suspicious regardless of model confidence. This is deterministic, explainable, and would immediately catch all 8 stress-test impersonation URLs. The trade-off is that it moves from pure ML to a hybrid ML + rules system.
