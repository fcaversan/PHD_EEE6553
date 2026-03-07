# Phase 3 — Technical Fact Sheet

## 1. Objective
Phase 3 focuses on improving real-world impersonation phishing detection by extending lexical heuristics with brand-aware features, while keeping the lightweight dual-input architecture from Phase 1.

## 2. Design Summary
- Architecture baseline: Phase 1 dual-input model (Character CNN-BiGRU-Attention + Lexical MLP)
- Major change: Branch B features expanded from 23 to 27
- New features:
  - `brand_in_domain`
  - `brand_count`
  - `trust_word_in_domain`
  - `min_brand_edit_distance`
- BERT branch removed to reduce complexity and runtime

## 3. Folder Structure
- `phase3/config.yaml`
- `phase3/requirements.txt`
- `phase3/src/`
  - `brand_features.py` (new)
  - `feature_engineering.py` (27 features)
  - `model_builder.py` (lexical input shape uses config `input_features`)
  - `train.py`, `evaluate.py`, `classify_url.py`, `external_phishing_test.py`
- `phase3/tests/`
  - `test_brand_features.py` (new)
  - `test_feature_engineering.py` (updated)
  - `test_model_builder.py` (updated)

## 4. Data and Split Configuration
- Dataset: `../../datasets/malicious_phish.csv`
- Split: 70% train / 15% val / 15% test
- Seed plan: 42, 123, 7

## 5. Model Configuration
- Branch A: identical to Phase 1
- Branch B: `input_features=27`, Dense(64) → Dropout(0.3) → Dense(32)
- Head: Concatenate(128+32) → Dense(128) → Dropout(0.5) → Softmax(4)

## 6. Validation Status
- Project scaffold: complete
- Phase 3 code migration: complete
- Brand-aware feature module: complete
 - Unit tests:
  - `test_brand_features.py`: passing
  - `test_feature_engineering.py`: passing
  - `test_model_builder.py`: core architecture/compilation tests passing

## 7. Experimental Runs Status
1. Train seed=42 — complete
2. Evaluate seed=42 (classification report + confusion matrix) — complete
3. External stress test (16 curated URLs) — complete
4. Repeat seeds 123 and 7 — pending
5. Aggregate multi-trial statistics — pending

## 8. Results (Seed=42)

### 8.1 Benchmark test set (`96,168` samples)
- Accuracy: **98.37%**
- Phishing precision: **0.9541**
- Phishing recall: **0.9372**
- Phishing F1: **0.9455**

Per-class F1:
- benign: 0.9905
- defacement: 0.9946
- malware: 0.9672
- phishing: 0.9455

### 8.2 External phishing stress test (16 curated URLs)
- Predicted benign: **15**
- Predicted phishing: **1**
- Impersonation phishing detected: **0 / 8**
- Benign false positives: **1 / 8** (`microsoft.com` flagged as phishing)

## 9. Comparison Table (P1 vs P2 vs P3)
| Metric | Phase 1 | Phase 2 | Phase 3 |
|---|---:|---:|---:|
| Accuracy | 98.19% | **98.45%** | 98.37% |
| Phishing F1 | 0.9396 | **0.9485** | 0.9455 |
| External phishing detected (8) | N/A | 0 | 0 |
| External benign false positives (8) | N/A | 2 | **1** |
| Trainable params | ~221K | ~221K | ~221K |
| Total params | ~221K | ~66.6M | ~221K |

Interpretation:
- Phase 3 improves over the latest regenerated Phase 1 benchmark metrics.
- Phase 3 does **not** surpass Phase 2 on benchmark phishing F1.
- On the external impersonation set, Phase 3 still fails the core objective (0/8 detected), though it reduces benign false positives vs Phase 2 (2 → 1).

## 10. Repro Commands
From `phase3/`:
- Train: `python src/train.py`
- Evaluate: `python src/evaluate.py`
- External test: `python src/external_phishing_test.py`
- Stats (after 3 trials): `python src/compute_trial_stats.py`

## 11. Current Conclusion
Phase 3's targeted brand-aware lexical features produce a moderate benchmark lift over the regenerated Phase 1 run, but still do not solve real-world impersonation phishing detection in the curated external set. The core generalization gap remains data-driven and likely requires either (a) explicit domain-identity/reputation signals or (b) training data augmentation focused on impersonation patterns.
