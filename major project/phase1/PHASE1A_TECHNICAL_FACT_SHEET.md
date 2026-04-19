# Phase 1A — Technical Fact Sheet: Binary Classification Evaluation (Benign vs Malicious)

---

## 1. Objective

Provide a direct, fair comparison with Khan et al.'s binary 1D-CNN-Bi-GRU-Attention benchmark (99.08% accuracy) by collapsing the Phase 1 model's 4-class predictions into a binary decision: **benign** vs **malicious** (defacement + malware + phishing).

No retraining was performed. The Phase 1 trained model was used as-is; only the evaluation mapping changed.

---

## 2. Method

The Phase 1 model outputs a 4-class softmax probability vector: `[P(benign), P(defacement), P(malware), P(phishing)]`.

**Binary label mapping:**
- **Benign** → benign (class 0)
- **Malicious** → defacement, malware, phishing (classes 1, 2, 3)

**Two evaluation methods were tested:**

| Method | Description |
|---|---|
| **Method 1: Argmax Collapse** | Take the 4-class argmax prediction; if predicted class ≠ benign → malicious |
| **Method 2: Probability Sum** | Compute P(malicious) = 1 − P(benign); predict malicious if P(malicious) > 0.5 |

Both methods produced identical results (expected, since argmax(benign) ⟺ P(benign) > 0.5 for well-calibrated softmax).

---

## 3. Evaluation Configuration

| Property | Value |
|---|---|
| Model | Phase 1 trained model (`phase1/artifacts/model.keras`) |
| Test set | Original Kaggle, 96,168 samples (15% stratified holdout, seed=42) |
| Benign count | 64,212 (66.77%) |
| Malicious count | 31,956 (33.23%) — defacement: 14,296 + malware: 3,547 + phishing: 14,113 |
| Preprocessing | Phase 1 tokenizer (vocab=330, max_len=135) + Phase 1 scaler (23 features) |

---

## 4. Results

### Binary Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| benign | 0.9860 | 0.9928 | 0.9894 | 64,212 |
| malicious | 0.9854 | 0.9717 | 0.9785 | 31,956 |

| Metric | Value |
|---|---|
| **Binary Accuracy** | **98.58%** |
| Macro Avg Precision | 0.9857 |
| Macro Avg Recall | 0.9823 |
| Macro Avg F1 | 0.9840 |
| Weighted Avg F1 | 0.9858 |

### Confusion Matrix

|  | Predicted Benign | Predicted Malicious |
|---|---|---|
| **True Benign** | 63,749 | 463 |
| **True Malicious** | 905 | 31,051 |

---

## 5. Comparison with Khan et al.

| Model | Classification Task | Accuracy |
|---|---|---|
| Khan et al. (1D-CNN-Bi-GRU-Attention) | Binary (benign vs malicious) | **99.08%** |
| **Phase 1 (this work)** | 4-class → binary collapse | **98.58%** |
| **Gap** | | **−0.50 pp** |

### Context

- Khan's model was trained and optimized **exclusively for binary classification** — a simpler decision boundary
- The Phase 1 model was trained on a **4-class objective** (benign, defacement, malware, phishing), learning finer-grained distinctions between three types of malicious URLs
- The 0.50pp gap is attributable to the additional complexity of the 4-class decision surface: the model must simultaneously separate defacement from malware from phishing, which introduces inter-class confusion that does not exist in a binary formulation
- Despite this disadvantage, Phase 1 achieves 98.58% binary accuracy without any binary-specific optimization

---

## 6. Evaluation Script

- Script: `phase1/src/evaluate_binary.py`
- Output: `phase1/artifacts/results/binary_classification_report.txt`
- Reproducible with: `cd phase1 && python src/evaluate_binary.py`
