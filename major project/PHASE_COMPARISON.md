# Cross-Phase Comparison — Malicious URL Detection

---

## Phase Summary

| Phase | Focus | Key Change |
|---|---|---|
| **Phase 1** | Baseline dual-input model | Char CNN-BiGRU-Attention + 23 lexical features |
| **Phase 2** | Add pretrained language model | + Frozen DistilBERT semantic branch (triple-input) |
| **Phase 3** | Brand-aware feature engineering | Remove BERT, add 4 brand-impersonation features (27 total) |
| **Phase 4** | Data augmentation at scale | + 732K real phishing URLs + 20K synthetic impersonation URLs |

---

## Architecture Evolution

| Property | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| Inputs | Dual (A+B) | Triple (A+B+C) | Dual (A+B) | Dual (A+B) |
| Branch A | Char CNN-BiGRU-Att | Char CNN-BiGRU-Att | Char CNN-BiGRU-Att | Char CNN-BiGRU-Att |
| Branch B | 23-feature MLP | 23-feature MLP | **27-feature MLP** | **27-feature MLP** |
| Branch C | — | Frozen DistilBERT | — | — |
| Concat dims | 160 | 288 | 160 | 160 |
| Trainable params | ~221K | ~221K | ~221K | ~122.5K |
| Total params | ~221K | ~66.6M | ~221K | ~122.5K |
| Vocab size | 330 | 271 | 271 | **331** |
| Max seq length | 134 | 135 | 135 | **168** |

---

## Dataset Evolution

| Property | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| Source(s) | Kaggle | Kaggle | Kaggle | Kaggle + GitHub Phishing DB + Synthetic |
| Total samples | 641,119 | 641,119 | 641,119 | **1,389,956** |
| Benign | 66.77% | 66.77% | 66.77% | **30.7%** |
| Defacement | 14.87% | 14.87% | 14.87% | **6.9%** |
| Phishing | 14.68% | 14.68% | 14.68% | **60.7%** |
| Malware | 3.69% | 3.69% | 3.69% | **1.7%** |
| Train samples | 448,783 | 448,783 | 448,783 | **972,968** |
| Test samples | 96,168 | 96,168 | 96,168 | **208,494** |

---

## Feature Engineering Evolution

| Feature Group | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| Length features (3) | ✓ | ✓ | ✓ | ✓ |
| Character counts (11) | ✓ | ✓ | ✓ | ✓ |
| Substring counts (3) | ✓ | ✓ | ✓ | ✓ |
| Character-type counts (2) | ✓ | ✓ | ✓ | ✓ |
| Path directories (1) | ✓ | ✓ | ✓ | ✓ |
| IP detection (1) | ✓ | ✓ | ✓ | ✓ |
| URL shortener (1) | ✓ | ✓ | ✓ | ✓ |
| Reserved placeholder (1) | ✓ | ✓ | ✓ | ✓ |
| `brand_in_domain` | — | — | **✓** | **✓** |
| `brand_count` | — | — | **✓** | **✓** |
| `trust_word_in_domain` | — | — | **✓** | **✓** |
| `min_brand_edit_distance` | — | — | **✓** | **✓** |
| DistilBERT embeddings | — | **✓** (frozen) | — | — |
| **Total features** | **23** | **23 + BERT** | **27** | **27** |

---

## Test Set Performance

### Overall Metrics

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| **Accuracy** | 98.30% | 98.45% | 98.37% | **99.11%** |
| Macro Avg F1 | 0.9749 | 0.9758 | — | **0.9826** |
| Weighted Avg F1 | 0.9830 | 0.9845 | — | **0.9911** |

### Per-Class F1-Score

| Class | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| Benign | 0.9904 | **0.9908** | 0.9905 | 0.9895 |
| Defacement | **0.9953** | **0.9955** | 0.9946 | 0.9908 |
| Malware | **0.9703** | 0.9683 | 0.9672 | 0.9572 |
| **Phishing** | 0.9436 | 0.9485 | 0.9455 | **0.9928** |

### Phishing Class Detail

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| Precision | 0.9478 | 0.9523 | 0.9541 | **0.9941** |
| Recall | 0.9395 | 0.9448 | 0.9372 | **0.9916** |
| F1 | 0.9436 | 0.9485 | 0.9455 | **0.9928** |

### Malware Class Detail

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| Precision | 0.9900 | 0.9888 | — | **0.9895** |
| Recall | **0.9510** | 0.9487 | — | 0.9270 |
| F1 | **0.9703** | 0.9683 | 0.9672 | 0.9572 |

---

## External Stress Test — Brand Impersonation (16 URLs)

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| Impersonation phishing detected (of 8) | N/A | **0/8** | **0/8** | **0/8** |
| Benign false positives (of 8) | N/A | 2/8 | 1/8 | **1/8** |
| External test available | No | Yes | Yes | Yes |

**Consistent finding**: Across all tested phases, the model fails to detect hand-crafted brand-impersonation URLs. All 8 phishing URLs (e.g., `paypal-security-center.com`, `appleid-verify-now.com`) are classified as benign with high confidence.

---

## Training Dynamics

| Property | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---|---|---|---|---|
| Batch size | 256 | 128 | 256 | 256 |
| Max epochs | 50 | 50 | 50 | 50 |
| Best epoch | 10 | 11 | — | **8** |
| Early stop triggered | Yes | Yes (ep. 16) | — | Yes (ep. 13) |
| LR schedule | ReduceLR | ReduceLR | ReduceLR | ReduceLR |
| Multi-seed eval | 3 seeds | 1 seed | 1 seed | 1 seed |
| GPU | RTX 4060 | RTX 4060 | RTX 4060 | RTX 4060 |

---

## Key Takeaways

### What Worked

1. **Phase 4 data augmentation** produced the largest accuracy gain (+0.81pp over Phase 1), with phishing F1 jumping from 0.9436 to 0.9928
2. **Phase 2 DistilBERT** provided a modest in-distribution improvement (+0.15pp accuracy) but at 300× parameter cost
3. **Phase 3 brand features** reduced benign false positives from 2 to 1 on the stress test
4. **All phases** achieve ≥98.3% test accuracy on the Kaggle distribution

### What Didn't Work

1. **Brand impersonation detection**: 0/8 across all phases — the core problem remains unsolved
2. **Frozen DistilBERT (Phase 2)**: 66M parameters for +0.0049 phishing F1; not worth the complexity
3. **Brand-aware features (Phase 3)**: 4 new features failed to improve phishing recall; F1 actually declined vs. Phase 2
4. **732K real phishing URLs (Phase 4)**: Improved in-distribution metrics dramatically but did not generalize to novel impersonation patterns

### Trade-Offs Observed

| Phase 4 Gain | Phase 4 Cost |
|---|---|
| Phishing F1: +0.0492 | Defacement F1: −0.0045 |
| Overall accuracy: +0.81pp | Malware recall: −0.024 |
| Phishing recall: +0.0521 | Malware F1: −0.0131 |

The phishing class improvement comes at the expense of minority classes (defacement, malware), consistent with the class distribution shift from 14.7% → 60.7% phishing.

---

## Open Problem

Despite four iterations spanning architecture changes (BERT), feature engineering (brand-aware), and data augmentation (732K+ real phishing URLs), the model consistently fails to detect **novel brand-impersonation phishing URLs** that are structurally similar to legitimate domains (e.g., `paypal-security-center.com`). The root cause is a **generalization gap**: the model learns surface-level statistical patterns from training data rather than the semantic concept of "this domain is impersonating a known brand."
