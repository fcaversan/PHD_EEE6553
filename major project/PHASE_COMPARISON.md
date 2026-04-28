# Cross-Phase Comparison — Malicious URL Detection

---

## Phase Summary

| Phase | Focus | Key Change |
|---|---|---|
| **Phase 1** | Baseline dual-input model | Char CNN-BiGRU-Attention + 23 lexical features |
| **Phase 2** | Add pretrained language model | + Frozen DistilBERT semantic branch (triple-input) |
| **Phase 3** | Brand-aware feature engineering | Remove BERT, add 4 brand-impersonation features (27 total) |
| **Phase 4** | Data augmentation at scale | + 732K real phishing URLs + 20K synthetic impersonation URLs |
| **Phase 5** | Gated Brand Cross-Attention | + 3-stream merge with learned sigmoid gate; binary + 4-class training |

---

## Architecture Evolution

| Property | Phase 1 | Phase 2 | Phase 3 | Phase 4 | **Phase 5** |
|---|---|---|---|---|---|
| Inputs | Dual (A+B) | Triple (A+B+C) | Dual (A+B) | Dual (A+B) | **Triple (A+Brand+B)** |
| Branch A | Char CNN-BiGRU-Att | Char CNN-BiGRU-Att | Char CNN-BiGRU-Att | Char CNN-BiGRU-Att | Char CNN-BiGRU-Att |
| Branch B | 23-feature MLP | 23-feature MLP | **27-feature MLP** | **27-feature MLP** | **27-feature MLP** |
| Branch C | — | Frozen DistilBERT | — | — | **Gated Brand Cross-Attn** |
| Concat dims | 160 | 288 | 160 | 160 | **288** |
| Trainable params | ~221K | ~221K | ~221K | ~122.5K | **~140K** |
| Total params | ~221K | ~66.6M | ~221K | ~122.5K | **~140K** |
| Vocab size | 330 | 271 | 271 | **331** | **332** |
| Max seq length | 134 | 135 | 135 | **168** | **168** |

---

## Dataset Evolution

| Property | Phase 1 | Phase 2 | Phase 3 | Phase 4 | **Phase 5** |
|---|---|---|---|---|---|
| Source(s) | Kaggle | Kaggle | Kaggle | Kaggle + GitHub Phishing DB + Synthetic | Same as Phase 4 |
| Total samples | 641,119 | 641,119 | 641,119 | **1,389,956** | **1,389,956** |
| Benign | 66.77% | 66.77% | 66.77% | **30.7%** | **30.7%** |
| Defacement | 14.87% | 14.87% | 14.87% | **6.9%** | **6.9%** |
| Phishing | 14.68% | 14.68% | 14.68% | **60.7%** | **60.7%** |
| Malware | 3.69% | 3.69% | 3.69% | **1.7%** | **1.7%** |
| Train samples | 448,783 | 448,783 | 448,783 | **972,968** | **972,968** |
| Test samples | 96,168 | 96,168 | 96,168 | **208,494** | **208,494** |

---

## Feature Engineering Evolution

| Feature Group | Phase 1 | Phase 2 | Phase 3 | Phase 4 | **Phase 5** |
|---|---|---|---|---|---|
| Length features (3) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Character counts (11) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Substring counts (3) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Character-type counts (2) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Path directories (1) | ✓ | ✓ | ✓ | ✓ | ✓ |
| IP detection (1) | ✓ | ✓ | ✓ | ✓ | ✓ |
| URL shortener (1) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Reserved placeholder (1) | ✓ | ✓ | ✓ | ✓ | ✓ |
| `brand_in_domain` | — | — | **✓** | **✓** | **✓** (+ cross-attn query) |
| `brand_count` | — | — | **✓** | **✓** | **✓** (+ cross-attn query) |
| `trust_word_in_domain` | — | — | **✓** | **✓** | **✓** (+ cross-attn query) |
| `min_brand_edit_distance` | — | — | **✓** | **✓** | **✓** (+ cross-attn query) |
| DistilBERT embeddings | — | **✓** (frozen) | — | — | — |
| **Total features** | **23** | **23 + BERT** | **27** | **27** | **27** |

---

## Test Set Performance

### Overall Metrics (Kaggle-Only Test Set, 96,168 samples)

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 | **Phase 5** |
|---|---|---|---|---|---|
| **4-Class Accuracy** | 98.30% | 98.45% | 98.37% | 98.58% | **98.69%** |
| **Binary Accuracy** | 98.58%* | — | — | 98.86%* | **99.15%** |
| Macro Avg F1 | 0.9749 | 0.9758 | — | 0.9790 | **0.9790** |
| Weighted Avg F1 | 0.9830 | 0.9845 | — | 0.9868 | **0.9868** |

*\*Binary accuracy from collapsed 4-class predictions, not retrained as binary.*

**Note:** Phase 4 previously reported 99.11% accuracy, which was measured on the augmented test set (208,494 samples). The Kaggle-only figure (96,168 samples) used here for fair cross-phase comparison is 98.58%.

### Per-Class F1-Score (Kaggle-Only Test Set)

| Class | Phase 1 | Phase 2 | Phase 3 | Phase 4 | **Phase 5** |
|---|---|---|---|---|---|
| Benign | 0.9904 | 0.9908 | 0.9905 | 0.9915 | **0.9922** |
| Defacement | 0.9953 | **0.9955** | 0.9946 | 0.9970 | **0.9972** |
| Malware | **0.9703** | 0.9683 | 0.9672 | 0.9680 | 0.9705 |
| **Phishing** | 0.9436 | 0.9485 | 0.9455 | 0.9523 | **0.9561** |

### Phishing Class Detail (Kaggle-Only Test Set)

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 | **Phase 5** |
|---|---|---|---|---|---|
| Precision | 0.9478 | 0.9523 | 0.9541 | 0.9578 | **0.9651** |
| Recall | 0.9395 | 0.9448 | 0.9372 | 0.9469 | **0.9472** |
| F1 | 0.9436 | 0.9485 | 0.9455 | 0.9523 | **0.9561** |
| Phishing→Benign | ~776 | ~734 | ~757 | ~715 | **704** |
| Phishing→Benign % | ~5.50% | ~5.20% | ~5.36% | ~5.07% | **4.99%** |

### Malware Class Detail (Kaggle-Only Test Set)

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 | **Phase 5** |
|---|---|---|---|---|---|
| Precision | 0.9900 | 0.9888 | — | 0.9895 | **0.9961** |
| Recall | **0.9510** | 0.9487 | — | 0.9270 | 0.9462 |
| F1 | **0.9703** | 0.9683 | 0.9672 | 0.9680 | 0.9705 |

---

## External Stress Test — Brand Impersonation (16 URLs)

| Metric | Phase 1 | Phase 2 | Phase 3 | Phase 4 | **Phase 5** |
|---|---|---|---|---|---|
| Impersonation phishing detected (of 8) | N/A | **0/8** | **0/8** | **0/8** | **0/8** |
| Benign correct (of 8) | N/A | 6/8 | 7/8 | 7/8 | **4/8** |
| External test available | No | Yes | Yes | Yes | Yes |

**Consistent finding**: Across all five phases, the model fails to detect hand-crafted brand-impersonation URLs. All 8 phishing URLs (e.g., `paypal-security-center.com`, `appleid-verify-now.com`) are classified as benign with high confidence (>98%). Phase 5's Gated Brand Cross-Attention made the model more aggressive on brand-containing URLs without learning the correct direction, resulting in more benign FPs (4/8 wrong vs 1/8 in Phase 4).

---

## Training Dynamics

| Property | Phase 1 | Phase 2 | Phase 3 | Phase 4 | **Phase 5 (binary)** | **Phase 5 (4-class)** |
|---|---|---|---|---|---|---|
| Batch size | 256 | 128 | 256 | 256 | 256 | 256 |
| Max epochs | 50 | 50 | 50 | 50 | 50 | 50 |
| Best epoch | 10 | 11 | — | **8** | **10** | **8** |
| Early stop triggered | Yes | Yes (ep. 16) | — | Yes (ep. 13) | Yes (ep. 25) | Yes (ep. 23) |
| ES patience | 5 | 5 | 5 | 5 | **15** | **15** |
| LR schedule | ReduceLR | ReduceLR | ReduceLR | ReduceLR | ReduceLR | ReduceLR |
| Multi-seed eval | 3 seeds | 1 seed | 1 seed | 1 seed | 1 seed | 1 seed |
| GPU | RTX 4060 | RTX 4060 | RTX 4060 | RTX 4060 | RTX 4060 | RTX 4060 |

---

## Key Takeaways

### What Worked

1. **Phase 4 data augmentation** produced the largest single-phase accuracy gain (+0.28pp over Phase 3), confirming training data diversity matters most
2. **Phase 5 Gated Brand Cross-Attention** improved both binary (+0.23pp) and 4-class (+0.11pp) accuracy with only ~18K additional parameters
3. **Phase 5 binary model** surpassed the Khan et al. 99.08% benchmark, achieving 99.15% (+0.07pp)
4. **Phase 5** is the first phase to satisfy the ≤5% phishing→benign safety threshold (4.99%)
5. **Phase 2 DistilBERT** provided a modest in-distribution improvement (+0.15pp accuracy) but at 300× parameter cost
6. **All phases** achieve ≥98.3% test accuracy on the Kaggle distribution

### What Didn't Work

1. **Brand impersonation detection**: 0/8 across all five phases — the core problem remains unsolved
2. **Frozen DistilBERT (Phase 2)**: 66M parameters for +0.15pp accuracy; not worth the complexity
3. **Brand-aware features (Phase 3)**: 4 new features failed to improve phishing recall; F1 actually declined vs. Phase 2
4. **Gated Brand Cross-Attention (Phase 5)**: Improved general accuracy but failed at its primary design goal (impersonation detection); made the model more aggressive on brand URLs without learning the correct direction (4/8 benign brand URLs misclassified vs 1/8 in Phase 4)

---

## Open Problem

Despite five iterations spanning architecture changes (DistilBERT, Gated Brand Cross-Attention), feature engineering (brand-aware features), and data augmentation (732K+ real phishing + 20K synthetic impersonation URLs), the model consistently fails to detect **novel brand-impersonation phishing URLs** that are structurally similar to legitimate domains (e.g., `paypal-security-center.com`). This is confirmed as a **domain identity problem**, not a pattern-matching problem — the model cannot learn from URL strings alone that `paypal.com` is legitimate while `paypal-security-center.com` is not. The most promising remaining approach is a **hybrid rule-based override**: if a hostname contains a known brand but the registered domain is not the official one, flag as suspicious regardless of model confidence.
