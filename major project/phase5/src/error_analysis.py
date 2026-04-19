"""
Error Analysis — Identify and characterize hard-to-classify samples.

Compares Phase 4 (4-class) and Phase 5 Stage 1 (binary) predictions on the
original Kaggle test set to find:
  1. Which samples Phase 4 misclassifies (especially phishing)
  2. Which samples Phase 5 misclassifies
  3. The "hard core" — samples BOTH models get wrong
  4. URL-level feature analysis of hard samples
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import Counter
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

from utils import set_all_seeds, get_config
from feature_engineering import extract_features_batch, load_and_apply_scaler
from text_processing import tokenize_and_pad, load_tokenizer


# ── Kaggle test set (with original 4-class labels) ─────────────
def load_kaggle_test_with_labels(kaggle_csv, config):
    """Load Kaggle test set preserving original 4-class labels."""
    df = pd.read_csv(kaggle_csv)
    df = df.dropna(subset=['url', 'type'])
    df = df.drop_duplicates(subset=['url'])

    X = df['url'].values
    y_labels = df['type'].values

    label_classes = sorted(df['type'].unique())
    label_to_int = {l: i for i, l in enumerate(label_classes)}
    y_int = np.array([label_to_int[l] for l in y_labels])

    seed = config['random_seed']
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_int, test_size=config['split']['test_ratio'],
        stratify=y_int, random_state=seed)
    val_adj = config['split']['val_ratio'] / (
        config['split']['train_ratio'] + config['split']['val_ratio'])
    train_test_split(X_temp, y_temp, test_size=val_adj,
                     stratify=y_temp, random_state=seed)

    int_to_label = {i: l for l, i in label_to_int.items()}
    y_test_labels = np.array([int_to_label[i] for i in y_test])

    return X_test, y_test, y_test_labels, label_classes


# ── URL-level analysis helpers ──────────────────────────────────
def extract_url_characteristics(url):
    """Quick feature extraction for analysis (no ML features)."""
    try:
        parsed_url = url if url.startswith(('http://', 'https://')) else 'http://' + url
        parsed = urlparse(parsed_url)
        hostname = parsed.netloc
        path = parsed.path
        tld = hostname.rsplit('.', 1)[-1] if '.' in hostname else ''
    except Exception:
        hostname, path, tld = '', '', ''

    return {
        'url_length': len(url),
        'hostname': hostname,
        'tld': tld,
        'path_length': len(path),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_slashes': url.count('/'),
        'num_digits': sum(c.isdigit() for c in url),
        'num_special': sum(not c.isalnum() and c not in './-:' for c in url),
        'has_ip': bool(any(c.isdigit() for c in hostname.split('.')[0])
                       and hostname.replace('.', '').isdigit()) if hostname else False,
        'has_at': '@' in url,
        'has_https': url.startswith('https'),
        'num_subdomains': hostname.count('.') - 1 if hostname else 0,
    }


def analyze_group(urls, label='Group'):
    """Print aggregate statistics for a set of URLs."""
    if len(urls) == 0:
        print(f"  {label}: 0 samples\n")
        return

    chars = [extract_url_characteristics(u) for u in urls]
    df = pd.DataFrame(chars)

    print(f"\n  {label} ({len(urls)} URLs)")
    print(f"  {'─'*50}")
    print(f"  URL length:   mean={df['url_length'].mean():.0f}  "
          f"median={df['url_length'].median():.0f}  "
          f"min={df['url_length'].min()}  max={df['url_length'].max()}")
    print(f"  Path length:  mean={df['path_length'].mean():.0f}  "
          f"median={df['path_length'].median():.0f}")
    print(f"  Dots:         mean={df['num_dots'].mean():.1f}")
    print(f"  Hyphens:      mean={df['num_hyphens'].mean():.1f}")
    print(f"  Digits:       mean={df['num_digits'].mean():.1f}")
    print(f"  Special chars: mean={df['num_special'].mean():.1f}")
    print(f"  Has IP addr:  {df['has_ip'].sum()} ({df['has_ip'].mean()*100:.1f}%)")
    print(f"  Has @:        {df['has_at'].sum()} ({df['has_at'].mean()*100:.1f}%)")
    print(f"  Has HTTPS:    {df['has_https'].sum()} ({df['has_https'].mean()*100:.1f}%)")
    print(f"  Subdomains:   mean={df['num_subdomains'].mean():.1f}")

    # Top TLDs
    tld_counts = Counter(df['tld'])
    top = tld_counts.most_common(10)
    print(f"  Top TLDs:     {top}")


def main():
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS — Hard-to-Classify Samples")
    print("=" * 70)

    config = get_config('config.yaml')
    set_all_seeds(config['random_seed'])

    kaggle_csv = config['data']['kaggle_path']

    # ── Load Kaggle test set with 4-class labels ─────────────
    print("\n[1] Loading Kaggle test set...")
    X_test, y_test_int, y_test_labels, label_classes = \
        load_kaggle_test_with_labels(kaggle_csv, config)
    y_test_binary = np.array([0 if l == 'benign' else 1 for l in y_test_labels])

    print(f"  Total: {len(X_test):,}")
    for cls in label_classes:
        cnt = np.sum(y_test_labels == cls)
        print(f"    {cls:15s}: {cnt:6d} ({cnt/len(X_test)*100:.1f}%)")

    # ── Load Phase 4 model (4-class) ────────────────────────
    print(f"\n[2] Loading Phase 4 model...")
    p4_dir = os.path.join(_PROJECT_DIR, '..', 'phase4')
    p4_config = get_config(os.path.join(p4_dir, 'config.yaml'))

    p4_model_path = os.path.join(p4_dir, p4_config['data']['model_path'])
    p4_scaler_path = os.path.join(p4_dir, p4_config['data']['scaler_path'])
    p4_tokenizer_path = os.path.join(p4_dir, p4_config['data']['tokenizer_path'])
    p4_meta_path = os.path.join(p4_dir, p4_config['data']['artifacts_dir'], 'metadata.json')

    p4_model = keras.models.load_model(p4_model_path)
    with open(p4_meta_path) as f:
        p4_meta = json.load(f)
    p4_max_seq = p4_meta['max_sequence_length']
    p4_tokenizer = load_tokenizer(p4_tokenizer_path)

    # Phase 4 inference
    print("  Preprocessing for Phase 4...")
    p4_seq = tokenize_and_pad(p4_tokenizer, X_test, p4_max_seq)
    p4_feat = extract_features_batch(X_test)
    p4_scaled = load_and_apply_scaler(p4_feat, p4_scaler_path)

    print("  Running Phase 4 inference...")
    p4_probs = p4_model.predict([p4_seq, p4_scaled], verbose=1)
    p4_pred_int = np.argmax(p4_probs, axis=1)
    p4_pred_labels = np.array([label_classes[i] for i in p4_pred_int])

    # Phase 4 binary predictions (collapse non-benign → malicious)
    p4_pred_binary = np.array([0 if l == 'benign' else 1 for l in p4_pred_labels])

    # ── Load Phase 5 Stage 1 model (binary) ─────────────────
    print(f"\n[3] Loading Phase 5 Stage 1 model...")
    s1 = config['stage1']
    p5_model = keras.models.load_model(s1['model_path'])
    with open(s1['metadata_path']) as f:
        p5_meta = json.load(f)
    p5_max_seq = p5_meta['max_sequence_length']
    p5_tokenizer = load_tokenizer(s1['tokenizer_path'])

    print("  Preprocessing for Phase 5...")
    p5_seq = tokenize_and_pad(p5_tokenizer, X_test, p5_max_seq)
    p5_feat = extract_features_batch(X_test)
    p5_scaled = load_and_apply_scaler(p5_feat, s1['scaler_path'])

    print("  Running Phase 5 inference...")
    p5_probs = p5_model.predict([p5_seq, p5_scaled], verbose=1).flatten()
    p5_pred_binary = (p5_probs > 0.5).astype(int)

    # ═══════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════

    # ── Phase 4: 4-class error breakdown ─────────────────────
    print(f"\n{'='*70}")
    print("PHASE 4 — 4-CLASS ERROR BREAKDOWN")
    print(f"{'='*70}")

    p4_correct = p4_pred_int == y_test_int
    p4_wrong_mask = ~p4_correct
    p4_wrong_count = p4_wrong_mask.sum()
    print(f"\nTotal misclassified: {p4_wrong_count:,} / {len(X_test):,} "
          f"({p4_wrong_count/len(X_test)*100:.2f}%)")

    # Per-class error counts
    print("\nPer-class misclassifications:")
    for cls in label_classes:
        cls_mask = y_test_labels == cls
        cls_total = cls_mask.sum()
        cls_wrong = (p4_wrong_mask & cls_mask).sum()
        print(f"  {cls:15s}: {cls_wrong:4d} / {cls_total:5d} missed "
              f"({cls_wrong/cls_total*100:.2f}% error rate)")

    # Where do misclassified phishing samples go?
    phishing_mask = y_test_labels == 'phishing'
    phishing_wrong_mask = p4_wrong_mask & phishing_mask
    phishing_wrong_preds = p4_pred_labels[phishing_wrong_mask]
    print(f"\nPhishing misclassification destinations (Phase 4):")
    for dest, cnt in Counter(phishing_wrong_preds).most_common():
        print(f"  phishing → {dest}: {cnt}")

    # Where do misclassified malware samples go?
    malware_mask = y_test_labels == 'malware'
    malware_wrong_mask = p4_wrong_mask & malware_mask
    malware_wrong_preds = p4_pred_labels[malware_wrong_mask]
    print(f"\nMalware misclassification destinations (Phase 4):")
    for dest, cnt in Counter(malware_wrong_preds).most_common():
        print(f"  malware → {dest}: {cnt}")

    # ── Phase 4 binary error breakdown ────────────────────────
    print(f"\n{'='*70}")
    print("PHASE 4 — BINARY ERROR BREAKDOWN (benign vs malicious)")
    print(f"{'='*70}")

    p4_bin_wrong = p4_pred_binary != y_test_binary
    p4_fn = (p4_pred_binary == 0) & (y_test_binary == 1)  # Malicious called benign
    p4_fp = (p4_pred_binary == 1) & (y_test_binary == 0)  # Benign called malicious

    print(f"\n  False Negatives (malicious → benign): {p4_fn.sum()}")
    print(f"  False Positives (benign → malicious): {p4_fp.sum()}")
    print(f"  Total binary errors: {p4_bin_wrong.sum()}")

    # FN breakdown by original class
    print(f"\n  FN breakdown by original class:")
    for cls in ['defacement', 'malware', 'phishing']:
        cnt = ((y_test_labels == cls) & p4_fn).sum()
        total = (y_test_labels == cls).sum()
        print(f"    {cls:15s}: {cnt:4d} / {total:5d} called benign "
              f"({cnt/total*100:.2f}%)")

    # ── Phase 5 binary error breakdown ────────────────────────
    print(f"\n{'='*70}")
    print("PHASE 5 STAGE 1 — BINARY ERROR BREAKDOWN")
    print(f"{'='*70}")

    p5_wrong = p5_pred_binary != y_test_binary
    p5_fn = (p5_pred_binary == 0) & (y_test_binary == 1)
    p5_fp = (p5_pred_binary == 1) & (y_test_binary == 0)

    print(f"\n  False Negatives (malicious → benign): {p5_fn.sum()}")
    print(f"  False Positives (benign → malicious): {p5_fp.sum()}")
    print(f"  Total binary errors: {p5_wrong.sum()}")

    # FN breakdown by original class
    print(f"\n  FN breakdown by original class:")
    for cls in ['defacement', 'malware', 'phishing']:
        cnt = ((y_test_labels == cls) & p5_fn).sum()
        total = (y_test_labels == cls).sum()
        print(f"    {cls:15s}: {cnt:4d} / {total:5d} called benign "
              f"({cnt/total*100:.2f}%)")

    # ═══════════════════════════════════════════════════════════
    # HARD CORE — Samples BOTH models get wrong (binary)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("HARD CORE — Samples BOTH Phase 4 and Phase 5 misclassify (binary)")
    print(f"{'='*70}")

    both_wrong = p4_bin_wrong & p5_wrong
    both_fn = p4_fn & p5_fn  # Both call malicious as benign
    both_fp = p4_fp & p5_fp  # Both call benign as malicious

    only_p4_wrong = p4_bin_wrong & ~p5_wrong  # Phase 4 wrong, Phase 5 right
    only_p5_wrong = ~p4_bin_wrong & p5_wrong  # Phase 5 wrong, Phase 4 right

    print(f"\n  Both wrong:         {both_wrong.sum():5d}")
    print(f"    Both FN:          {both_fn.sum():5d}")
    print(f"    Both FP:          {both_fp.sum():5d}")
    print(f"  Only Phase 4 wrong: {only_p4_wrong.sum():5d}")
    print(f"  Only Phase 5 wrong: {only_p5_wrong.sum():5d}")

    # Hard core FN by original class
    print(f"\n  Hard-core FN (both miss) by original class:")
    for cls in ['defacement', 'malware', 'phishing']:
        cnt = ((y_test_labels == cls) & both_fn).sum()
        total = (y_test_labels == cls).sum()
        print(f"    {cls:15s}: {cnt:4d} / {total:5d} "
              f"({cnt/total*100:.2f}%)")

    # ═══════════════════════════════════════════════════════════
    # URL CHARACTERISTIC COMPARISON
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("URL CHARACTERISTICS — Correctly vs Misclassified")
    print(f"{'='*70}")

    # Phishing analysis (the hardest class)
    phishing_urls = X_test[phishing_mask]
    phishing_correct_p4 = X_test[phishing_mask & p4_correct]
    phishing_wrong_p4 = X_test[phishing_wrong_mask]
    phishing_both_fn_urls = X_test[(y_test_labels == 'phishing') & both_fn]

    analyze_group(phishing_correct_p4, "Phishing — Phase 4 CORRECT")
    analyze_group(phishing_wrong_p4, "Phishing — Phase 4 WRONG")
    analyze_group(phishing_both_fn_urls, "Phishing — BOTH models miss (hard core)")

    # Compare with correctly classified benign (what do benign-looking phish look like?)
    benign_correct = X_test[(y_test_labels == 'benign') & p4_correct]
    analyze_group(benign_correct[:5000], "Benign — Phase 4 CORRECT (sample of 5000)")

    # ═══════════════════════════════════════════════════════════
    # SAMPLE HARD URLs
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SAMPLE HARD-CORE URLs (both models miss)")
    print(f"{'='*70}")

    # Sample phishing URLs both models call benign
    phishing_hard_idx = np.where((y_test_labels == 'phishing') & both_fn)[0]
    print(f"\nPhishing called benign by BOTH ({len(phishing_hard_idx)} total):")
    for i, idx in enumerate(phishing_hard_idx[:30]):
        url = X_test[idx]
        p4_label = p4_pred_labels[idx]
        p5_prob = p5_probs[idx]
        print(f"  [{i+1:2d}] P5_prob={p5_prob:.4f}  P4_pred={p4_label:12s}  "
              f"URL={url[:120]}")

    # Sample benign URLs both models call malicious
    benign_hard_idx = np.where((y_test_labels == 'benign') & both_fp)[0]
    print(f"\nBenign called malicious by BOTH ({len(benign_hard_idx)} total):")
    for i, idx in enumerate(benign_hard_idx[:20]):
        url = X_test[idx]
        p4_label = p4_pred_labels[idx]
        p5_prob = p5_probs[idx]
        print(f"  [{i+1:2d}] P5_prob={p5_prob:.4f}  P4_pred={p4_label:12s}  "
              f"URL={url[:120]}")

    # Sample malware URLs both models call benign
    malware_hard_idx = np.where((y_test_labels == 'malware') & both_fn)[0]
    print(f"\nMalware called benign by BOTH ({len(malware_hard_idx)} total):")
    for i, idx in enumerate(malware_hard_idx[:20]):
        url = X_test[idx]
        p5_prob = p5_probs[idx]
        print(f"  [{i+1:2d}] P5_prob={p5_prob:.4f}  URL={url[:120]}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 5 CONFIDENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PHASE 5 CONFIDENCE — Where is the model uncertain?")
    print(f"{'='*70}")

    # Confidence = distance from 0.5 threshold
    confidence = np.abs(p5_probs - 0.5)

    # Confidence distribution for correct vs incorrect
    correct_mask = p5_pred_binary == y_test_binary
    print(f"\n  Correct predictions:   mean confidence={confidence[correct_mask].mean():.4f}")
    print(f"  Incorrect predictions: mean confidence={confidence[~correct_mask].mean():.4f}")

    # How many errors in the "uncertain zone" (0.3 < prob < 0.7)?
    uncertain = (p5_probs > 0.3) & (p5_probs < 0.7)
    print(f"\n  Uncertain zone (0.3-0.7): {uncertain.sum():,} samples "
          f"({uncertain.sum()/len(X_test)*100:.1f}% of test)")
    print(f"  Errors in uncertain zone: {(uncertain & ~correct_mask).sum()}")
    print(f"  Errors outside uncertain zone: {(~uncertain & ~correct_mask).sum()}")

    # Confidence distribution for phishing
    phishing_probs = p5_probs[phishing_mask]
    print(f"\n  Phishing predictions (should be >0.5):")
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                   (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                   (0.8, 0.9), (0.9, 1.01)]:
        cnt = ((phishing_probs >= lo) & (phishing_probs < hi)).sum()
        bar = '█' * (cnt // 20)
        print(f"    [{lo:.1f}-{hi:.1f}): {cnt:5d} {bar}")

    # ═══════════════════════════════════════════════════════════
    # SAVE MISCLASSIFIED URLs TO CSV
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Saving misclassified URLs to CSV...")
    print(f"{'='*70}")

    results_dir = config['data']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # All misclassified (by either model)
    either_wrong = p4_bin_wrong | p5_wrong
    misclassified_df = pd.DataFrame({
        'url': X_test[either_wrong],
        'true_label': y_test_labels[either_wrong],
        'true_binary': y_test_binary[either_wrong],
        'p4_pred_4class': p4_pred_labels[either_wrong],
        'p4_pred_binary': p4_pred_binary[either_wrong],
        'p5_prob': p5_probs[either_wrong],
        'p5_pred_binary': p5_pred_binary[either_wrong],
        'p4_wrong': p4_bin_wrong[either_wrong].astype(int),
        'p5_wrong': p5_wrong[either_wrong].astype(int),
        'both_wrong': both_wrong[either_wrong].astype(int),
    })
    csv_path = os.path.join(results_dir, 'misclassified_urls.csv')
    misclassified_df.to_csv(csv_path, index=False)
    print(f"  Saved {len(misclassified_df)} misclassified URLs to {csv_path}")

    # Hard core only
    hard_df = pd.DataFrame({
        'url': X_test[both_wrong],
        'true_label': y_test_labels[both_wrong],
        'true_binary': y_test_binary[both_wrong],
        'p4_pred_4class': p4_pred_labels[both_wrong],
        'p5_prob': p5_probs[both_wrong],
        'error_type': np.where(both_fn[both_wrong], 'FN', 'FP'),
    })
    hard_path = os.path.join(results_dir, 'hard_core_misclassified.csv')
    hard_df.to_csv(hard_path, index=False)
    print(f"  Saved {len(hard_df)} hard-core URLs to {hard_path}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Kaggle test set:            {len(X_test):,}")
    print(f"  Phase 4 binary errors:      {p4_bin_wrong.sum():,} ({p4_bin_wrong.sum()/len(X_test)*100:.2f}%)")
    print(f"  Phase 5 binary errors:      {p5_wrong.sum():,} ({p5_wrong.sum()/len(X_test)*100:.2f}%)")
    print(f"  BOTH wrong (hard core):     {both_wrong.sum():,} ({both_wrong.sum()/len(X_test)*100:.2f}%)")
    print(f"  Only Phase 4 wrong:         {only_p4_wrong.sum():,}")
    print(f"  Only Phase 5 wrong:         {only_p5_wrong.sum():,}")
    print(f"\n  If we could fix hard core → accuracy would be:")
    fixable = p5_wrong.sum() - both_wrong.sum()  # Phase 5 errors that Phase 4 gets right
    max_acc = (len(X_test) - fixable) / len(X_test) * 100
    print(f"    Phase 5 unique errors:    {fixable}")
    print(f"    Theoretical max (P4∪P5):  {max_acc:.2f}%")
    hard_core = both_wrong.sum()
    theoretical_max = (len(X_test) - hard_core) / len(X_test) * 100
    print(f"    Hard ceiling (both wrong): {theoretical_max:.2f}%")


if __name__ == '__main__':
    main()
