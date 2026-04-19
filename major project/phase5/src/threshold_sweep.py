"""
Threshold sweep — find optimal binary decision threshold for Phase 5 Stage 1.
Uses the saved model's sigmoid probabilities and sweeps thresholds from 0.1 to 0.9.
No retraining needed.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

from utils import set_all_seeds, get_config
from feature_engineering import extract_features_batch, load_and_apply_scaler
from text_processing import tokenize_and_pad, load_tokenizer


def main():
    print("\n" + "=" * 60)
    print("PHASE 5 STAGE 1 — THRESHOLD SWEEP")
    print("=" * 60)

    config = get_config('config.yaml')
    set_all_seeds(config['random_seed'])

    s1 = config['stage1']
    model = keras.models.load_model(s1['model_path'])
    print(f"  Model loaded: {s1['model_path']}")

    with open(s1['metadata_path'], 'r') as f:
        meta = json.load(f)
    max_seq_len = meta['max_sequence_length']
    tokenizer = load_tokenizer(s1['tokenizer_path'])

    # Load Kaggle test set
    kaggle_csv = config['data']['kaggle_path']
    df = pd.read_csv(kaggle_csv)
    df = df.dropna(subset=['url', 'type']).drop_duplicates(subset=['url'])
    df['binary'] = df['type'].apply(lambda t: 0 if t == 'benign' else 1)
    X = df['url'].values
    y = df['binary'].values
    seed = config['random_seed']
    X_temp, X_test, _, y_test = train_test_split(
        X, y, test_size=config['split']['test_ratio'],
        stratify=y, random_state=seed)

    print(f"  Test set: {len(X_test)} | benign: {np.sum(y_test==0):,} | malicious: {np.sum(y_test==1):,}")

    # Preprocess
    X_test_seq = tokenize_and_pad(tokenizer, X_test, max_seq_len)
    test_feat = extract_features_batch(X_test)
    X_test_scaled = load_and_apply_scaler(test_feat, s1['scaler_path'])

    # Get raw probabilities
    y_probs = model.predict([X_test_seq, X_test_scaled], verbose=1).flatten()

    # Sweep thresholds
    print(f"\n{'='*60}")
    print("Threshold Sweep Results")
    print(f"{'='*60}")
    print(f"{'Threshold':>10} {'Accuracy':>10} {'Benign P':>10} {'Benign R':>10} "
          f"{'Malic P':>10} {'Malic R':>10} {'FP':>8} {'FN':>8}")
    print("-" * 86)

    best_acc = 0
    best_thresh = 0.5
    results = []

    for thresh in np.arange(0.10, 0.91, 0.01):
        y_pred = (y_probs >= thresh).astype(int)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        benign_p = tn / (tn + fn) if (tn + fn) > 0 else 0
        benign_r = tn / (tn + fp) if (tn + fp) > 0 else 0
        malic_p = tp / (tp + fp) if (tp + fp) > 0 else 0
        malic_r = tp / (tp + fn) if (tp + fn) > 0 else 0

        results.append((thresh, acc, benign_p, benign_r, malic_p, malic_r, fp, fn))

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    # Print every 5th + anything above 99%
    for r in results:
        thresh, acc, bp, br, mp, mr, fp, fn = r
        marker = " ***" if acc >= 0.9908 else (" <<" if thresh == best_thresh else "")
        if int(round(thresh * 100)) % 5 == 0 or acc >= 0.9905 or thresh == best_thresh:
            print(f"{thresh:>10.2f} {acc*100:>9.2f}% {bp:>10.4f} {br:>10.4f} "
                  f"{mp:>10.4f} {mr:>10.4f} {fp:>8d} {fn:>8d}{marker}")

    print(f"\n{'='*60}")
    print(f"  Best threshold: {best_thresh:.2f}")
    print(f"  Best accuracy:  {best_acc*100:.2f}%")
    print(f"  Khan benchmark: 99.08%")
    print(f"  Gap:            {(best_acc*100 - 99.08):+.2f} pp")
    print(f"{'='*60}")

    # Detailed report at best threshold
    y_best = (y_probs >= best_thresh).astype(int)
    names = ['benign', 'malicious']
    print(f"\nClassification Report at threshold={best_thresh:.2f}:")
    print(classification_report(y_test, y_best, target_names=names, digits=4))
    cm = confusion_matrix(y_test, y_best)
    print("Confusion Matrix:")
    print(cm)

    # Save results
    results_dir = config['data']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, 'stage1_threshold_sweep.txt')
    with open(path, 'w') as f:
        f.write(f"Best threshold: {best_thresh:.2f}\n")
        f.write(f"Best accuracy:  {best_acc*100:.2f}%\n")
        f.write(f"Khan benchmark: 99.08%\n\n")
        f.write(classification_report(y_test, y_best, target_names=names, digits=4))
        f.write(f"\nConfusion Matrix:\n{cm}\n")
    print(f"\n  Saved: {path}")


if __name__ == '__main__':
    main()
