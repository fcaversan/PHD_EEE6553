"""
Evaluation — Phase 5 Stage 1 Binary Model on Original Kaggle Test Set.
Direct comparison with Khan et al. (99.08%).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
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


def load_kaggle_binary_test(csv_path, config):
    """Load original Kaggle dataset, binary-mapped, same split."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['url', 'type'])
    df = df.drop_duplicates(subset=['url'])
    print(f"  Kaggle dataset: {df.shape}")

    # Binary mapping
    df['binary'] = df['type'].apply(lambda t: 0 if t == 'benign' else 1)

    X = df['url'].values
    y = df['binary'].values

    seed = config['random_seed']
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config['split']['test_ratio'],
        stratify=y, random_state=seed)
    val_adj = config['split']['val_ratio'] / (config['split']['train_ratio'] + config['split']['val_ratio'])
    train_test_split(X_temp, y_temp, test_size=val_adj, stratify=y_temp, random_state=seed)

    print(f"  Test set: {len(X_test)} | benign: {np.sum(y_test==0):,} | malicious: {np.sum(y_test==1):,}")
    return X_test, y_test


def main():
    print("\n" + "=" * 60)
    print("PHASE 5 STAGE 1 — BINARY EVAL ON KAGGLE TEST SET")
    print("=" * 60)

    config = get_config('config.yaml')
    set_all_seeds(config['random_seed'])

    # Load model & artifacts
    s1 = config['stage1']
    model = keras.models.load_model(s1['model_path'])
    print(f"  Model loaded: {s1['model_path']}")

    with open(s1['metadata_path'], 'r') as f:
        meta = json.load(f)
    max_seq_len = meta['max_sequence_length']
    tokenizer = load_tokenizer(s1['tokenizer_path'])

    # Load Kaggle test set
    kaggle_csv = config['data']['kaggle_path']
    X_test, y_true = load_kaggle_binary_test(kaggle_csv, config)

    # Preprocess
    X_test_seq = tokenize_and_pad(tokenizer, X_test, max_seq_len)
    test_feat = extract_features_batch(X_test)
    X_test_scaled = load_and_apply_scaler(test_feat, s1['scaler_path'])

    # Predict
    y_pred_probs = model.predict([X_test_seq, X_test_scaled], verbose=1).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Report
    names = ['benign', 'malicious']
    report = classification_report(y_true, y_pred, target_names=names, digits=4)
    print("\n" + report)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    acc = np.sum(y_pred == y_true) / len(y_true) * 100
    print(f"\n  Phase 5 Stage 1 Binary Accuracy (Kaggle): {acc:.2f}%")
    print(f"  Khan et al. benchmark:                     99.08%")

    # Also evaluate on the augmented test set for completeness
    print(f"\n{'='*60}")
    print("Augmented test set evaluation")
    print(f"{'='*60}")
    from data_loader import load_dataset
    _, _, _, _, X_test_aug, y_test_aug, _ = load_dataset(
        config['data']['dataset_path'], config, mode='binary')
    y_test_aug = y_test_aug[:, 1:2]
    X_aug_seq = tokenize_and_pad(tokenizer, X_test_aug, max_seq_len)
    aug_feat = extract_features_batch(X_test_aug)
    X_aug_scaled = load_and_apply_scaler(aug_feat, s1['scaler_path'])
    y_aug_probs = model.predict([X_aug_seq, X_aug_scaled], verbose=1).flatten()
    y_aug_pred = (y_aug_probs > 0.5).astype(int)
    y_aug_true = y_test_aug.flatten().astype(int)
    aug_report = classification_report(y_aug_true, y_aug_pred, target_names=names, digits=4)
    print("\n" + aug_report)
    aug_acc = np.sum(y_aug_pred == y_aug_true) / len(y_aug_true) * 100
    print(f"  Augmented Binary Accuracy: {aug_acc:.2f}%")

    # Save
    results_dir = config['data']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, 'stage1_kaggle_binary_report.txt')
    with open(path, 'w') as f:
        f.write("PHASE 5 STAGE 1 — BINARY CLASSIFICATION (Kaggle test)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test set: {len(X_test)} samples (original Kaggle)\n\n")
        f.write(report + "\n")
        f.write(f"Binary Accuracy: {acc:.2f}%\n")
        f.write(f"Khan et al.:     99.08%\n\n")
        f.write(f"Augmented test set:\n")
        f.write(aug_report + "\n")
        f.write(f"Augmented Accuracy: {aug_acc:.2f}%\n")
    print(f"\n  Report saved: {path}")


if __name__ == '__main__':
    main()
