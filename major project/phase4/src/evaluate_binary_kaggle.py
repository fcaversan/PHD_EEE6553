"""
Phase 4A: Binary Classification Evaluation
Collapse Phase 4's 4-class predictions into binary (benign vs malicious)
on the ORIGINAL Kaggle test set, for fair comparison with Khan (99.08%).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

from utils import set_all_seeds, get_config
from feature_engineering import extract_features_batch, load_and_apply_scaler
from text_processing import tokenize_and_pad, load_tokenizer


def load_kaggle_test_set(kaggle_csv, config):
    df = pd.read_csv(kaggle_csv)
    df = df.dropna(subset=['url', 'type'])
    df = df.drop_duplicates(subset=['url'])
    print(f"  Kaggle dataset: {df.shape}")

    X = df['url'].values
    y = df['type'].values

    label_classes = sorted(df['type'].unique())
    label_to_int = {label: idx for idx, label in enumerate(label_classes)}
    y_int = np.array([label_to_int[label] for label in y])

    seed = config['random_seed']
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_int, test_size=config['split']['test_ratio'],
        stratify=y_int, random_state=seed
    )
    val_adj = config['split']['val_ratio'] / (config['split']['train_ratio'] + config['split']['val_ratio'])
    train_test_split(X_temp, y_temp, test_size=val_adj, stratify=y_temp, random_state=seed)

    y_test_oh = to_categorical(y_test, num_classes=config['head']['num_classes'])
    print(f"  Test set: {len(X_test)} samples")
    return X_test, y_test_oh, label_classes


def main():
    print("\n" + "=" * 60)
    print("PHASE 4A — BINARY CLASSIFICATION (Kaggle test set)")
    print("=" * 60)

    config = get_config('config.yaml')
    set_all_seeds(config['random_seed'])

    # Load Phase 4 artifacts
    model = keras.models.load_model(config['data']['model_path'])
    metadata_path = os.path.join(config['data']['artifacts_dir'], 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    max_seq_len = metadata['max_sequence_length']
    label_classes = metadata['label_classes']
    tokenizer = load_tokenizer(config['data']['tokenizer_path'])
    print(f"  Model loaded, max_seq_len={max_seq_len}, labels={label_classes}")

    # Load Kaggle test set
    kaggle_csv = "../../datasets/malicious_phish.csv"
    X_test, y_test_oh, _ = load_kaggle_test_set(kaggle_csv, config)

    # Preprocess
    X_test_seq = tokenize_and_pad(tokenizer, X_test, max_seq_len)
    test_features = extract_features_batch(X_test)
    X_test_scaled = load_and_apply_scaler(test_features, config['data']['scaler_path'])

    # 4-class predictions
    y_pred_probs = model.predict([X_test_seq, X_test_scaled], verbose=1)
    y_pred_4class = np.argmax(y_pred_probs, axis=1)
    y_true_4class = np.argmax(y_test_oh, axis=1)

    # Binary collapse
    benign_idx = label_classes.index('benign')
    y_true_bin = (y_true_4class != benign_idx).astype(int)
    y_pred_bin = (y_pred_4class != benign_idx).astype(int)

    binary_names = ['benign', 'malicious']
    report = classification_report(y_true_bin, y_pred_bin, target_names=binary_names, digits=4)
    print("\n" + report)

    acc = np.sum(y_pred_bin == y_true_bin) / len(y_true_bin) * 100
    print(f"Phase 4 Binary Accuracy (Kaggle test): {acc:.2f}%")
    print(f"Khan et al. benchmark:                 99.08%")

    # Save
    results_dir = config['data']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, 'binary_kaggle_classification_report.txt')
    with open(path, 'w') as f:
        f.write("PHASE 4A — BINARY CLASSIFICATION (Kaggle test set)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test set: {len(X_test)} samples (original Kaggle)\n")
        f.write(f"Benign: {np.sum(y_true_bin==0):,} | Malicious: {np.sum(y_true_bin==1):,}\n\n")
        f.write(report + "\n")
        f.write(f"Binary Accuracy: {acc:.2f}%\n")
        f.write(f"Khan et al.: 99.08%\n")
    print(f"Report saved: {path}")


if __name__ == '__main__':
    main()
