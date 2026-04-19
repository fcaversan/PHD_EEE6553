"""
Phase 1A: Binary Classification Evaluation
Collapse Phase 1's 4-class predictions into binary (benign vs malicious)
for a fair comparison with Khan's binary 1D-CNN-Bi-GRU-Attention benchmark (99.08%).

Approach:
  - Load Phase 1 trained model + tokenizer + scaler
  - Load original Kaggle dataset, split 70/15/15 with seed=42
  - Get 4-class softmax predictions
  - Binary mapping: benign → benign, {defacement, malware, phishing} → malicious
  - Method 1: Collapse argmax (predicted class → binary label)
  - Method 2: Sum malicious probabilities (P_mal = P_def + P_mal + P_phi)
  - Compute binary classification report
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

# Ensure src/ is on the path and working directory is the project root
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

from utils import set_all_seeds, get_config
from feature_engineering import extract_features_batch, load_and_apply_scaler
from text_processing import tokenize_and_pad, load_tokenizer


def load_kaggle_test_set(kaggle_csv: str, config: dict):
    """Load original Kaggle dataset and reproduce Phase 1 test split."""
    df = pd.read_csv(kaggle_csv)
    print(f"  Loaded: {df.shape}")
    df = df.dropna(subset=['url', 'type'])
    df = df.drop_duplicates(subset=['url'])
    print(f"  After cleanup: {df.shape}")

    X = df['url'].values
    y = df['type'].values

    label_classes = sorted(df['type'].unique())
    label_to_int = {label: idx for idx, label in enumerate(label_classes)}
    y_int = np.array([label_to_int[label] for label in y])

    seed = config['random_seed']
    test_ratio = config['split']['test_ratio']
    val_ratio = config['split']['val_ratio']
    train_ratio = config['split']['train_ratio']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_int, test_size=test_ratio, stratify=y_int, random_state=seed
    )
    val_adj = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adj, stratify=y_temp, random_state=seed
    )

    num_classes = config['head']['num_classes']
    y_test_oh = to_categorical(y_test, num_classes=num_classes)

    print(f"  Test set: {len(X_test)} samples")
    print(f"  Label classes: {label_classes}")
    return X_test, y_test_oh, label_classes


def main():
    print("\n" + "=" * 60)
    print("PHASE 1A — BINARY CLASSIFICATION EVALUATION")
    print("(benign vs malicious) for comparison with Khan et al.")
    print("=" * 60)

    config = get_config('config.yaml')
    set_all_seeds(config['random_seed'])

    # ── Load Phase 1 artifacts ───────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 1: Loading Phase 1 Model & Artifacts")
    print(f"{'=' * 60}")

    model = keras.models.load_model(config['data']['model_path'])
    print(f"  Model loaded: {config['data']['model_path']}")

    metadata_path = os.path.join(config['data']['artifacts_dir'], 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    max_seq_len = metadata['max_sequence_length']
    label_classes = metadata['label_classes']
    print(f"  Metadata: max_seq_len={max_seq_len}")
    print(f"  4-class labels: {label_classes}")

    tokenizer = load_tokenizer(config['data']['tokenizer_path'])

    # ── Load Kaggle test set ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 2: Loading Kaggle Test Set")
    print(f"{'=' * 60}")

    kaggle_csv = config['data']['dataset_path']
    X_test, y_test_oh, _ = load_kaggle_test_set(kaggle_csv, config)

    # ── Preprocess ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 3: Preprocessing")
    print(f"{'=' * 60}")

    X_test_seq = tokenize_and_pad(tokenizer, X_test, max_seq_len)
    test_features = extract_features_batch(X_test)
    X_test_scaled = load_and_apply_scaler(test_features, config['data']['scaler_path'])

    # ── 4-class predictions ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 4: Running 4-Class Inference")
    print(f"{'=' * 60}")

    y_pred_probs = model.predict([X_test_seq, X_test_scaled], verbose=1)
    y_pred_4class = np.argmax(y_pred_probs, axis=1)
    y_true_4class = np.argmax(y_test_oh, axis=1)

    # ── Binary mapping ───────────────────────────────────────
    # label_classes = ['benign', 'defacement', 'malware', 'phishing'] (sorted)
    benign_idx = label_classes.index('benign')
    print(f"\n  Benign index: {benign_idx}")
    print(f"  Malicious indices: {[i for i in range(len(label_classes)) if i != benign_idx]}")

    # True binary labels: 0 = benign, 1 = malicious
    y_true_binary = (y_true_4class != benign_idx).astype(int)
    
    binary_names = ['benign', 'malicious']

    # ── Method 1: Collapse argmax ────────────────────────────
    print(f"\n{'=' * 60}")
    print("METHOD 1: Collapse Argmax (predicted class → binary)")
    print(f"{'=' * 60}")

    y_pred_binary_m1 = (y_pred_4class != benign_idx).astype(int)

    report_m1 = classification_report(
        y_true_binary, y_pred_binary_m1, target_names=binary_names, digits=4
    )
    print("\n" + report_m1)

    cm_m1 = confusion_matrix(y_true_binary, y_pred_binary_m1)
    print("Confusion Matrix:")
    print(cm_m1)

    acc_m1 = np.sum(y_pred_binary_m1 == y_true_binary) / len(y_true_binary) * 100
    print(f"\nBinary Accuracy (Method 1 — argmax collapse): {acc_m1:.2f}%")

    # ── Method 2: Sum malicious probabilities ────────────────
    print(f"\n{'=' * 60}")
    print("METHOD 2: Sum Malicious Probabilities")
    print(f"  P(malicious) = P(defacement) + P(malware) + P(phishing)")
    print(f"  Predict malicious if P(malicious) > 0.5")
    print(f"{'=' * 60}")

    p_benign = y_pred_probs[:, benign_idx]
    p_malicious = 1.0 - p_benign  # sum of the other 3 classes
    y_pred_binary_m2 = (p_malicious > 0.5).astype(int)

    report_m2 = classification_report(
        y_true_binary, y_pred_binary_m2, target_names=binary_names, digits=4
    )
    print("\n" + report_m2)

    cm_m2 = confusion_matrix(y_true_binary, y_pred_binary_m2)
    print("Confusion Matrix:")
    print(cm_m2)

    acc_m2 = np.sum(y_pred_binary_m2 == y_true_binary) / len(y_true_binary) * 100
    print(f"\nBinary Accuracy (Method 2 — probability sum): {acc_m2:.2f}%")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY — Phase 1A Binary Classification")
    print(f"{'=' * 60}")
    print(f"  Test set: {len(X_test)} samples (original Kaggle)")
    print(f"  Benign:    {np.sum(y_true_binary == 0):,}")
    print(f"  Malicious: {np.sum(y_true_binary == 1):,}")
    print(f"")
    print(f"  Method 1 (argmax collapse):     {acc_m1:.2f}%")
    print(f"  Method 2 (probability sum):     {acc_m2:.2f}%")
    print(f"  Khan et al. benchmark (binary): 99.08%")

    # ── Save ─────────────────────────────────────────────────
    results_dir = config['data']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, 'binary_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("PHASE 1A — BINARY CLASSIFICATION EVALUATION\n")
        f.write("(Phase 1 model, Kaggle test set, benign vs malicious)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test set: {len(X_test)} samples\n")
        f.write(f"Benign: {np.sum(y_true_binary == 0):,} | "
                f"Malicious: {np.sum(y_true_binary == 1):,}\n\n")
        f.write("METHOD 1: Collapse Argmax\n")
        f.write(report_m1 + "\n")
        f.write(f"Binary Accuracy: {acc_m1:.2f}%\n\n")
        f.write("METHOD 2: Sum Malicious Probabilities\n")
        f.write(report_m2 + "\n")
        f.write(f"Binary Accuracy: {acc_m2:.2f}%\n\n")
        f.write(f"Khan et al. benchmark (binary): 99.08%\n")
    print(f"\n  Report saved to: {report_path}")


if __name__ == '__main__':
    main()
