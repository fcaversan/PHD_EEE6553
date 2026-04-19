"""
Evaluate Phase 4 trained model on the ORIGINAL Kaggle-only test set.
This gives an apples-to-apples comparison with Phases 1-3.

Approach:
  - Load Phase 4 model + tokenizer + scaler (trained on merged data)
  - Load original Kaggle dataset, split 70/15/15 with seed=42
  - Preprocess Kaggle test set using Phase 4 artifacts
  - Generate classification report on the Kaggle test set
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

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
    """
    Load original Kaggle dataset and produce the same test split as Phase 1.
    Uses identical preprocessing: dropna, drop_duplicates, stratified 70/15/15.
    """
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical

    df = pd.read_csv(kaggle_csv)
    print(f"  Loaded Kaggle dataset: {df.shape}")

    # Same cleanup as Phase 1 data_loader
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

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_int, test_size=test_ratio, stratify=y_int, random_state=seed
    )
    # Second split: train vs val
    val_adj = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adj, stratify=y_temp, random_state=seed
    )

    num_classes = config['head']['num_classes']
    y_test_oh = to_categorical(y_test, num_classes=num_classes)

    print(f"  Kaggle test set: {len(X_test)} samples")
    print(f"  Label classes: {label_classes}")
    print(f"  Class distribution in test set:")
    for cls_name, cls_idx in label_to_int.items():
        count = np.sum(y_test == cls_idx)
        print(f"    {cls_name:15s}: {count:6d} ({count/len(y_test)*100:.2f}%)")

    return X_test, y_test_oh, label_classes


def main():
    print("\n" + "=" * 60)
    print("PHASE 4 MODEL — KAGGLE-ONLY TEST SET EVALUATION")
    print("=" * 60)

    config = get_config('config.yaml')
    set_all_seeds(config['random_seed'])

    # ── Load Phase 4 artifacts ───────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 1: Loading Phase 4 Model & Artifacts")
    print(f"{'=' * 60}")

    model_path = config['data']['model_path']
    scaler_path = config['data']['scaler_path']
    tokenizer_path = config['data']['tokenizer_path']
    metadata_path = os.path.join(config['data']['artifacts_dir'], 'metadata.json')

    model = keras.models.load_model(model_path)
    print(f"  Model loaded: {model_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    max_seq_len = metadata['max_sequence_length']
    n_features = int(metadata.get('n_lexical_features', 27))
    print(f"  Metadata: max_seq_len={max_seq_len}, n_features={n_features}")

    tokenizer = load_tokenizer(tokenizer_path)
    print(f"  Tokenizer loaded")

    # ── Load original Kaggle test set ────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 2: Loading Original Kaggle Test Set (same split as Phase 1)")
    print(f"{'=' * 60}")

    kaggle_csv = "../../datasets/malicious_phish.csv"
    X_test, y_test, label_classes = load_kaggle_test_set(kaggle_csv, config)

    # ── Preprocess with Phase 4 artifacts ────────────────────
    print(f"\n{'=' * 60}")
    print("Step 3: Preprocessing with Phase 4 Tokenizer & Scaler")
    print(f"{'=' * 60}")

    # Branch A: char sequences (padded to Phase 4's max_seq_len=168)
    X_test_seq = tokenize_and_pad(tokenizer, X_test, max_seq_len)

    # Branch B: 27 lexical features scaled with Phase 4's scaler
    test_features = extract_features_batch(X_test)
    X_test_scaled = load_and_apply_scaler(test_features, scaler_path)

    # ── Inference ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 4: Running Inference")
    print(f"{'=' * 60}")

    y_pred_probs = model.predict([X_test_seq, X_test_scaled], verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # ── Results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Step 5: Classification Report (Phase 4 model → Kaggle test set)")
    print(f"{'=' * 60}")

    report = classification_report(
        y_true, y_pred, target_names=label_classes, digits=4
    )
    print("\n" + report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Phishing→Benign misclassification rate
    try:
        phishing_idx = label_classes.index('phishing')
        benign_idx = label_classes.index('benign')
        phish_to_benign = cm[phishing_idx, benign_idx]
        total_phishing = cm[phishing_idx, :].sum()
        rate = phish_to_benign / total_phishing * 100 if total_phishing else 0
        print(f"\nPhishing→Benign misclassification: {phish_to_benign}/{total_phishing} ({rate:.2f}%)")
    except ValueError:
        pass

    # Overall accuracy
    accuracy = np.sum(y_pred == y_true) / len(y_true) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Save results
    results_dir = config['data']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, 'kaggle_only_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("PHASE 4 MODEL — KAGGLE-ONLY TEST SET EVALUATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test set: Original Kaggle ({len(X_test)} samples)\n")
        f.write(f"Split: 70/15/15 stratified, seed=42\n\n")
        f.write(report + "\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
