"""
Data loader for Phase 5 — Hierarchical Two-Stage Classification.
Supports both binary (Stage 1: benign vs malicious) and
3-class (Stage 2: defacement, malware, phishing) loading modes.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_dataset(csv_path: str, config: dict, mode: str = 'binary') -> tuple:
    """
    Load and preprocess URL dataset for either binary or 3-class training.

    Args:
        csv_path: Path to CSV file (must have 'url' and 'type' columns)
        config: Configuration dictionary
        mode: 'binary' (benign vs malicious) or 'malicious_only' (3-class sub-classification)

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, label_classes)
    """
    df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
    print(f"  Loaded: {df.shape}")

    df = df.dropna(subset=['url', 'type'])
    df = df.drop_duplicates(subset=['url'])
    print(f"  After cleanup: {df.shape}")

    if mode == 'binary':
        # Map: benign → 'benign', everything else → 'malicious'
        df['type'] = df['type'].apply(lambda t: 'benign' if t == 'benign' else 'malicious')
        label_classes = ['benign', 'malicious']
        num_classes = config['stage1']['num_classes']
        print(f"\n  Binary mode: benign vs malicious")

    elif mode == 'malicious_only':
        # Keep only malicious classes
        df = df[df['type'].isin(['defacement', 'malware', 'phishing'])].copy()
        label_classes = ['defacement', 'malware', 'phishing']
        num_classes = config['stage2']['num_classes']
        print(f"\n  Malicious-only mode: {label_classes}")
        print(f"  Filtered to {len(df)} malicious URLs")

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'binary' or 'malicious_only'")

    # Print class distribution
    print(f"\n  Class distribution:")
    for cls_name, count in df['type'].value_counts().items():
        print(f"    {cls_name:15s}: {count:7d} ({count/len(df)*100:.2f}%)")

    X = df['url'].values
    y = df['type'].values

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

    y_train_oh = to_categorical(y_train, num_classes=num_classes)
    y_val_oh = to_categorical(y_val, num_classes=num_classes)
    y_test_oh = to_categorical(y_test, num_classes=num_classes)

    print(f"\n  Splits:")
    print(f"    Train: {len(X_train):,}")
    print(f"    Val:   {len(X_val):,}")
    print(f"    Test:  {len(X_test):,}")

    return (X_train, y_train_oh, X_val, y_val_oh, X_test, y_test_oh, label_classes)
