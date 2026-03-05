"""
Data loader — Phase 2
Same split logic as Phase 1, plus BERT subword encoding (Branch C).
Returns five arrays per split: char_sequences, bert_ids, bert_mask, lex_features, labels.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_malicious_urls_dataset(csv_path: str, config: dict) -> tuple:
    """
    Load, deduplicate, and stratified-split the malicious_phish.csv dataset.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, label_classes)
            X splits contain raw URL strings (numpy array of str).
            y splits contain one-hot encoded labels (N, 4).
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded dataset: {csv_path}")
        print(f"  Initial shape: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download from Kaggle: sid321axn/malicious-urls-dataset"
        )

    if 'url' not in df.columns or 'type' not in df.columns:
        raise ValueError(f"Expected 'url' and 'type' columns. Got: {df.columns.tolist()}")

    df = df.dropna(subset=['url', 'type'])
    df = df.drop_duplicates(subset=['url'])
    print(f"  Final shape after dedup: {df.shape}")

    print(f"\n{'='*60}\nClass Distribution\n{'='*60}")
    class_counts = df['type'].value_counts()
    for cls, cnt in class_counts.items():
        print(f"  {cls:15s}: {cnt:7d} ({cnt/len(df)*100:5.2f}%)")

    X = df['url'].values
    y = df['type'].values
    label_classes = sorted(df['type'].unique())
    label_to_int = {lbl: idx for idx, lbl in enumerate(label_classes)}
    y_int = np.array([label_to_int[lbl] for lbl in y])
    print(f"\n  Label encoding: {label_to_int}")

    seed        = config['random_seed']
    test_ratio  = config['split']['test_ratio']
    val_ratio   = config['split']['val_ratio']
    train_ratio = config['split']['train_ratio']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_int, test_size=test_ratio, stratify=y_int, random_state=seed
    )
    val_adj = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adj, stratify=y_temp, random_state=seed
    )

    num_classes = config['head']['num_classes']
    y_train = to_categorical(y_train, num_classes)
    y_val   = to_categorical(y_val,   num_classes)
    y_test  = to_categorical(y_test,  num_classes)

    print(f"\n{'='*60}\nData Splits\n{'='*60}")
    print(f"  Train: {len(X_train):6d} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val):6d} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test):6d} samples ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test, label_classes
