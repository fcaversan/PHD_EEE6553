"""
Data loader for Malicious URL Detection Model
Handles CSV loading, preprocessing, train/val/test splitting, and label encoding
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_malicious_urls_dataset(csv_path: str, config: dict) -> tuple:
    """
    Load and preprocess the Kaggle malicious URLs dataset.
    
    Performs:
    - CSV loading with error handling
    - Duplicate & NaN removal with logging
    - Class distribution analysis
    - Stratified train/val/test split (70/15/15)
    - One-hot encoding of labels (4 classes)
    
    Args:
        csv_path: Path to the malicious-urls-dataset.csv file
        config: Configuration dictionary containing split ratios and random seed
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, label_encoder_classes)
            - X splits contain raw URL strings
            - y splits contain one-hot encoded labels (N, 4)
            - label_encoder_classes: list of class names in encoding order
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing or data is invalid
    """
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded dataset: {csv_path}")
        print(f"  Initial shape: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            f"Please download from Kaggle: sid321axn/malicious-urls-dataset"
        )
    
    # Validate required columns
    if 'url' not in df.columns or 'type' not in df.columns:
        raise ValueError(
            f"Dataset must contain 'url' and 'type' columns. "
            f"Found: {df.columns.tolist()}"
        )
    
    # Remove NaN values
    initial_count = len(df)
    df = df.dropna(subset=['url', 'type'])
    nan_dropped = initial_count - len(df)
    if nan_dropped > 0:
        print(f"  Dropped {nan_dropped} rows with NaN values")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['url'])
    duplicates_dropped = initial_count - nan_dropped - len(df)
    if duplicates_dropped > 0:
        print(f"  Dropped {duplicates_dropped} duplicate URLs")
    
    print(f"  Final shape: {df.shape}")
    
    # Class distribution analysis
    print(f"\n{'='*60}")
    print("Class Distribution")
    print(f"{'='*60}")
    class_counts = df['type'].value_counts()
    for class_name, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name:15s}: {count:7d} ({percentage:5.2f}%)")
    
    # Prepare features and labels
    X = df['url'].values
    y = df['type'].values
    
    # Encode labels to integers
    label_classes = sorted(df['type'].unique())
    label_to_int = {label: idx for idx, label in enumerate(label_classes)}
    y_int = np.array([label_to_int[label] for label in y])
    
    print(f"\n  Label encoding: {label_to_int}")
    
    # Stratified split: first split off test set (15%)
    test_size = config['split']['test_ratio']
    val_size = config['split']['val_ratio']
    train_size = config['split']['train_ratio']
    random_seed = config['random_seed']
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_int,
        test_size=test_size,
        stratify=y_int,
        random_state=random_seed
    )
    
    # Second split: train vs val (from remaining data)
    # val_size relative to temp size = val_size / (train_size + val_size)
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_seed
    )
    
    # One-hot encode labels (4 classes)
    num_classes = config['head']['num_classes']
    y_train_onehot = to_categorical(y_train, num_classes=num_classes)
    y_val_onehot = to_categorical(y_val, num_classes=num_classes)
    y_test_onehot = to_categorical(y_test, num_classes=num_classes)
    
    print(f"\n{'='*60}")
    print("Data Splits")
    print(f"{'='*60}")
    print(f"  Train: {len(X_train):6d} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val):6d} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test):6d} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Label shape: {y_train_onehot.shape} (one-hot encoded)")
    
    return (X_train, y_train_onehot, 
            X_val, y_val_onehot, 
            X_test, y_test_onehot, 
            label_classes)
