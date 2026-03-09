"""
Training pipeline for Phase 4 Malicious URL Detection Model
Enhanced training data: Kaggle + PhishTank + Synthetic impersonation URLs
Same architecture as Phase 3 (Char CNN-BiGRU-Attention + Brand-Aware MLP)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Ensure src/ is on the path and working directory is the project root
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

# Import local modules
from utils import set_all_seeds, check_gpu, get_config, create_directories
from data_loader import load_malicious_urls_dataset
from feature_engineering import extract_features_batch, fit_and_save_scaler, apply_scaler
from text_processing import process_training_urls, tokenize_and_pad
from model_builder import build_and_compile_model


def plot_training_history(history: keras.callbacks.History, save_path: str) -> None:
    """Plot and save training curves (loss and accuracy)."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Training curves saved to: {save_path}")
    plt.close()


def train_model():
    """
    Complete training pipeline:
    1. Set random seeds and check GPU
    2. Load merged dataset and split
    3. Extract and scale 27 lexical features (Branch B)
    4. Tokenize and pad URL sequences (Branch A)
    5. Build dual-input model
    6. Train with callbacks
    7. Save artifacts
    8. Plot training curves
    """

    print("\n" + "=" * 60)
    print("PHASE 4 — ENHANCED DATA TRAINING PIPELINE")
    print("=" * 60)

    # Load configuration
    config = get_config('config.yaml')

    # Set random seeds for reproducibility
    set_all_seeds(config['random_seed'])

    # Check GPU availability
    check_gpu()

    # Create output directories
    create_directories(config)

    # ========== Load and split data ==========
    print(f"\n{'=' * 60}")
    print("Step 1: Data Loading (merged dataset)")
    print(f"{'=' * 60}")

    dataset_path = config['data']['dataset_path']
    X_train, y_train, X_val, y_val, X_test, y_test, label_classes = \
        load_malicious_urls_dataset(dataset_path, config)

    # ========== Extract lexical features (Branch B) ==========
    print(f"\n{'=' * 60}")
    print("Step 2: Lexical Feature Extraction (27 features)")
    print(f"{'=' * 60}")

    train_features = extract_features_batch(X_train)
    val_features = extract_features_batch(X_val)
    test_features = extract_features_batch(X_test)

    scaler_path = config['data']['scaler_path']
    scaler = fit_and_save_scaler(train_features, scaler_path)

    X_train_scaled = apply_scaler(scaler, train_features)
    X_val_scaled = apply_scaler(scaler, val_features)
    X_test_scaled = apply_scaler(scaler, test_features)

    # ========== Tokenize URL sequences (Branch A) ==========
    print(f"\n{'=' * 60}")
    print("Step 3: URL Sequence Processing")
    print(f"{'=' * 60}")

    tokenizer_path = config['data']['tokenizer_path']

    X_train_sequences, tokenizer, max_sequence_length = \
        process_training_urls(X_train, tokenizer_path, percentile=95)

    X_val_sequences = tokenize_and_pad(tokenizer, X_val, max_sequence_length)
    X_test_sequences = tokenize_and_pad(tokenizer, X_test, max_sequence_length)

    vocab_size = len(tokenizer.word_index) + 1

    # Save metadata for inference
    import json
    metadata = {
        'max_sequence_length': max_sequence_length,
        'vocab_size': vocab_size,
        'label_classes': label_classes,
        'n_lexical_features': int(X_train_scaled.shape[1]),
        'phase': 4,
        'dataset': 'merged (kaggle + synthetic)',
        'train_samples': int(len(X_train)),
    }
    metadata_path = os.path.join(config['data']['artifacts_dir'], 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to: {metadata_path}")

    # ========== Build model ==========
    print(f"\n{'=' * 60}")
    print("Step 4: Model Construction")
    print(f"{'=' * 60}")

    model = build_and_compile_model(vocab_size, max_sequence_length, config)

    # ========== Train model ==========
    print(f"\n{'=' * 60}")
    print("Step 5: Model Training")
    print(f"{'=' * 60}")

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config['training']['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config['training']['reduce_lr_factor'],
        patience=config['training']['reduce_lr_patience'],
        verbose=1,
        min_lr=1e-7
    )

    print(f"\nStarting training...")
    history = model.fit(
        x=[X_train_sequences, X_train_scaled],
        y=y_train,
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        validation_data=([X_val_sequences, X_val_scaled], y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # ========== Save model and artifacts ==========
    print(f"\n{'=' * 60}")
    print("Step 6: Saving Artifacts")
    print(f"{'=' * 60}")

    model_path = config['data']['model_path']
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    curves_path = os.path.join(config['data']['results_dir'], 'training_curves.png')
    plot_training_history(history, curves_path)

    # ========== Evaluate on validation set ==========
    print(f"\n{'=' * 60}")
    print("Step 7: Validation Results")
    print(f"{'=' * 60}")

    val_results = model.evaluate(
        x=[X_val_sequences, X_val_scaled],
        y=y_val,
        verbose=0
    )

    print(f"  Validation Loss:      {val_results[0]:.4f}")
    print(f"  Validation Accuracy:  {val_results[1]:.4f}")
    print(f"  Validation Precision: {val_results[2]:.4f}")
    print(f"  Validation Recall:    {val_results[3]:.4f}")

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"\nArtifacts saved:")
    print(f"  - Model:      {model_path}")
    print(f"  - Scaler:     {scaler_path}")
    print(f"  - Tokenizer:  {tokenizer_path}")
    print(f"  - Metadata:   {metadata_path}")
    print(f"  - Curves:     {curves_path}")
    print(f"\nNext step: python src/evaluate.py")


if __name__ == '__main__':
    train_model()
