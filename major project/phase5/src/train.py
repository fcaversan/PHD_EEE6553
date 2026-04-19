"""
Training pipeline for Phase 5 — Hierarchical Two-Stage Classification.
Usage:
    python src/train.py --stage 1   # Binary: benign vs malicious
    python src/train.py --stage 2   # 3-class: defacement / malware / phishing
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

from utils import set_all_seeds, check_gpu, get_config, create_directories
from data_loader import load_dataset
from feature_engineering import extract_features_batch, fit_and_save_scaler, apply_scaler
from text_processing import process_training_urls, tokenize_and_pad
from model_builder import build_and_compile


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history['accuracy'], label='Train Acc', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Curves saved: {save_path}")
    plt.close()


def train_stage(stage: int):
    print("\n" + "=" * 60)
    print(f"PHASE 5 — STAGE {stage} TRAINING")
    if stage == 1:
        print("Binary: benign vs malicious")
    else:
        print("3-class: defacement / malware / phishing")
    print("=" * 60)

    config = get_config('config.yaml')
    set_all_seeds(config['random_seed'])
    check_gpu()
    create_directories(config)

    stage_key = f'stage{stage}'
    mode = 'binary' if stage == 1 else 'malicious_only'
    num_classes = config[stage_key]['num_classes']

    # ── Load data ────────────────────────────────────────────
    print(f"\n{'='*60}\nStep 1: Loading Data ({mode})\n{'='*60}")
    dataset_path = config['data']['dataset_path']
    X_train, y_train, X_val, y_val, X_test, y_test, label_classes = \
        load_dataset(dataset_path, config, mode=mode)

    # For binary with sigmoid, y should be (N, 1) not one-hot
    if stage == 1:
        y_train = y_train[:, 1:2]  # P(malicious)
        y_val = y_val[:, 1:2]
        y_test = y_test[:, 1:2]

    # ── Feature extraction ───────────────────────────────────
    print(f"\n{'='*60}\nStep 2: Feature Extraction (27 features)\n{'='*60}")
    train_feat = extract_features_batch(X_train)
    val_feat = extract_features_batch(X_val)
    test_feat = extract_features_batch(X_test)

    scaler_path = config[stage_key]['scaler_path']
    scaler = fit_and_save_scaler(train_feat, scaler_path)
    X_train_scaled = apply_scaler(scaler, train_feat)
    X_val_scaled = apply_scaler(scaler, val_feat)
    X_test_scaled = apply_scaler(scaler, test_feat)

    # ── Tokenization ─────────────────────────────────────────
    print(f"\n{'='*60}\nStep 3: Character Tokenization\n{'='*60}")
    tokenizer_path = config[stage_key]['tokenizer_path']
    X_train_seq, tokenizer, max_seq_len = process_training_urls(
        X_train, tokenizer_path, percentile=95)
    X_val_seq = tokenize_and_pad(tokenizer, X_val, max_seq_len)
    X_test_seq = tokenize_and_pad(tokenizer, X_test, max_seq_len)
    vocab_size = len(tokenizer.word_index) + 1

    # Save metadata
    metadata = {
        'max_sequence_length': max_seq_len,
        'vocab_size': vocab_size,
        'label_classes': label_classes,
        'n_lexical_features': int(X_train_scaled.shape[1]),
        'stage': stage,
        'num_classes': num_classes,
        'train_samples': int(len(X_train)),
    }
    meta_path = config[stage_key]['metadata_path']
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {meta_path}")

    # ── Build model ──────────────────────────────────────────
    print(f"\n{'='*60}\nStep 4: Building Model ({num_classes}-class)\n{'='*60}")
    model = build_and_compile(vocab_size, max_seq_len, num_classes, config)

    # ── Train ────────────────────────────────────────────────
    print(f"\n{'='*60}\nStep 5: Training\n{'='*60}")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=config['training']['early_stopping_patience'],
            restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=config['training']['reduce_lr_factor'],
            patience=config['training']['reduce_lr_patience'], verbose=1, min_lr=1e-7),
    ]

    history = model.fit(
        x=[X_train_seq, X_train_scaled], y=y_train,
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        validation_data=([X_val_seq, X_val_scaled], y_val),
        callbacks=callbacks, verbose=1,
    )

    # ── Save ─────────────────────────────────────────────────
    print(f"\n{'='*60}\nStep 6: Saving Artifacts\n{'='*60}")
    model_path = config[stage_key]['model_path']
    model.save(model_path)
    print(f"  Model saved: {model_path}")

    curves_path = os.path.join(config['data']['results_dir'],
                               f'stage{stage}_training_curves.png')
    plot_training_history(history, curves_path)

    # ── Quick validation ─────────────────────────────────────
    val_results = model.evaluate([X_val_seq, X_val_scaled], y_val, verbose=0)
    print(f"\n  Validation results: {dict(zip(model.metrics_names, val_results))}")

    print(f"\n✓ Stage {stage} training complete.\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2],
                        help='1 = binary (benign vs malicious), 2 = malicious sub-class')
    args = parser.parse_args()
    train_stage(args.stage)
