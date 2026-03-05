"""
Training pipeline — Phase 2
Extends Phase 1 pipeline with Branch C (DistilBERT semantic encoding).
Produces a triple-input model: char sequence + lexical features + BERT subword tokens.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

from utils import set_all_seeds, check_gpu, get_config
from data_loader import load_malicious_urls_dataset
from feature_engineering import extract_features_batch, fit_and_save_scaler, apply_scaler
from url_tokenizer import batch_url_to_words, bert_encode_urls
from model_builder import build_and_compile_model


# ── Helpers ─────────────────────────────────────────────────────────────────

def _ensure_dirs(config: dict) -> None:
    os.makedirs(config['data']['artifacts_dir'], exist_ok=True)
    os.makedirs(config['data']['results_dir'],    exist_ok=True)


def _init_char_tokenizer(urls, tokenizer_path: str, percentile: int = 95):
    """Fit a character-level Keras tokenizer on training URLs."""
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    char_lists = [list(u) for u in urls]
    tokenizer = Tokenizer(char_level=False, oov_token='<UNK>')
    tokenizer.fit_on_texts(char_lists)

    lengths = [len(u) for u in urls]
    max_len = int(np.percentile(lengths, percentile))

    # Tokenize & pad
    seqs   = tokenizer.texts_to_sequences(char_lists)
    padded = pad_sequences(seqs, maxlen=max_len, padding='post').astype('int32')

    # Persist tokenizer
    with open(tokenizer_path, 'w') as f:
        f.write(tokenizer.to_json())
    print(f"✓ Char tokenizer fitted and saved to: {tokenizer_path}")
    print(f"  Vocabulary size: {len(tokenizer.word_index) + 1} unique characters")
    print(f"  URL length stats:  mean={np.mean(lengths):.1f}  "
          f"95th pct={max_len}")

    return padded, tokenizer, max_len


def _apply_char_tokenizer(tokenizer, urls, max_len: int):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    char_lists = [list(u) for u in urls]
    seqs = tokenizer.texts_to_sequences(char_lists)
    return pad_sequences(seqs, maxlen=max_len, padding='post').astype('int32')


def _plot_history(history, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'],     label='Train Loss', lw=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss',   lw=2)
    axes[0].set(xlabel='Epoch', ylabel='Loss',
                title='Training and Validation Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['accuracy'],     label='Train Accuracy', lw=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy',   lw=2)
    axes[1].set(xlabel='Epoch', ylabel='Accuracy',
                title='Training and Validation Accuracy')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to: {save_path}")
    plt.close()


# ── Main pipeline ────────────────────────────────────────────────────────────

def train_model():
    print("\n" + "="*60)
    print("PHASE 2 — TRIPLE-INPUT URL DETECTION — TRAINING")
    print("="*60)

    config = get_config('config.yaml')
    set_all_seeds(config['random_seed'])
    check_gpu()
    _ensure_dirs(config)

    # ── 1. Load data ────────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 1: Data Loading\n{'='*60}")
    dataset_path = config['data']['dataset_path']
    X_train, y_train, X_val, y_val, X_test, y_test, label_classes = \
        load_malicious_urls_dataset(dataset_path, config)

    # ── 2. Lexical features (Branch B) ──────────────────────────
    print(f"\n{'='*60}\nPhase 2: Lexical Feature Extraction\n{'='*60}")
    train_lex = extract_features_batch(X_train)
    val_lex   = extract_features_batch(X_val)
    test_lex  = extract_features_batch(X_test)

    scaler       = fit_and_save_scaler(train_lex, config['data']['scaler_path'])
    train_lex_sc = apply_scaler(scaler, train_lex)
    val_lex_sc   = apply_scaler(scaler, val_lex)
    test_lex_sc  = apply_scaler(scaler, test_lex)

    # ── 3. Char tokenization (Branch A) ─────────────────────────
    print(f"\n{'='*60}\nPhase 3: Character Sequence Processing\n{'='*60}")
    char_tok_path = config['data']['char_tokenizer_path']
    X_train_char, char_tok, max_char_len = _init_char_tokenizer(
        X_train, char_tok_path, percentile=95
    )
    X_val_char  = _apply_char_tokenizer(char_tok, X_val,  max_char_len)
    X_test_char = _apply_char_tokenizer(char_tok, X_test, max_char_len)
    vocab_size = len(char_tok.word_index) + 1

    # ── 4. BERT subword encoding (Branch C) ─────────────────────
    print(f"\n{'='*60}\nPhase 4: BERT Subword Encoding\n{'='*60}")
    from transformers import AutoTokenizer
    bert_name    = config['branch_c']['model_name']
    max_bert_len = config['branch_c']['max_bert_length']
    print(f"Loading BERT tokenizer: {bert_name}  (max_length={max_bert_len})")
    bert_tok = AutoTokenizer.from_pretrained(bert_name)

    print("Encoding train URLs ...")
    train_words = batch_url_to_words(X_train, verbose=True)
    X_train_ids, X_train_mask = bert_encode_urls(train_words, bert_tok, max_bert_len)

    print("Encoding val URLs ...")
    val_words = batch_url_to_words(X_val)
    X_val_ids, X_val_mask = bert_encode_urls(val_words, bert_tok, max_bert_len)

    print("Encoding test URLs ...")
    test_words = batch_url_to_words(X_test)
    X_test_ids, X_test_mask = bert_encode_urls(test_words, bert_tok, max_bert_len)
    print("✓ BERT encoding complete")

    # ── 5. Save metadata ────────────────────────────────────────
    metadata = {
        'max_sequence_length': max_char_len,
        'max_bert_length':     max_bert_len,
        'vocab_size':          vocab_size,
        'bert_model_name':     bert_name,
        'label_classes':       label_classes,
        'phase':               2,
    }
    meta_path = config['data']['metadata_path']
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {meta_path}")

    # ── 6. Build & compile model ─────────────────────────────────
    print(f"\n{'='*60}\nPhase 5: Model Construction\n{'='*60}")
    model = build_and_compile_model(vocab_size, max_char_len, max_bert_len, config)

    # ── 7. Train ─────────────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 6: Training\n{'='*60}")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training']['reduce_lr_factor'],
            patience=config['training']['reduce_lr_patience'],
            verbose=1, min_lr=1e-7
        ),
    ]

    history = model.fit(
        x=[X_train_char, train_lex_sc, X_train_ids, X_train_mask],
        y=y_train,
        batch_size=config['training']['batch_size'],
        epochs=config['training']['epochs'],
        validation_data=(
            [X_val_char, val_lex_sc, X_val_ids, X_val_mask], y_val
        ),
        callbacks=callbacks,
        verbose=1,
    )

    # ── 8. Save artifacts ────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 7: Saving Artifacts\n{'='*60}")
    model.save(config['data']['model_path'])
    print(f"✓ Model saved to: {config['data']['model_path']}")
    _plot_history(history, os.path.join(config['data']['results_dir'],
                                         'training_curves.png'))

    # Val summary
    val_metrics = model.evaluate(
        [X_val_char, val_lex_sc, X_val_ids, X_val_mask], y_val, verbose=0
    )
    print(f"\n{'='*60}\nPhase 8: Validation Results\n{'='*60}")
    for name, val in zip(model.metrics_names, val_metrics):
        print(f"  {name}: {val:.4f}")

    print(f"\n{'='*60}\nTRAINING COMPLETE!\n{'='*60}")
    print("Next step: run evaluate.py")


if __name__ == '__main__':
    train_model()
