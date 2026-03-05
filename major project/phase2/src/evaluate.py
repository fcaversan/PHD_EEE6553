"""
Evaluation pipeline — Phase 2
Extends Phase 1 evaluate.py for the triple-input model (char + lexical + BERT).
Generates classification report, confusion matrix, and Phishing-vs-Benign analysis.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.patches import Rectangle
from tensorflow import keras
from transformers import TFDistilBertModel

_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

from utils import set_all_seeds, get_config
from data_loader import load_malicious_urls_dataset
from feature_engineering import extract_features_batch, load_and_apply_scaler
from url_tokenizer import batch_url_to_words, bert_encode_urls


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_char_tokenizer(tokenizer_path: str):
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    with open(tokenizer_path, 'r') as f:
        return tokenizer_from_json(f.read())


def _apply_char_tokenizer(tokenizer, urls, max_len: int):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seqs = tokenizer.texts_to_sequences([list(u) for u in urls])
    return pad_sequences(seqs, maxlen=max_len, padding='post').astype('int32')


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray'
    )
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label',      fontsize=13, fontweight='bold')
    plt.title('Confusion Matrix — Phase 2 Malicious URL Detection',
              fontsize=15, fontweight='bold', pad=20)

    if 'phishing' in class_names and 'benign' in class_names:
        try:
            phi_idx = class_names.index('phishing')
            ben_idx = class_names.index('benign')
            ax  = plt.gca()
            ax.add_patch(Rectangle((ben_idx, phi_idx), 1, 1,
                                    fill=False, edgecolor='red', linewidth=3))
            miscount = cm[phi_idx, ben_idx]
            total    = cm[phi_idx].sum()
            rate     = (miscount / total * 100) if total else 0
            plt.text(0.5, -0.15,
                     f'Phishing→Benign: {miscount:,} ({rate:.2f}%)',
                     transform=ax.transAxes, ha='center', fontsize=11,
                     color='red', fontweight='bold')
        except (ValueError, IndexError):
            pass

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    plt.close()


# ── Main evaluation ──────────────────────────────────────────────────────────

def evaluate_model():
    print("\n" + "="*60)
    print("PHASE 2 — TRIPLE-INPUT URL DETECTION — EVALUATION")
    print("="*60)

    config = get_config('config.yaml')
    set_all_seeds(config['random_seed'])
    results_dir  = config['data']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # ── 1. Load artifacts ────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 1: Loading Artifacts\n{'='*60}")
    meta_path = os.path.join(config['data']['artifacts_dir'], 'metadata.json')
    with open(meta_path) as f:
        metadata = json.load(f)
    max_char_len  = metadata['max_sequence_length']
    max_bert_len  = metadata['max_bert_length']
    label_classes = metadata['label_classes']
    bert_name     = metadata['bert_model_name']
    print(f"✓ Metadata loaded  max_char={max_char_len}  max_bert={max_bert_len}")

    model = keras.models.load_model(
        config['data']['model_path'],
        custom_objects={'TFDistilBertModel': TFDistilBertModel}
    )
    print(f"✓ Model loaded from: {config['data']['model_path']}")

    char_tok = _load_char_tokenizer(config['data']['char_tokenizer_path'])
    print(f"✓ Char tokenizer loaded")

    # ── 2. Load test data ────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 2: Loading Test Data\n{'='*60}")
    _, _, _, _, X_test, y_test, _ = load_malicious_urls_dataset(
        config['data']['dataset_path'], config
    )

    # ── 3. Preprocess ────────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 3: Preprocessing\n{'='*60}")
    # Branch A
    X_test_char = _apply_char_tokenizer(char_tok, X_test, max_char_len)
    # Branch B
    test_lex    = extract_features_batch(X_test)
    X_test_lex  = load_and_apply_scaler(test_lex, config['data']['scaler_path'])
    # Branch C
    from transformers import AutoTokenizer as HFAutoTokenizer
    bert_tok    = HFAutoTokenizer.from_pretrained(bert_name)
    print("Encoding test URLs for BERT ...")
    test_words  = batch_url_to_words(X_test, verbose=True)
    X_test_ids, X_test_mask = bert_encode_urls(test_words, bert_tok, max_bert_len)
    print("✓ Preprocessing complete")

    # ── 4. Generate predictions ──────────────────────────────────
    print(f"\n{'='*60}\nPhase 4: Inference\n{'='*60}")
    print(f"Running on {len(X_test):,} test samples ...")
    y_prob = model.predict(
        [X_test_char, X_test_lex, X_test_ids, X_test_mask], verbose=1
    )
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(y_test,  axis=1)
    print("✓ Predictions generated")

    # ── 5. Metrics ───────────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 5: Metrics\n{'='*60}")
    report = classification_report(y_true, y_pred,
                                   target_names=label_classes, digits=4)
    print("\n" + report)

    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("PHASE 2 — TRIPLE-INPUT MODEL — CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write(f"\nTest samples: {len(X_test)}\n")
        f.write(f"Classes:      {label_classes}\n")
    print(f"✓ Classification report saved to: {report_path}")

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, label_classes,
                          os.path.join(results_dir, 'confusion_matrix.png'))

    # ── 6. Phishing misclassification analysis ───────────────────
    if 'phishing' in label_classes:
        phi = label_classes.index('phishing')
        ben = label_classes.index('benign') if 'benign' in label_classes else None
        total_phi = (y_true == phi).sum()
        correct   = (( y_true == phi) & (y_pred == phi)).sum()
        phi_f1    = 2*cm[phi,phi] / (cm[phi].sum() + cm[:,phi].sum()) if total_phi else 0
        print(f"\nPhishing Summary:")
        print(f"  Total phishing test samples : {total_phi:,}")
        print(f"  Correctly classified        : {correct:,} ({correct/total_phi*100:.2f}%)")
        if ben is not None:
            missed = cm[phi, ben]
            print(f"  Misclassified as benign     : {missed:,} ({missed/total_phi*100:.2f}%)")
        print(f"  Phishing F1                 : {phi_f1:.4f}")

    print(f"\n{'='*60}\nEVALUATION COMPLETE\n{'='*60}")


if __name__ == '__main__':
    evaluate_model()
