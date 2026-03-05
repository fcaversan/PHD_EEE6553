"""
Single-URL classifier — Phase 2
Loads all Phase 2 artifacts (model, char tokenizer, scaler, BERT tokenizer)
and classifies a single URL from the command line.
"""

import argparse
import json
import os
import sys
import numpy as np
from tensorflow import keras
from transformers import TFDistilBertModel

_SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

from utils import get_config
from feature_engineering import extract_lexical_features, load_and_apply_scaler
from url_tokenizer import url_to_words, bert_encode_urls


# ── Helpers ─────────────────────────────────────────────────────────────────

def _strip_scheme(url: str) -> str:
    for prefix in ('https://', 'http://'):
        if url.startswith(prefix):
            return url[len(prefix):]
    return url


def _load_char_tokenizer(tokenizer_path: str):
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    with open(tokenizer_path) as f:
        return tokenizer_from_json(f.read())


def _char_encode(tokenizer, url: str, max_len: int):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = tokenizer.texts_to_sequences([list(url)])
    return pad_sequences(seq, maxlen=max_len, padding='post').astype('int32')


# ── Public API ───────────────────────────────────────────────────────────────

def classify_url(url: str, verbose: bool = True) -> dict:
    """
    Classify a single URL using trained Phase 2 model and saved artifacts.

    Args:
        url:     Raw URL string to classify.
        verbose: If True, print detailed output.

    Returns:
        dict with keys: url, predicted_class, confidence, probabilities.
    """
    original_url = url
    url          = _strip_scheme(url)

    config = get_config('config.yaml')

    # ── Load metadata ────────────────────────────────────────────
    meta_path = os.path.join(config['data']['artifacts_dir'], 'metadata.json')
    with open(meta_path) as f:
        metadata = json.load(f)
    max_char_len  = metadata['max_sequence_length']
    max_bert_len  = metadata['max_bert_length']
    label_classes = metadata['label_classes']
    bert_name     = metadata['bert_model_name']

    if verbose:
        print(f"\n{'='*60}")
        print("Phase 2 — Single-URL Classification")
        print(f"{'='*60}")
        print(f"URL: {original_url}")
        if original_url != url:
            print(f"  → Scheme stripped: {url}")
        print()

    # ── Load model + tokenizers ──────────────────────────────────
    model    = keras.models.load_model(
        config['data']['model_path'],
        custom_objects={'TFDistilBertModel': TFDistilBertModel}
    )
    char_tok = _load_char_tokenizer(config['data']['char_tokenizer_path'])

    from transformers import AutoTokenizer as HFAutoTok
    bert_tok = HFAutoTok.from_pretrained(bert_name)

    # ── Preprocess ───────────────────────────────────────────────
    # Branch A — char sequence
    X_char = _char_encode(char_tok, url, max_char_len)

    # Branch B — lexical features
    feat_dict = extract_lexical_features(url)
    feat_arr  = np.array([list(feat_dict.values())], dtype=np.float32)
    X_lex     = load_and_apply_scaler(feat_arr, config['data']['scaler_path'])

    # Branch C — BERT subword tokens
    word_str        = url_to_words(url)
    ids, mask       = bert_encode_urls([word_str], bert_tok, max_bert_len)
    X_bert_ids      = ids   # shape (1, max_bert_len)
    X_bert_mask     = mask  # shape (1, max_bert_len)

    # ── Inference ────────────────────────────────────────────────
    probs     = model.predict([X_char, X_lex, X_bert_ids, X_bert_mask], verbose=0)
    pred_idx  = int(np.argmax(probs[0]))
    pred_cls  = label_classes[pred_idx]
    conf      = float(probs[0][pred_idx])

    results = {
        'url':             url,
        'predicted_class': pred_cls,
        'confidence':      conf,
        'probabilities':   {label_classes[i]: float(probs[0][i])
                            for i in range(len(label_classes))},
    }

    if verbose:
        print(f"\n{'='*60}")
        print("Classification Results")
        print(f"{'='*60}")
        print(f"Predicted Class : {pred_cls.upper()}")
        print(f"Confidence      : {conf*100:.2f}%")
        print(f"\nClass Probabilities:")
        for cls_name, p in results['probabilities'].items():
            print(f"  {cls_name:15s}: {p*100:6.2f}%")
        print(f"{'='*60}\n")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Phase 2 — Classify a URL as benign / defacement / phishing / malware',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python classify_url.py --url "http://example.com"
  python classify_url.py --url "secure-login.verify-account.com/paypal"
  python classify_url.py --url "192.168.1.1/admin/setup.php" --quiet
        '''
    )
    parser.add_argument('--url',   required=True, help='URL to classify')
    parser.add_argument('--quiet', action='store_true',
                        help='Output JSON only (no decorative text)')
    args = parser.parse_args()

    result = classify_url(args.url, verbose=not args.quiet)

    if args.quiet:
        import json as _json
        print(_json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
