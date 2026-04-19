"""
Brand Impersonation Stress Test — Phase 5 Stage 1.
Same 16 URLs used across Phases 2-4 (8 legit, 8 impersonation phishing).
Previous result: 0/8 impersonation detected in ALL prior phases.
"""

import os, sys, json
import numpy as np
from tensorflow import keras

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
os.chdir(_PROJECT_DIR)

from utils import get_config
from feature_engineering import extract_features_batch, load_and_apply_scaler
from text_processing import load_tokenizer, tokenize_and_pad

# ── The 16 stress-test URLs (same as Phase 2/3/4) ──────────
STRESS_URLS = [
    ("accounts.google.com/signin/v2/",                    "benign"),
    ("secure-login-google.com/auth/session",              "phishing"),
    ("paypal.com/myaccount/summary",                      "benign"),
    ("paypal-security-center.com/verify/account",         "phishing"),
    ("www.microsoft.com/en-us/security",                  "benign"),
    ("microsoftonline-authentication.com/login",          "phishing"),
    ("github.com/login",                                  "benign"),
    ("github-secure-auth.com/session/recover",            "phishing"),
    ("appleid.apple.com/",                                "benign"),
    ("appleid-verify-now.com/icloud/recovery",            "phishing"),
    ("www.dropbox.com/login",                             "benign"),
    ("dropbox-file-share-secure.com/open",                "phishing"),
    ("www.netflix.com/login",                             "benign"),
    ("netflix-account-security-center.com/signin",        "phishing"),
    ("portal.office.com/",                                "benign"),
    ("office365-credential-check.com/owa/auth",           "phishing"),
]


def main():
    print("\n" + "=" * 70)
    print("PHASE 5 — BRAND IMPERSONATION STRESS TEST (Stage 1 Binary)")
    print("=" * 70)

    config = get_config('config.yaml')
    s1 = config['stage1']

    # Load model
    model = keras.models.load_model(s1['model_path'])
    with open(s1['metadata_path']) as f:
        meta = json.load(f)
    max_seq = meta['max_sequence_length']
    tokenizer = load_tokenizer(s1['tokenizer_path'])

    urls = [u for u, _ in STRESS_URLS]
    tags = [t for _, t in STRESS_URLS]

    # Preprocess
    seqs = tokenize_and_pad(tokenizer, np.array(urls), max_seq)
    feats = extract_features_batch(np.array(urls))
    scaled = load_and_apply_scaler(feats, s1['scaler_path'])

    # Predict
    probs = model.predict([seqs, scaled], verbose=0).flatten()

    # Display results
    print(f"\n{'#':>2}  {'URL':<55} {'Tag':>10}  {'Pred':>10}  {'Conf':>8}  {'OK?':>4}")
    print("-" * 100)

    benign_correct = 0
    phishing_correct = 0
    benign_total = 0
    phishing_total = 0

    for i, (url, tag) in enumerate(STRESS_URLS):
        p_mal = probs[i]
        pred = "malicious" if p_mal > 0.5 else "benign"
        conf = p_mal if p_mal > 0.5 else (1 - p_mal)
        correct = (tag == "benign" and pred == "benign") or (tag == "phishing" and pred == "malicious")

        if tag == "benign":
            benign_total += 1
            if correct:
                benign_correct += 1
        else:
            phishing_total += 1
            if correct:
                phishing_correct += 1

        mark = "Y" if correct else "X"
        print(f"{i+1:>2}  {url:<55} {tag:>10}  {pred:>10}  {conf*100:>7.2f}%  {mark:>4}")

    print("-" * 100)
    print(f"\nLegitimate URLs correct:     {benign_correct}/{benign_total}")
    print(f"Impersonation phishing detected: {phishing_correct}/{phishing_total}")
    print(f"\nPrevious phases (2, 3, 4):   0/8 impersonation detected")
    print(f"Phase 5 (brand cross-attn):  {phishing_correct}/8 impersonation detected")

    if phishing_correct > 0:
        print(f"\n*** IMPROVEMENT: {phishing_correct} impersonation URLs now detected! ***")
    else:
        print(f"\n  No improvement on impersonation detection.")


if __name__ == '__main__':
    main()
