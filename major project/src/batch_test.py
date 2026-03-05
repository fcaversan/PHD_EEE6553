"""Quick batch test of classify_url for various URLs."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from utils import get_config

cfg = get_config()
art = os.path.join(os.path.dirname(__file__), '..', cfg['data']['artifacts_dir'])

# Load model, tokenizer, scaler once
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
model = tf.keras.models.load_model(os.path.join(art, 'model.keras'))

import json
with open(os.path.join(art, 'tokenizer.json')) as f:
    tok_cfg = json.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tok_cfg))

with open(os.path.join(art, 'metadata.json')) as f:
    meta = json.load(f)
max_len = meta['max_sequence_length']

import joblib
scaler = joblib.load(os.path.join(art, 'scaler.pkl'))

from feature_engineering import extract_lexical_features
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

LABELS = ['benign', 'defacement', 'malware', 'phishing']

test_urls = [
    # Expected benign
    ("www.google.com", "benign"),
    ("stackoverflow.com/questions/12345/how-to-parse-xml", "benign"),
    ("amazon.co.uk/dp/B08N5WRWNW", "benign"),
    ("en.wikipedia.org/wiki/Machine_learning", "benign"),
    ("youtube.com/watch?v=dQw4w9WgXcQ", "benign"),
    ("netflix.com/browse", "benign"),
    ("github.com/microsoft/vscode", "benign"),
    ("docs.python.org/3/library/urllib.parse.html", "benign"),
    ("reddit.com/r/MachineLearning", "benign"),
    ("bbc.co.uk/news/technology", "benign"),
    # Expected phishing
    ("secure-paypal-login.suspicious-site.com/verify", "phishing"),
    ("login-microsoft365.tk/auth/signin", "phishing"),
    ("appleid.apple.com.verify-account.tk/login", "phishing"),
    ("192.168.1.1/admin/login.php", "phishing"),
    ("bit.ly/3xYz123", "phishing"),
    # Expected malware
    ("download-free-antivirus.xyz/setup.exe", "malware"),
]

print(f"{'URL':<60} {'Expected':<12} {'Predicted':<12} {'Conf':>6} {'Match'}")
print("-" * 100)

results = []
for url, expected in test_urls:
    # Tokenize
    chars = list(url)
    seq = tokenizer.texts_to_sequences([chars])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding='post')
    
    # Features
    feats = extract_lexical_features(url)
    feat_df = pd.DataFrame([feats])
    feat_scaled = scaler.transform(feat_df)
    
    # Predict
    probs = model.predict([padded, feat_scaled], verbose=0)[0]
    pred_idx = np.argmax(probs)
    pred_label = LABELS[pred_idx]
    conf = probs[pred_idx] * 100
    
    match = "✓" if pred_label == expected else "✗"
    line = f"{url:<60} {expected:<12} {pred_label:<12} {conf:5.1f}% {match}"
    print(line, flush=True)
    results.append(line)

print("-" * 100)

# Also write to file
with open(os.path.join(os.path.dirname(__file__), '..', 'batch_test_results.txt'), 'w') as f:
    f.write(f"{'URL':<60} {'Expected':<12} {'Predicted':<12} {'Conf':>6} {'Match'}\n")
    f.write("-" * 100 + "\n")
    for r in results:
        f.write(r + "\n")
    f.write("-" * 100 + "\n")
print("Results also saved to batch_test_results.txt")
