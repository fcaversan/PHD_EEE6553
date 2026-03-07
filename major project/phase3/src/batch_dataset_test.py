"""Batch test: sample 10 URLs per class from dataset and classify."""
import os, sys, json, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from utils import get_config
from feature_engineering import extract_lexical_features
import joblib

cfg = get_config()
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
art = os.path.join(base, cfg['data']['artifacts_dir'])

# Load artifacts once
model = tf.keras.models.load_model(os.path.join(art, 'model.keras'))
with open(os.path.join(art, 'tokenizer.json')) as f:
    tok_cfg = json.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tok_cfg))
with open(os.path.join(art, 'metadata.json')) as f:
    meta = json.load(f)
max_len = meta['max_sequence_length']
scaler = joblib.load(os.path.join(art, 'scaler.pkl'))

LABELS = ['benign', 'defacement', 'malware', 'phishing']

# Load dataset, sample 10 per class
ds_path = os.path.join(base, cfg['data']['dataset_path'])
df = pd.read_csv(ds_path)
samples = df.groupby('type').apply(lambda x: x.sample(10, random_state=99)).reset_index(drop=True)


def classify(url):
    chars = list(url)
    seq = tokenizer.texts_to_sequences([chars])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=max_len, padding='post'
    )
    feats = extract_lexical_features(url)
    feat_df = pd.DataFrame([feats])
    feat_scaled = scaler.transform(feat_df)
    probs = model.predict([padded, feat_scaled], verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return LABELS[pred_idx], probs[pred_idx] * 100


header = f"{'URL':<75} {'True':<12} {'Pred':<12} {'Conf':>6} OK"
sep = "-" * 115
print(header)
print(sep)

correct = 0
total = 0
class_correct = {l: 0 for l in LABELS}
class_total = {l: 0 for l in LABELS}

for _, row in samples.iterrows():
    url = str(row['url'])
    true_label = row['type']
    pred_label, conf = classify(url)

    ok = "Y" if pred_label == true_label else "N"
    if pred_label == true_label:
        correct += 1
        class_correct[true_label] += 1
    total += 1
    class_total[true_label] += 1

    disp = url[:72] + "..." if len(url) > 75 else url
    print(f"{disp:<75} {true_label:<12} {pred_label:<12} {conf:5.1f}% {ok}")

print(sep)
print(f"Overall: {correct}/{total} = {correct/total*100:.1f}%")
for l in LABELS:
    print(f"  {l:<12}: {class_correct[l]}/{class_total[l]}")
