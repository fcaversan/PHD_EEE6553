"""
Feature engineering — Phase 2
Identical 23 lexical features as Phase 1 (Branch B is unchanged).
"""

import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import joblib
from sklearn.preprocessing import StandardScaler


SHORTENER_PATTERNS = [
    r'bit\.ly', r'goo\.gl', r'tinyurl\.com', r'ow\.ly', r't\.co',
    r'short\.link', r'tiny\.cc', r'is\.gd', r'buff\.ly', r'adf\.ly'
]


def extract_lexical_features(url: str) -> dict:
    """Extract 23 lexical heuristic features from a single URL (unchanged from Phase 1)."""
    try:
        parse_url = url if url.startswith(('http://', 'https://')) else 'http://' + url
        parsed = urlparse(parse_url)
        hostname = parsed.netloc
        path = parsed.path
    except Exception:
        hostname = ''
        path = ''

    features = {}
    features['url_length']        = float(len(url))
    features['hostname_length']   = float(len(hostname))
    features['path_length']       = float(len(path))
    features['count_dots']        = float(url.count('.'))
    features['count_hyphens']     = float(url.count('-'))
    features['count_at']          = float(url.count('@'))
    features['count_question']    = float(url.count('?'))
    features['count_ampersand']   = float(url.count('&'))
    features['count_equals']      = float(url.count('='))
    features['count_underscore']  = float(url.count('_'))
    features['count_tilde']       = float(url.count('~'))
    features['count_percent']     = float(url.count('%'))
    features['count_asterisk']    = float(url.count('*'))
    features['count_colon']       = float(url.count(':'))
    features['count_www']         = float(url.lower().count('www'))
    features['count_https']       = float(url.lower().count('https'))
    features['count_http']        = float(url.lower().count('http'))
    features['count_digits']      = float(sum(c.isdigit() for c in url))
    features['count_letters']     = float(sum(c.isalpha() for c in url))
    features['count_directories'] = float(path.count('/'))
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    features['use_of_ip']         = float(1 if re.search(ip_pattern, url) else 0)
    is_shortener = any(re.search(p, url, re.IGNORECASE) for p in SHORTENER_PATTERNS)
    features['shortening_service'] = float(1 if is_shortener else 0)
    features['reserved_feature']  = float(0)
    return features


def extract_features_batch(urls: np.ndarray) -> pd.DataFrame:
    """Extract lexical features for a batch of URLs."""
    print(f"\nExtracting lexical features from {len(urls)} URLs...")
    features_list = [extract_lexical_features(url) for url in urls]
    df = pd.DataFrame(features_list)
    if df.shape[1] != 23:
        raise ValueError(f"Expected 23 features, got {df.shape[1]}")
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        raise ValueError(f"Feature extraction produced {nan_count} NaN values")
    df = df.astype(np.float32)
    print(f"✓ Extracted features: {df.shape}")
    print(f"  Dtype: {df.dtypes.unique()}")
    print(f"  NaN count: {nan_count}")
    return df


def fit_and_save_scaler(train_features: pd.DataFrame, scaler_path: str) -> StandardScaler:
    """Fit StandardScaler on training features and save to disk."""
    print(f"\nFitting StandardScaler on training features...")
    scaler = StandardScaler()
    scaler.fit(train_features)
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler fitted and saved to: {scaler_path}")
    print(f"  Feature means: min={scaler.mean_.min():.2f}, max={scaler.mean_.max():.2f}")
    print(f"  Feature stds: min={scaler.scale_.min():.2f}, max={scaler.scale_.max():.2f}")
    return scaler


def load_and_apply_scaler(features: pd.DataFrame, scaler_path: str) -> np.ndarray:
    """Load fitted scaler from disk and apply to features."""
    print(f"\nLoading scaler from: {scaler_path}")
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Scaler not found: {scaler_path}\nRun train.py first.")
    return scaler.transform(features).astype(np.float32)


def apply_scaler(scaler: StandardScaler, features: pd.DataFrame) -> np.ndarray:
    """Apply an already-loaded scaler to features."""
    return scaler.transform(features).astype(np.float32)
