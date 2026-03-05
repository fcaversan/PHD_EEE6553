"""
Feature engineering for Malicious URL Detection Model
Extracts 23 deterministic lexical heuristic features from raw URLs
"""

import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import joblib
from sklearn.preprocessing import StandardScaler


# Known URL shortening services (common patterns)
SHORTENER_PATTERNS = [
    r'bit\.ly', r'goo\.gl', r'tinyurl\.com', r'ow\.ly', r't\.co',
    r'short\.link', r'tiny\.cc', r'is\.gd', r'buff\.ly', r'adf\.ly'
]


def extract_lexical_features(url: str) -> dict:
    """
    Extract all 23 lexical heuristic features from a single URL.
    
    Args:
        url: Raw URL string
    
    Returns:
        dict: Dictionary with 23 feature names and their float32 values
    """
    
    # Parse URL components.
    # Many URLs in the dataset lack a scheme (e.g. "github.com/path").
    # urlparse treats those as an opaque path (netloc=''), giving
    # hostname_length=0 and path_length=full URL.  Temporarily
    # prepending "http://" ensures correct hostname/path splitting
    # without altering the raw URL used for character tokenisation.
    try:
        parse_url = url if url.startswith(('http://', 'https://')) else 'http://' + url
        parsed = urlparse(parse_url)
        hostname = parsed.netloc
        path = parsed.path
    except Exception:
        # If URL parsing fails, use empty strings
        hostname = ''
        path = ''
    
    features = {}
    
    # Length features (1-3)
    features['url_length'] = float(len(url))
    features['hostname_length'] = float(len(hostname))
    features['path_length'] = float(len(path))
    
    # Character count features (4-14)
    features['count_dots'] = float(url.count('.'))
    features['count_hyphens'] = float(url.count('-'))
    features['count_at'] = float(url.count('@'))
    features['count_question'] = float(url.count('?'))
    features['count_ampersand'] = float(url.count('&'))
    features['count_equals'] = float(url.count('='))
    features['count_underscore'] = float(url.count('_'))
    features['count_tilde'] = float(url.count('~'))
    features['count_percent'] = float(url.count('%'))
    features['count_asterisk'] = float(url.count('*'))
    features['count_colon'] = float(url.count(':'))
    
    # Substring count features (15-17)
    features['count_www'] = float(url.lower().count('www'))
    features['count_https'] = float(url.lower().count('https'))
    features['count_http'] = float(url.lower().count('http'))
    
    # Character type count features (18-19)
    features['count_digits'] = float(sum(c.isdigit() for c in url))
    features['count_letters'] = float(sum(c.isalpha() for c in url))
    
    # Path directory count (20)
    features['count_directories'] = float(path.count('/'))
    
    # Boolean features: IP address detection (21)
    # IPv4 pattern: xxx.xxx.xxx.xxx
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    features['use_of_ip'] = float(1 if re.search(ip_pattern, url) else 0)
    
    # Boolean features: URL shortening service detection (22)
    is_shortener = any(re.search(pattern, url, re.IGNORECASE) 
                      for pattern in SHORTENER_PATTERNS)
    features['shortening_service'] = float(1 if is_shortener else 0)
    
    # Reserved feature slot (23) - placeholder for future expansion
    features['reserved_feature'] = float(0)
    
    return features


def extract_features_batch(urls: np.ndarray) -> pd.DataFrame:
    """
    Extract lexical features for a batch of URLs.
    
    Args:
        urls: NumPy array of URL strings (N,)
    
    Returns:
        pd.DataFrame: Feature matrix (N, 23) with float32 dtype
    
    Raises:
        ValueError: If any extracted features contain NaN values
    """
    
    print(f"\nExtracting lexical features from {len(urls)} URLs...")
    
    # Extract features for all URLs
    features_list = [extract_lexical_features(url) for url in urls]
    
    # Convert to DataFrame
    df_features = pd.DataFrame(features_list)
    
    # Validate output
    if df_features.shape[1] != 23:
        raise ValueError(
            f"Expected 23 features, got {df_features.shape[1]}. "
            f"Columns: {df_features.columns.tolist()}"
        )
    
    # Check for NaN values
    nan_count = df_features.isnull().sum().sum()
    if nan_count > 0:
        raise ValueError(
            f"Feature extraction produced {nan_count} NaN values. "
            f"All features must be deterministic and complete."
        )
    
    # Cast to float32 for memory efficiency
    df_features = df_features.astype(np.float32)
    
    print(f"✓ Extracted features: {df_features.shape}")
    print(f"  Dtype: {df_features.dtypes.unique()}")
    print(f"  NaN count: {nan_count}")
    
    return df_features


def fit_and_save_scaler(train_features: pd.DataFrame, scaler_path: str) -> StandardScaler:
    """
    Fit StandardScaler on training features and save to disk.
    
    Args:
        train_features: Training feature DataFrame (N, 23)
        scaler_path: Path to save the fitted scaler (.pkl)
    
    Returns:
        StandardScaler: Fitted scaler object
    """
    
    print(f"\nFitting StandardScaler on training features...")
    
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler fitted and saved to: {scaler_path}")
    print(f"  Feature means: min={scaler.mean_.min():.2f}, max={scaler.mean_.max():.2f}")
    print(f"  Feature stds: min={scaler.scale_.min():.2f}, max={scaler.scale_.max():.2f}")
    
    return scaler


def load_and_apply_scaler(features: pd.DataFrame, scaler_path: str) -> np.ndarray:
    """
    Load fitted scaler from disk and apply to features.
    
    Args:
        features: Feature DataFrame (N, 23)
        scaler_path: Path to saved scaler (.pkl)
    
    Returns:
        np.ndarray: Scaled features (N, 23) as float32
    
    Raises:
        FileNotFoundError: If scaler file doesn't exist
    """
    
    print(f"\nLoading scaler from: {scaler_path}")
    
    try:
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            f"Please train the model first to generate the scaler."
        )
    
    # Apply scaling
    scaled_features = scaler.transform(features)
    
    print(f"✓ Features scaled using saved scaler")
    
    return scaled_features.astype(np.float32)


def apply_scaler(scaler: StandardScaler, features: pd.DataFrame) -> np.ndarray:
    """
    Apply an already-loaded scaler to features.
    
    Args:
        scaler: Fitted StandardScaler object
        features: Feature DataFrame (N, 23)
    
    Returns:
        np.ndarray: Scaled features (N, 23) as float32
    """
    
    scaled_features = scaler.transform(features)
    return scaled_features.astype(np.float32)
