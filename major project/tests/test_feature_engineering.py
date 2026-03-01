"""
Unit tests for feature engineering module
Tests lexical feature extraction and scaler functionality
"""

import pytest
import numpy as np
import pandas as pd
from src.feature_engineering import (
    extract_lexical_features,
    extract_features_batch,
    fit_and_save_scaler,
    load_and_apply_scaler
)


def test_extract_lexical_features():
    """Test that feature extraction produces correct output structure."""
    url = "https://www.example.com/path?query=value"
    features = extract_lexical_features(url)
    
    # Should return dict with 23 features
    assert isinstance(features, dict)
    assert len(features) == 23
    
    # All values should be float
    for value in features.values():
        assert isinstance(value, float)
    
    # Check some expected features
    assert features['url_length'] == len(url)
    assert features['count_dots'] >= 2  # At least in 'www.' and '.com'
    assert features['count_question'] == 1
    assert features['count_equals'] == 1


def test_use_of_ip_detection():
    """Test IP address detection feature."""
    # URL with IP address
    ip_url = "http://192.168.1.1/login"
    features_ip = extract_lexical_features(ip_url)
    assert features_ip['use_of_ip'] == 1.0
    
    # URL without IP address
    normal_url = "https://www.example.com/login"
    features_normal = extract_lexical_features(normal_url)
    assert features_normal['use_of_ip'] == 0.0


def test_shortening_service_detection():
    """Test URL shortening service detection."""
    # URL with shortener
    short_url = "https://bit.ly/abc123"
    features_short = extract_lexical_features(short_url)
    assert features_short['shortening_service'] == 1.0
    
    # Normal URL
    normal_url = "https://www.example.com/page"
    features_normal = extract_lexical_features(normal_url)
    assert features_normal['shortening_service'] == 0.0


def test_extract_features_batch():
    """Test batch feature extraction produces correct shape and dtype."""
    urls = np.array([
        "https://www.example.com",
        "http://192.168.1.1/phishing",
        "https://bit.ly/short"
    ])
    
    features_df = extract_features_batch(urls)
    
    # Check shape
    assert features_df.shape == (3, 23)
    
    # Check dtype
    assert features_df.dtypes[0] == np.float32
    
    # Check no NaN values
    assert features_df.isnull().sum().sum() == 0


def test_scaler_save_and_load(tmp_path):
    """Test scaler can be saved and reloaded correctly."""
    # Create sample data
    train_data = pd.DataFrame(np.random.randn(100, 23), dtype=np.float32)
    test_data = pd.DataFrame(np.random.randn(10, 23), dtype=np.float32)
    
    scaler_path = tmp_path / "test_scaler.pkl"
    
    # Fit and save scaler
    scaler = fit_and_save_scaler(train_data, str(scaler_path))
    
    # Apply scaler directly
    scaled_direct = scaler.transform(test_data)
    
    # Load scaler and apply
    scaled_loaded = load_and_apply_scaler(test_data, str(scaler_path))
    
    # Results should be identical
    np.testing.assert_array_almost_equal(scaled_direct, scaled_loaded)


def test_no_nan_features():
    """Test that feature extraction never produces NaN values."""
    # Test with various edge cases
    edge_case_urls = [
        "",  # Empty string
        "http://",  # Minimal URL
        "https://a.b",  # Very short domain
        "http://192.168.1.1",  # IP only
        "http://example.com" + "a"*1000,  # Very long URL
    ]
    
    for url in edge_case_urls:
        features = extract_lexical_features(url)
        for key, value in features.items():
            assert not np.isnan(value), f"NaN found in {key} for URL: {url}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
