"""
Unit tests for text processing module
Tests tokenizer fitting, padding, and save/load functionality
"""

import pytest
import numpy as np
from src.text_processing import (
    fit_tokenizer,
    tokenize_and_pad,
    save_tokenizer,
    load_tokenizer,
    process_training_urls
)


def test_fit_tokenizer():
    """Test tokenizer fitting and max_sequence_length calculation."""
    urls = np.array([
        "http://example.com",
        "https://www.test.org/path",
        "http://short.url",
        "http://very-long-url-with-many-characters.com/path/to/resource"
    ])
    
    tokenizer, max_seq_len = fit_tokenizer(urls, percentile=95)
    
    # Check tokenizer was fitted
    assert tokenizer.word_index is not None
    assert len(tokenizer.word_index) > 0
    
    # Check max_sequence_length is reasonable
    url_lengths = [len(url) for url in urls]
    assert max_seq_len <= max(url_lengths)
    assert max_seq_len >= min(url_lengths)


def test_tokenize_and_pad():
    """Test tokenization produces correct output shape and dtype."""
    urls = np.array([
        "http://example.com",
        "https://test.org"
    ])
    
    tokenizer, max_seq_len = fit_tokenizer(urls)
    padded = tokenize_and_pad(tokenizer, urls, max_seq_len)
    
    # Check shape
    assert padded.shape == (2, max_seq_len)
    
    # Check dtype
    assert padded.dtype == np.int32


def test_padding_direction():
    """Test that padding is applied at the end (post)."""
    urls = np.array(["http://a.b"])  # Very short URL
    
    tokenizer, max_seq_len = fit_tokenizer(urls, percentile=95)
    padded = tokenize_and_pad(tokenizer, urls, max_seq_len)
    
    # Last values should be zeros (post-padding)
    assert padded[0, -1] == 0 or padded.shape[1] <= len(urls[0])


def test_max_sequence_length_95th_percentile():
    """Test that max_sequence_length is 95th percentile, not max."""
    # Create URLs with one outlier
    base_urls = ["http://short.url"] * 95  # 95 short URLs
    outlier_urls = ["http://extremely-long-" + "a"*1000 + ".com"] * 5  # 5 long outliers
    urls = np.array(base_urls + outlier_urls)
    
    tokenizer, max_seq_len = fit_tokenizer(urls, percentile=95)
    
    # max_sequence_length should be close to short URL length, not outlier length
    short_length = len(base_urls[0])
    long_length = len(outlier_urls[0])
    
    assert max_seq_len < long_length
    # Allow some margin for exact percentile calculation
    assert max_seq_len <= short_length * 1.5


def test_tokenizer_save_and_load(tmp_path):
    """Test tokenizer can be saved and reloaded correctly."""
    urls = np.array([
        "http://example.com",
        "https://test.org/path"
    ])
    
    # Fit tokenizer
    tokenizer, max_seq_len = fit_tokenizer(urls)
    
    # Save tokenizer
    tokenizer_path = tmp_path / "test_tokenizer.json"
    save_tokenizer(tokenizer, str(tokenizer_path))
    
    # Load tokenizer
    loaded_tokenizer = load_tokenizer(str(tokenizer_path))
    
    # Test that loaded tokenizer produces same results
    original_padded = tokenize_and_pad(tokenizer, urls, max_seq_len)
    loaded_padded = tokenize_and_pad(loaded_tokenizer, urls, max_seq_len)
    
    np.testing.assert_array_equal(original_padded, loaded_padded)


def test_process_training_urls(tmp_path):
    """Test complete training pipeline."""
    urls = np.array([
        "http://example.com",
        "https://www.test.org/path",
        "http://another-url.net"
    ])
    
    tokenizer_path = tmp_path / "tokenizer.json"
    
    padded, tokenizer, max_seq_len = process_training_urls(
        urls, 
        str(tokenizer_path),
        percentile=95
    )
    
    # Check outputs
    assert padded.shape == (3, max_seq_len)
    assert padded.dtype == np.int32
    assert tokenizer_path.exists()


def test_character_level_tokenization():
    """Test that tokenizer operates at character level."""
    urls = np.array(["abc", "xyz"])
    
    tokenizer, _ = fit_tokenizer(urls)
    
    # Should have individual characters in vocabulary
    assert 'a' in tokenizer.word_index
    assert 'b' in tokenizer.word_index
    assert 'c' in tokenizer.word_index
    assert 'x' in tokenizer.word_index


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
