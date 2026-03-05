"""
Tests for url_tokenizer.py — Phase 2
Verifies url_to_words() and bert_encode_urls() with known examples.
"""

import sys
import os
import numpy as np
import pytest

# Ensure src/ is importable
_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, _SRC_DIR)

from url_tokenizer import url_to_words, batch_url_to_words, bert_encode_urls


# ── url_to_words ─────────────────────────────────────────────────────────────

class TestUrlToWords:

    def test_strips_https_scheme(self):
        result = url_to_words('https://example.com/login')
        assert 'https' not in result and 'http' not in result

    def test_strips_http_scheme(self):
        result = url_to_words('http://malicious.site/steal')
        assert 'http' not in result

    def test_no_scheme_unchanged(self):
        result = url_to_words('example.com/login')
        assert 'example' in result
        assert 'login'   in result

    def test_splits_on_dots(self):
        words = url_to_words('verify.paypal.com').split()
        assert 'verify' in words
        assert 'paypal' in words
        assert 'com'    in words

    def test_splits_on_slashes(self):
        words = url_to_words('site.com/admin/panel').split()
        assert 'site'  in words
        assert 'admin' in words
        assert 'panel' in words

    def test_filters_short_tokens(self):
        # single-char tokens should be filtered out
        words = url_to_words('a.b.example.com').split()
        assert 'a' not in words
        assert 'b' not in words
        assert 'example' in words

    def test_filters_pure_numeric(self):
        words = url_to_words('192.168.1.1/admin').split()
        for w in words:
            assert not w.isdigit(), f"Expected no pure-numeric token, got: {w}"

    def test_lowercases_result(self):
        result = url_to_words('MyBank.COM/Login')
        assert result == result.lower()

    def test_empty_url_no_crash(self):
        result = url_to_words('')
        assert isinstance(result, str)

    def test_query_string_split(self):
        words = url_to_words('site.com/search?q=malware&type=virus').split()
        assert 'search' in words

    def test_returns_string(self):
        assert isinstance(url_to_words('http://example.com'), str)


class TestBatchUrlToWords:

    def test_batch_length_preserved(self):
        urls   = ['http://a.com', 'http://b.org/path', 'https://c.net']
        result = batch_url_to_words(urls)
        assert len(result) == len(urls)

    def test_batch_returns_list_of_strings(self):
        urls   = ['http://example.com', 'http://phishing.site']
        result = batch_url_to_words(urls)
        assert all(isinstance(s, str) for s in result)

    def test_empty_list(self):
        assert batch_url_to_words([]) == []


class TestBertEncodeUrls:
    """
    Uses a lightweight tokenizer (bert-base-uncased via fast tokenizer) if
    available; otherwise the test is skipped to avoid downloading weights in CI.
    """

    @pytest.fixture(scope='module')
    def bert_tokenizer(self):
        pytest.importorskip('transformers')
        from transformers import AutoTokenizer
        try:
            return AutoTokenizer.from_pretrained('distilbert-base-uncased')
        except Exception:
            pytest.skip('distilbert-base-uncased not available')

    def test_output_shapes(self, bert_tokenizer):
        word_strs  = ['verify paypal login', 'malware download exe', 'legit site blog']
        max_len    = 16
        ids, mask  = bert_encode_urls(word_strs, bert_tokenizer, max_len)
        assert ids.shape  == (3, max_len), f"ids shape mismatch: {ids.shape}"
        assert mask.shape == (3, max_len), f"mask shape mismatch: {mask.shape}"

    def test_output_dtype_int32(self, bert_tokenizer):
        ids, mask = bert_encode_urls(['example domain path'], bert_tokenizer, 16)
        assert ids.dtype  == np.int32
        assert mask.dtype == np.int32

    def test_mask_values_binary(self, bert_tokenizer):
        ids, mask = bert_encode_urls(['short url'], bert_tokenizer, 32)
        unique = set(mask.flatten().tolist())
        assert unique.issubset({0, 1}), f"Unexpected mask values: {unique}"

    def test_padded_positions_have_zero_mask(self, bert_tokenizer):
        # A very short string should be padded; padded positions → mask=0
        ids, mask = bert_encode_urls(['hi'], bert_tokenizer, 64)
        assert 0 in mask.flatten().tolist(), "Expected some padding in a 64-length encoding"

    def test_empty_list(self, bert_tokenizer):
        ids, mask = bert_encode_urls([], bert_tokenizer, 16)
        assert ids.shape[0] == 0
