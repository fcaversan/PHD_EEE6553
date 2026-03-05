"""
URL Subword Tokenizer — Phase 2, Branch C
Converts raw URLs into space-separated word tokens suitable for BERT input.

Strategy: strip scheme → split on URL structural delimiters → filter noise → join.
This mimics how a human reader segments a URL to extract semantic meaning.
"""

import re
from typing import List


def url_to_words(url: str) -> str:
    """
    Convert a URL string into space-separated word tokens for BERT input.

    Processing steps:
        1. Strip scheme (http://, https://)
        2. Split on URL structural delimiters: . - / _ ? = & % + @ # : , ; ~
        3. Filter tokens shorter than 2 characters and purely numeric tokens
        4. Lowercase all tokens
        5. Join with spaces → suitable for BERT WordPiece tokenizer

    Examples:
        "secure-paypal-login.verify.tk/account/update"
        → "secure paypal login verify tk account update"

        "jakarta.apache.org/bsf/"
        → "jakarta apache org bsf"

        "https://docs.python.org/3/library/urllib.parse.html"
        → "docs python org library urllib parse html"

        "192.168.1.1/admin/login.php"
        → "admin login php"

        "bit.ly/3xYz123"
        → "bit ly xYz"

    Args:
        url: Raw URL string (with or without scheme)

    Returns:
        str: Space-separated word tokens (minimum "unknown" for empty results)
    """
    # Strip scheme
    url = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)

    # Split on all URL structural delimiters
    tokens = re.split(r'[.\-/_?=&%+@#:,;~!]', url)

    # Normalise: lowercase, filter short/purely-numeric tokens
    tokens = [
        t.lower() for t in tokens
        if len(t) >= 2 and not t.isdigit()
    ]

    return ' '.join(tokens) if tokens else 'unknown'


def batch_url_to_words(urls, verbose: bool = False) -> List[str]:
    """
    Apply url_to_words to an iterable of URL strings.

    Args:
        urls: Iterable of raw URL strings
        verbose: If True, print progress

    Returns:
        List[str]: List of word-tokenized URL strings
    """
    result = [url_to_words(str(u)) for u in urls]
    if verbose:
        print(f"✓ URL→words tokenization applied to {len(result)} URLs")
        # Example output for first 3
        for i, (orig, words) in enumerate(zip(list(urls)[:3], result[:3])):
            print(f"  [{i}] {str(orig)[:60]!r} → {words!r}")
    return result


def bert_encode_urls(
    word_strings: List[str],
    bert_tokenizer,
    max_length: int
):
    """
    Encode a list of word-tokenized URL strings using a HuggingFace tokenizer.

    Args:
        word_strings: List of space-separated word strings (from batch_url_to_words)
        bert_tokenizer: Loaded HuggingFace PreTrainedTokenizer
        max_length: Maximum subword token length (padding/truncation target)

    Returns:
        tuple: (input_ids, attention_mask) — both numpy int32 arrays of shape (N, max_length)
    """
    import numpy as np

    encoding = bert_tokenizer(
        word_strings,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np',
        add_special_tokens=True,   # adds [CLS] and [SEP]
    )

    input_ids      = encoding['input_ids'].astype(np.int32)
    attention_mask = encoding['attention_mask'].astype(np.int32)

    return input_ids, attention_mask
