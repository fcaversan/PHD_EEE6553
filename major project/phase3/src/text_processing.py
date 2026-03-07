"""
Text processing for Malicious URL Detection Model
Character-level tokenization and sequence padding with data-driven max_length
"""

import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


def fit_tokenizer(urls: np.ndarray, percentile: int = 95) -> tuple:
    """
    Fit character-level tokenizer on training URLs and compute data-driven max_sequence_length.
    
    Args:
        urls: NumPy array of training URL strings (N,)
        percentile: Percentile for max_sequence_length calculation (default: 95)
    
    Returns:
        tuple: (fitted_tokenizer, max_sequence_length)
            - fitted_tokenizer: Keras Tokenizer object fitted on training URLs
            - max_sequence_length: Integer, 95th percentile of training URL lengths
    """
    
    print(f"\nFitting character-level tokenizer...")
    
    # Create character-level tokenizer
    tokenizer = Tokenizer(char_level=True, lower=False, oov_token='<OOV>')
    tokenizer.fit_on_texts(urls)
    
    # Compute max_sequence_length from percentile of URL lengths
    url_lengths = np.array([len(url) for url in urls])
    max_sequence_length = int(np.percentile(url_lengths, percentile))
    
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token
    
    print(f"✓ Tokenizer fitted on {len(urls)} URLs")
    print(f"  Vocabulary size: {vocab_size} unique characters")
    print(f"  URL length stats:")
    print(f"    Min:  {url_lengths.min()}")
    print(f"    Mean: {url_lengths.mean():.1f}")
    print(f"    Max:  {url_lengths.max()}")
    print(f"    {percentile}th percentile: {max_sequence_length}")
    print(f"  → max_sequence_length set to: {max_sequence_length}")
    
    return tokenizer, max_sequence_length


def tokenize_and_pad(tokenizer: Tokenizer, urls: np.ndarray, max_sequence_length: int) -> np.ndarray:
    """
    Convert URLs to padded character sequences.
    
    Args:
        tokenizer: Fitted Keras Tokenizer object
        urls: NumPy array of URL strings (N,)
        max_sequence_length: Maximum sequence length for padding
    
    Returns:
        np.ndarray: Padded sequences (N, max_sequence_length) as int32
    """
    
    # Convert URLs to sequences of character indices
    sequences = tokenizer.texts_to_sequences(urls)
    
    # Pad sequences (post-padding with zeros)
    padded_sequences = pad_sequences(
        sequences,
        maxlen=max_sequence_length,
        padding='post',  # Pad at the end
        truncating='post',  # Truncate at the end if longer
        dtype='int32'
    )
    
    print(f"✓ Tokenized and padded {len(urls)} URLs")
    print(f"  Output shape: {padded_sequences.shape}")
    print(f"  Dtype: {padded_sequences.dtype}")
    print(f"  Padding direction: post (trailing zeros)")
    
    return padded_sequences


def save_tokenizer(tokenizer: Tokenizer, tokenizer_path: str) -> None:
    """
    Save fitted tokenizer to JSON file.
    
    Args:
        tokenizer: Fitted Keras Tokenizer object
        tokenizer_path: Path to save tokenizer (.json)
    """
    
    tokenizer_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tokenizer_json)
    
    print(f"✓ Tokenizer saved to: {tokenizer_path}")


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """
    Load fitted tokenizer from JSON file.
    
    Args:
        tokenizer_path: Path to saved tokenizer (.json)
    
    Returns:
        Tokenizer: Loaded Keras Tokenizer object
    
    Raises:
        FileNotFoundError: If tokenizer file doesn't exist
    """
    
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}\n"
            f"Please train the model first to generate the tokenizer."
        )
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"✓ Tokenizer loaded (vocabulary size: {vocab_size})")
    
    return tokenizer


def process_training_urls(urls: np.ndarray, tokenizer_path: str, percentile: int = 95) -> tuple:
    """
    Complete training pipeline: fit tokenizer, compute max_length, tokenize, pad, and save.
    
    Args:
        urls: Training URL strings (N,)
        tokenizer_path: Path to save fitted tokenizer
        percentile: Percentile for max_sequence_length
    
    Returns:
        tuple: (padded_sequences, tokenizer, max_sequence_length)
    """
    
    # Fit tokenizer and compute max_length
    tokenizer, max_sequence_length = fit_tokenizer(urls, percentile)
    
    # Tokenize and pad
    padded_sequences = tokenize_and_pad(tokenizer, urls, max_sequence_length)
    
    # Save tokenizer
    save_tokenizer(tokenizer, tokenizer_path)
    
    return padded_sequences, tokenizer, max_sequence_length


def process_inference_urls(urls: np.ndarray, tokenizer_path: str, max_sequence_length: int) -> np.ndarray:
    """
    Inference pipeline: load tokenizer, tokenize, and pad.
    
    Args:
        urls: URL strings to process (N,)
        tokenizer_path: Path to saved tokenizer
        max_sequence_length: Max sequence length from training
    
    Returns:
        np.ndarray: Padded sequences (N, max_sequence_length) as int32
    """
    
    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Tokenize and pad
    padded_sequences = tokenize_and_pad(tokenizer, urls, max_sequence_length)
    
    return padded_sequences
