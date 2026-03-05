"""
Utility functions for the Malicious URL Detection Model
Provides seed setting, GPU checking, and config loading helpers
"""

import os
import random
import yaml
import numpy as np
import tensorflow as tf


def set_all_seeds(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility across numpy, tensorflow, and Python's random module.
    Also sets PYTHONHASHSEED environment variable.
    
    Args:
        seed: Integer seed value for all random number generators
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"✓ All random seeds set to {seed}")


def check_gpu() -> bool:
    """
    Check for GPU availability and log device information.
    Logs all available physical devices (CPU and GPU).
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print(f"\n{'='*60}")
    print("Hardware Detection")
    print(f"{'='*60}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CPUs detected: {len(cpus)}")
    print(f"GPUs detected: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        print("✓ GPU acceleration available")
        return True
    else:
        print("⚠ No GPU detected - training will use CPU (slower)")
        return False


def get_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file with error handling.
    
    Args:
        config_path: Path to config.yaml file (relative to project root)
    
    Returns:
        dict: Parsed configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure config.yaml exists in the project root."
        )
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from {config_path}")
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing config file: {e}")


def normalize_url(url: str) -> str:
    """
    Ensure every URL has an explicit scheme so that ``urlparse`` can
    correctly split *hostname* from *path*.

    The malicious_phish.csv dataset stores most benign URLs **without**
    a scheme (e.g. ``github.com/path``), while phishing/malware URLs
    are mixed.  ``urlparse`` treats a scheme-less URL as an opaque
    path — ``hostname_length`` becomes 0 and ``path_length`` equals
    the full URL length, which is wrong.

    By prepending ``http://`` to any URL that lacks a scheme we make
    all URLs structurally uniform so the model learns from
    *genuine* lexical signals (hostname length, path depth, etc.)
    rather than scheme-presence artefacts.

    Args:
        url: Raw URL string, possibly missing a scheme

    Returns:
        str: URL guaranteed to start with ``http://`` or ``https://``
    """
    if not url.startswith('http://') and not url.startswith('https://'):
        return 'http://' + url
    return url


def normalize_urls_array(urls: np.ndarray) -> np.ndarray:
    """
    Apply normalize_url to every element of a NumPy array.
    
    Args:
        urls: 1-D array of URL strings
    
    Returns:
        np.ndarray: 1-D array of scheme-stripped URL strings
    """
    return np.array([normalize_url(u) for u in urls])


def create_directories(config: dict) -> None:
    """
    Create necessary output directories if they don't exist.
    
    Args:
        config: Configuration dictionary containing paths
    """
    dirs_to_create = [
        config['data']['artifacts_dir'],
        config['data']['results_dir']
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"✓ Output directories verified/created")
