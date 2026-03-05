"""
Utility functions — Phase 2 (identical to Phase 1 with added torch seed support)
"""

import os
import random
import yaml
import numpy as np
import tensorflow as tf


def set_all_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility (Python, NumPy, TF, PyTorch)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Seed PyTorch if available (used by HuggingFace tokenizer internals)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    print(f"✓ All random seeds set to {seed}")


def check_gpu() -> bool:
    """Check for GPU availability and log device information."""
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
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Run from the phase2/ directory so config.yaml is found."
        )
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Configuration loaded from {config_path}")
    return config
