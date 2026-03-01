"""
Unit tests for model builder module
Tests model architecture, input/output shapes, and layer structure
"""

import pytest
import numpy as np
from tensorflow import keras
from src.model_builder import build_model, compile_model, build_and_compile_model


# Sample config for testing
TEST_CONFIG = {
    'branch_a': {
        'embedding_dim': 16,
        'cnn_filters': 32,
        'cnn_kernel_size': 3,
        'gru_units': 16
    },
    'branch_b': {
        'dense_units_1': 32,
        'dense_units_2': 16,
        'dropout': 0.3
    },
    'head': {
        'dense_units': 64,
        'dropout': 0.5,
        'num_classes': 4
    },
    'training': {
        'learning_rate': 0.001
    }
}


def test_build_model():
    """Test model builds without error."""
    vocab_size = 100
    max_seq_len = 50
    
    model = build_model(vocab_size, max_seq_len, TEST_CONFIG)
    
    assert isinstance(model, keras.Model)
    assert model.name == 'malicious_url_detector'


def test_input_names():
    """Test model has correct input names."""
    vocab_size = 100
    max_seq_len = 50
    
    model = build_model(vocab_size, max_seq_len, TEST_CONFIG)
    
    input_names = [inp.name for inp in model.inputs]
    assert 'url_sequence' in input_names[0]
    assert 'lexical_features' in input_names[1]


def test_output_shape():
    """Test model output has correct shape (4 classes)."""
    vocab_size = 100
    max_seq_len = 50
    
    model = build_model(vocab_size, max_seq_len, TEST_CONFIG)
    
    # Test with dummy data
    batch_size = 2
    dummy_sequences = np.random.randint(0, vocab_size, (batch_size, max_seq_len))
    dummy_features = np.random.randn(batch_size, 23).astype(np.float32)
    
    output = model.predict([dummy_sequences, dummy_features], verbose=0)
    
    assert output.shape == (batch_size, 4)
    # Output should be probabilities (sum to 1)
    np.testing.assert_almost_equal(output.sum(axis=1), np.ones(batch_size), decimal=5)


def test_input_shapes():
    """Test model accepts correct input shapes."""
    vocab_size = 100
    max_seq_len = 50
    
    model = build_model(vocab_size, max_seq_len, TEST_CONFIG)
    
    # Input 0: sequences
    assert model.inputs[0].shape[1:] == (max_seq_len,)
    
    # Input 1: features
    assert model.inputs[1].shape[1:] == (23,)


def test_branch_independence():
    """Test that branches are separate until merge."""
    vocab_size = 100
    max_seq_len = 50
    
    model = build_model(vocab_size, max_seq_len, TEST_CONFIG)
    
    # Check that concatenate layer exists
    layer_names = [layer.name for layer in model.layers]
    assert 'concatenate' in layer_names
    
    # Check architecture has both branch names
    assert any('branch_a' in name or 'embedding' in name for name in layer_names)
    assert any('branch_b' in name or 'dense_b' in name for name in layer_names)


def test_model_compilation():
    """Test model compiles correctly."""
    vocab_size = 100
    max_seq_len = 50
    
    model = build_model(vocab_size, max_seq_len, TEST_CONFIG)
    compiled_model = compile_model(model, TEST_CONFIG)
    
    # Check optimizer
    assert isinstance(compiled_model.optimizer, keras.optimizers.Adam)
    
    # Check loss
    assert isinstance(compiled_model.loss, keras.losses.CategoricalCrossentropy)
    
    # Check metrics
    metric_names = [m.name for m in compiled_model.metrics]
    assert 'accuracy' in metric_names or 'acc' in metric_names
    assert 'precision' in metric_names
    assert 'recall' in metric_names


def test_build_and_compile_convenience():
    """Test convenience function builds and compiles in one call."""
    vocab_size = 100
    max_seq_len = 50
    
    model = build_and_compile_model(vocab_size, max_seq_len, TEST_CONFIG)
    
    # Should be compiled
    assert model.optimizer is not None
    assert model.loss is not None


def test_model_training_compatibility():
    """Test model can be trained on dummy data."""
    vocab_size = 100
    max_seq_len = 50
    batch_size = 10
    
    model = build_and_compile_model(vocab_size, max_seq_len, TEST_CONFIG)
    
    # Create dummy training data
    X_seq = np.random.randint(0, vocab_size, (batch_size, max_seq_len))
    X_feat = np.random.randn(batch_size, 23).astype(np.float32)
    y = keras.utils.to_categorical(np.random.randint(0, 4, batch_size), num_classes=4)
    
    # Train for 1 epoch (should not raise error)
    history = model.fit(
        [X_seq, X_feat],
        y,
        epochs=1,
        verbose=0
    )
    
    assert 'loss' in history.history
    assert len(history.history['loss']) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
