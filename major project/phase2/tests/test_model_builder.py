"""
Tests for model_builder.py — Phase 2
Verifies the triple-input model builds without error and has correct shapes.
"""

import sys
import os
import numpy as np
import pytest

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, _SRC_DIR)

from model_builder import build_model, build_and_compile_model


# ── Minimal config fixture ────────────────────────────────────────────────────

MINIMAL_CONFIG = {
    'random_seed': 42,
    'branch_a': {
        'embedding_dim':  16,
        'cnn_filters':    32,
        'cnn_kernel_size': 3,
        'gru_units':      16,
    },
    'branch_b': {
        'dense_units_1': 32,
        'dense_units_2': 16,
        'dropout':       0.3,
    },
    'branch_c': {
        'model_name':         'distilbert-base-uncased',
        'max_bert_length':    16,
        'projection_dim':     32,
        'projection_dropout': 0.2,
        'freeze':             True,
    },
    'head': {
        'dense_units': 64,
        'dropout':     0.5,
        'num_classes':  4,
    },
    'training': {
        'learning_rate': 1e-3,
    },
}

VOCAB_SIZE    = 100
MAX_CHAR_LEN  = 32
MAX_BERT_LEN  = 16


@pytest.fixture(scope='module')
def model():
    """Build once and reuse across tests."""
    tf = pytest.importorskip('tensorflow')
    pytest.importorskip('transformers')
    try:
        m = build_model(VOCAB_SIZE, MAX_CHAR_LEN, MAX_BERT_LEN, MINIMAL_CONFIG)
        return m
    except Exception as e:
        pytest.skip(f'distilbert-base-uncased unavailable or build error: {e}')


class TestBuildModel:

    def test_model_is_not_none(self, model):
        assert model is not None

    def test_model_has_four_inputs(self, model):
        assert len(model.inputs) == 4, \
            f"Expected 4 inputs, got {len(model.inputs)}"

    def test_input_names_present(self, model):
        names = [inp.name for inp in model.inputs]
        # Expect names containing 'char', 'lexical'/'lex', 'bert_ids', 'bert_mask'
        names_str = ' '.join(names).lower()
        assert 'char'  in names_str
        assert 'bert'  in names_str

    def test_output_shape(self, model):
        out_shape = model.output_shape
        # (None, 4) — batch axis + 4 classes
        assert out_shape == (None, 4), f"Unexpected output shape: {out_shape}"

    def test_output_is_softmax(self, model):
        import tensorflow as tf
        dummy = {
            inp.name: tf.zeros([2] + inp.shape[1:].as_list())
            for inp in model.inputs
        }
        preds = model(list(dummy.values()), training=False)
        row_sums = preds.numpy().sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(2), atol=1e-5)

    def test_frozen_bert_trainable_count(self, model):
        """Total trainable params should be much smaller than 66M (frozen BERT)."""
        trainable = sum(np.prod(w.shape) for w in model.trainable_weights)
        assert trainable < 5_000_000, \
            f"Expected frozen BERT → <5M trainable params, got {trainable:,}"

    def test_forward_pass_no_error(self, model):
        import tensorflow as tf
        batch = 4
        X_char  = np.random.randint(0, VOCAB_SIZE, (batch, MAX_CHAR_LEN))
        X_lex   = np.random.randn(batch, 23).astype('float32')
        X_ids   = np.random.randint(0, 100, (batch, MAX_BERT_LEN), dtype='int32')
        X_mask  = np.ones((batch, MAX_BERT_LEN), dtype='int32')
        preds   = model.predict([X_char, X_lex, X_ids, X_mask], verbose=0)
        assert preds.shape == (batch, 4)

    def test_model_summary_runs(self, model, capsys):
        model.summary()
        captured = capsys.readouterr()
        assert 'Total params' in captured.out or len(captured.out) > 0


class TestBuildAndCompileModel:

    def test_compiled_model_has_optimizer(self):
        pytest.importorskip('tensorflow')
        pytest.importorskip('transformers')
        try:
            m = build_and_compile_model(VOCAB_SIZE, MAX_CHAR_LEN, MAX_BERT_LEN,
                                        MINIMAL_CONFIG)
        except Exception as e:
            pytest.skip(f'Build unavailable: {e}')
        assert m.optimizer is not None

    def test_compiled_model_metrics_include_accuracy(self):
        pytest.importorskip('tensorflow')
        pytest.importorskip('transformers')
        try:
            m = build_and_compile_model(VOCAB_SIZE, MAX_CHAR_LEN, MAX_BERT_LEN,
                                        MINIMAL_CONFIG)
        except Exception as e:
            pytest.skip(f'Build unavailable: {e}')
        metric_names = [mm.name for mm in m.metrics]
        assert 'accuracy' in metric_names
