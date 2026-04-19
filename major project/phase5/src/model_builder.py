"""
Model builder — Phase 5 Hierarchical Two-Stage Classification.
Dual-input architecture with optional Gated Brand Cross-Attention.

When brand_attention.enabled = true, the 4 brand-aware features are
projected into a query that cross-attends over the BiGRU sequence.
A learned sigmoid gate (conditioned on brand features) shuts the
branch off for non-brand URLs, preserving baseline accuracy while
giving the model a dedicated pathway to detect brand impersonation.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def build_model(vocab_size: int, max_sequence_length: int, num_classes: int, config: dict) -> Model:
    """
    Build dual-input model with configurable output classes.

    Args:
        vocab_size: Character vocabulary size
        max_sequence_length: Max char-level URL length
        num_classes: 2 for Stage 1 (binary), 3 for Stage 2 (malicious sub-class)
        config: Configuration dictionary
    """
    embedding_dim   = config['branch_a']['embedding_dim']
    cnn_filters     = config['branch_a']['cnn_filters']
    cnn_kernel_size = config['branch_a']['cnn_kernel_size']
    gru_units       = config['branch_a']['gru_units']

    dense_b_1  = config['branch_b']['dense_units_1']
    dense_b_2  = config['branch_b']['dense_units_2']
    dropout_b  = config['branch_b']['dropout']
    n_features = config['branch_b']['input_features']

    dense_head   = config['head']['dense_units']
    dropout_head = config['head']['dropout']

    brand_cfg     = config.get('brand_attention', {})
    brand_enabled = brand_cfg.get('enabled', False)
    n_brand       = brand_cfg.get('n_brand_features', 4)

    # ── Branch A: Character Sequence ────────────────────────
    input_seq = layers.Input(shape=(max_sequence_length,), dtype=tf.int32, name='url_sequence')
    x = layers.Embedding(vocab_size, embedding_dim, name='embedding')(input_seq)
    x = layers.Conv1D(cnn_filters, cnn_kernel_size, activation='relu',
                      padding='same', name='conv1d')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool')(x)
    x = layers.Bidirectional(
        layers.GRU(gru_units, return_sequences=True), name='bidirectional_gru'
    )(x)

    # Save BiGRU output for cross-attention before self-attention
    bigru_seq = x  # (batch, T, gru_units*2)

    # Self-attention on sequence
    x = layers.Attention(name='self_attention')([x, x])
    branch_a = layers.GlobalAveragePooling1D(name='branch_a_output')(x)

    # ── Branch B: Lexical Heuristics ────────────────────────
    input_feat = layers.Input(shape=(n_features,), dtype=tf.float32, name='lexical_features')
    y = layers.Dense(dense_b_1, activation='relu', name='dense_b1')(input_feat)
    y = layers.Dropout(dropout_b, name='dropout_b')(y)
    branch_b = layers.Dense(dense_b_2, activation='relu', name='branch_b_output')(y)

    # ── Gated Brand Cross-Attention (impersonation defense) ─
    if brand_enabled:
        seq_dim = gru_units * 2  # BiGRU output dim (128)

        # Slice the 4 brand features (last n_brand of 27)
        brand_feat = layers.Lambda(
            lambda f: f[:, -n_brand:], name='brand_slice'
        )(input_feat)

        # Project brand features → query matching BiGRU dim
        brand_query = layers.Dense(
            seq_dim, activation='relu', name='brand_query_proj'
        )(brand_feat)
        brand_query = layers.Reshape(
            (1, seq_dim), name='brand_query_reshape'
        )(brand_query)

        # Cross-attention: brand query attends to BiGRU sequence
        brand_ctx = layers.Attention(
            name='brand_cross_attention'
        )([brand_query, bigru_seq])
        brand_ctx = layers.Reshape(
            (seq_dim,), name='brand_context_flat'
        )(brand_ctx)

        # Sigmoid gate conditioned on raw brand features
        # When brand signals are zero → gate ≈ 0 → no interference
        brand_gate = layers.Dense(
            seq_dim, activation='sigmoid', name='brand_gate'
        )(brand_feat)
        brand_ctx = layers.Multiply(
            name='brand_context_gated'
        )([brand_ctx, brand_gate])

        # 3-stream merge: general seq + brand context + lexical
        merged = layers.Concatenate(name='concatenate')(
            [branch_a, brand_ctx, branch_b]
        )
        print(f"  Brand cross-attention: {n_brand} brand feats → gated {seq_dim}-d context")
    else:
        # Original 2-stream merge
        merged = layers.Concatenate(name='concatenate')([branch_a, branch_b])

    # ── Classification Head ─────────────────────────────────
    h = layers.Dense(dense_head, activation='relu', name='dense_head')(merged)
    h = layers.Dropout(dropout_head, name='dropout_head')(h)

    activation = 'sigmoid' if num_classes == 2 else 'softmax'
    output_units = 1 if num_classes == 2 else num_classes

    output = layers.Dense(output_units, activation=activation, name='output')(h)

    model = Model(inputs=[input_seq, input_feat], outputs=output,
                  name=f'url_detector_stage{"1" if num_classes == 2 else "2"}')

    model.summary()
    print(f"\n✓ Model built — {num_classes}-class, {model.count_params():,} params")
    if brand_enabled:
        print(f"  Gated Brand Cross-Attention: ENABLED")
    return model


def compile_model(model: Model, config: dict, num_classes: int) -> Model:
    lr = config['training']['learning_rate']
    if num_classes == 2:
        loss = keras.losses.BinaryCrossentropy()
        metrics = ['accuracy',
                   keras.metrics.Precision(name='precision'),
                   keras.metrics.Recall(name='recall')]
    else:
        loss = keras.losses.CategoricalCrossentropy()
        metrics = ['accuracy',
                   keras.metrics.Precision(name='precision'),
                   keras.metrics.Recall(name='recall')]

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=loss, metrics=metrics)
    print(f"✓ Compiled (lr={lr}, loss={loss.name})")
    return model


def build_and_compile(vocab_size, max_seq_len, num_classes, config):
    model = build_model(vocab_size, max_seq_len, num_classes, config)
    return compile_model(model, config, num_classes)
