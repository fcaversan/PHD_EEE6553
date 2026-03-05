"""
Model builder — Phase 2
Triple-input hybrid architecture:
  Branch A: Character-level (Embedding → Conv1D → MaxPool → BiGRU → Attention → GAP)
  Branch B: Lexical heuristics (Dense → Dropout → Dense)
  Branch C: Semantic / DistilBERT (frozen) → CLS token → projection Dense
  Head: Concatenate(A, B, C) → Dense → Dropout → Softmax(4)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def build_model(
    vocab_size: int,
    max_sequence_length: int,
    max_bert_length: int,
    config: dict,
) -> Model:
    """
    Build triple-input malicious URL detection model.

    Args:
        vocab_size: Character vocabulary size (from Phase 2 char tokenizer)
        max_sequence_length: Maximum char-level URL length
        max_bert_length: Maximum subword token length for BERT branch
        config: Configuration dictionary

    Returns:
        Compiled Keras Model with three inputs
    """
    from transformers import TFDistilBertModel

    print(f"\n{'='*60}")
    print("Building Triple-Input Model (Phase 2)")
    print(f"{'='*60}")

    # ── Config ──────────────────────────────────────────────────
    embedding_dim   = config['branch_a']['embedding_dim']
    cnn_filters     = config['branch_a']['cnn_filters']
    cnn_kernel_size = config['branch_a']['cnn_kernel_size']
    gru_units       = config['branch_a']['gru_units']

    dense_b_1    = config['branch_b']['dense_units_1']
    dense_b_2    = config['branch_b']['dense_units_2']
    dropout_b    = config['branch_b']['dropout']

    bert_name       = config['branch_c']['model_name']
    proj_dim        = config['branch_c']['projection_dim']
    proj_dropout    = config['branch_c']['projection_dropout']
    freeze_bert     = config['branch_c']['freeze']

    dense_head   = config['head']['dense_units']
    dropout_head = config['head']['dropout']
    num_classes  = config['head']['num_classes']

    # ── Branch A: Character Sequence ────────────────────────────
    input_sequence = layers.Input(
        shape=(max_sequence_length,), dtype=tf.int32, name='url_sequence'
    )
    x = layers.Embedding(vocab_size, embedding_dim, name='embedding')(input_sequence)
    x = layers.Conv1D(cnn_filters, cnn_kernel_size, activation='relu',
                      padding='same', name='conv1d')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool')(x)
    x = layers.Bidirectional(
        layers.GRU(gru_units, return_sequences=True), name='bidirectional_gru'
    )(x)
    x = layers.Attention(name='attention')([x, x])
    branch_a_out = layers.GlobalAveragePooling1D(name='branch_a_output')(x)
    # shape: (batch, 128)

    # ── Branch B: Lexical Heuristics ────────────────────────────
    input_features = layers.Input(shape=(23,), dtype=tf.float32, name='lexical_features')
    y = layers.Dense(dense_b_1, activation='relu', name='dense_b1')(input_features)
    y = layers.Dropout(dropout_b, name='dropout_b')(y)
    branch_b_out = layers.Dense(dense_b_2, activation='relu', name='branch_b_output')(y)
    # shape: (batch, 32)

    # ── Branch C: Semantic (DistilBERT) ─────────────────────────
    print(f"\nLoading DistilBERT: {bert_name} (freeze={freeze_bert}) ...")
    bert_model = TFDistilBertModel.from_pretrained(bert_name)
    bert_model.trainable = not freeze_bert
    n_bert_params = bert_model.count_params()
    status = "frozen" if freeze_bert else "trainable"
    print(f"✓ DistilBERT loaded — {n_bert_params:,} params ({status})")

    input_ids   = layers.Input(
        shape=(max_bert_length,), dtype=tf.int32, name='bert_input_ids'
    )
    attn_mask   = layers.Input(
        shape=(max_bert_length,), dtype=tf.int32, name='bert_attention_mask'
    )

    # Run through DistilBERT — last_hidden_state is index [0]
    bert_out    = bert_model(input_ids, attention_mask=attn_mask)[0]   # (batch, seq, 768)
    cls_token   = bert_out[:, 0, :]                                     # [CLS]: (batch, 768)
    z           = layers.Dense(proj_dim, activation='relu',
                               name='bert_projection')(cls_token)
    branch_c_out = layers.Dropout(proj_dropout, name='bert_dropout')(z)
    # shape: (batch, 128)

    # ── Classification Head ──────────────────────────────────────
    # Concatenate: 128 + 32 + 128 = 288
    merged = layers.Concatenate(name='concatenate')(
        [branch_a_out, branch_b_out, branch_c_out]
    )
    h = layers.Dense(dense_head, activation='relu', name='dense_head')(merged)
    h = layers.Dropout(dropout_head, name='dropout_head')(h)
    output = layers.Dense(num_classes, activation='softmax', name='output')(h)

    # ── Assemble ─────────────────────────────────────────────────
    model = Model(
        inputs=[input_sequence, input_features, input_ids, attn_mask],
        outputs=output,
        name='malicious_url_detector_phase2'
    )

    print("\n" + "="*60)
    model.summary()
    print("="*60)
    print(f"\n✓ Model built successfully")
    print(f"  Branch A input (char):  {input_sequence.shape}")
    print(f"  Branch B input (lex):   {input_features.shape}")
    print(f"  Branch C input ids:     {input_ids.shape}")
    print(f"  Branch C attn mask:     {attn_mask.shape}")
    print(f"  Output:                 {output.shape}")
    print(f"  Total parameters:       {model.count_params():,}")
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"  Trainable parameters:   {trainable:,}")

    return model


def compile_model(model: Model, config: dict) -> Model:
    """Compile model with Adam, CategoricalCrossentropy, and standard metrics."""
    lr = config['training']['learning_rate']
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ]
    )
    print(f"\n✓ Model compiled  (lr={lr})")
    return model


def build_and_compile_model(
    vocab_size: int,
    max_sequence_length: int,
    max_bert_length: int,
    config: dict,
) -> Model:
    """Convenience wrapper: build + compile in one call."""
    model = build_model(vocab_size, max_sequence_length, max_bert_length, config)
    return compile_model(model, config)
