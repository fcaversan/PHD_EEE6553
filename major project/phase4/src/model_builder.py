"""
Model builder for Malicious URL Detection Model
Dual-input hybrid architecture using Keras Functional API
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


def build_model(vocab_size: int, max_sequence_length: int, config: dict) -> Model:
    """
    Build dual-input malicious URL detection model using Keras Functional API.
    
    Architecture:
        Branch A (Sequence): Embedding → Conv1D → MaxPool → Bi-GRU → Attention
        Branch B (Heuristic): Dense(64) → Dropout(0.3) → Dense(32)
        Head: Concatenate → Dense(128) → Dropout(0.5) → Dense(4, softmax)
    
    Args:
        vocab_size: Size of character vocabulary (from tokenizer)
        max_sequence_length: Maximum URL sequence length (from data analysis)
        config: Configuration dictionary with hyperparameters
    
    Returns:
        Model: Compiled Keras model with two inputs and one output
    """
    
    print(f"\n{'='*60}")
    print("Building Dual-Input Model")
    print(f"{'='*60}")
    
    # Extract hyperparameters from config
    embedding_dim = config['branch_a']['embedding_dim']
    cnn_filters = config['branch_a']['cnn_filters']
    cnn_kernel_size = config['branch_a']['cnn_kernel_size']
    gru_units = config['branch_a']['gru_units']
    
    dense_b_1 = config['branch_b']['dense_units_1']
    dense_b_2 = config['branch_b']['dense_units_2']
    dropout_b = config['branch_b']['dropout']
    
    dense_head = config['head']['dense_units']
    dropout_head = config['head']['dropout']
    num_classes = config['head']['num_classes']
    
    # ========== Branch A: Character Sequence Processing ==========
    input_sequence = layers.Input(shape=(max_sequence_length,), name='url_sequence')
    
    # Embedding layer
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name='embedding'
    )(input_sequence)
    
    # 1D Convolutional layer
    x = layers.Conv1D(
        filters=cnn_filters,
        kernel_size=cnn_kernel_size,
        activation='relu',
        padding='same',
        name='conv1d'
    )(x)
    
    # Max pooling
    x = layers.MaxPooling1D(pool_size=2, name='maxpool')(x)
    
    # Bidirectional GRU
    x = layers.Bidirectional(
        layers.GRU(gru_units, return_sequences=True),
        name='bidirectional_gru'
    )(x)
    
    # Attention mechanism (using built-in Attention layer)
    # Self-attention: query and value are the same
    attention_output = layers.Attention(name='attention')([x, x])
    
    # Global average pooling to get fixed-size vector
    branch_a_output = layers.GlobalAveragePooling1D(name='branch_a_output')(attention_output)
    
    # ========== Branch B: Lexical Heuristic Features ==========
    n_features = int(config.get('branch_b', {}).get('input_features', 27))
    input_features = layers.Input(shape=(n_features,), name='lexical_features')
    
    # Dense layer 1
    y = layers.Dense(
        units=dense_b_1,
        activation='relu',
        name='dense_b1'
    )(input_features)
    
    # Dropout
    y = layers.Dropout(rate=dropout_b, name='dropout_b')(y)
    
    # Dense layer 2
    branch_b_output = layers.Dense(
        units=dense_b_2,
        activation='relu',
        name='branch_b_output'
    )(y)
    
    # ========== Classification Head ==========
    # Concatenate both branches
    merged = layers.Concatenate(name='concatenate')([branch_a_output, branch_b_output])
    
    # Dense layer
    z = layers.Dense(
        units=dense_head,
        activation='relu',
        name='dense_head'
    )(merged)
    
    # Dropout
    z = layers.Dropout(rate=dropout_head, name='dropout_head')(z)
    
    # Output layer (4-class softmax)
    output = layers.Dense(
        units=num_classes,
        activation='softmax',
        name='output'
    )(z)
    
    # ========== Create Model ==========
    model = Model(
        inputs=[input_sequence, input_features],
        outputs=output,
        name='malicious_url_detector'
    )
    
    # Print model summary
    print("\n" + "="*60)
    model.summary()
    print("="*60)
    
    print(f"\n✓ Model built successfully")
    print(f"  Branch A input: {input_sequence.shape}")
    print(f"  Branch B input: {input_features.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Total parameters: {model.count_params():,}")
    
    return model


def compile_model(model: Model, config: dict) -> Model:
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras Model to compile
        config: Configuration dictionary with training parameters
    
    Returns:
        Model: Compiled Keras model
    """
    
    learning_rate = config['training']['learning_rate']
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    print(f"\n✓ Model compiled")
    print(f"  Optimizer: Adam (lr={learning_rate})")
    print(f"  Loss: CategoricalCrossentropy")
    print(f"  Metrics: accuracy, precision, recall")
    
    return model


def build_and_compile_model(vocab_size: int, max_sequence_length: int, config: dict) -> Model:
    """
    Convenience function to build and compile model in one call.
    
    Args:
        vocab_size: Size of character vocabulary
        max_sequence_length: Maximum URL sequence length  
        config: Configuration dictionary
    
    Returns:
        Model: Built and compiled Keras model
    """
    
    model = build_model(vocab_size, max_sequence_length, config)
    model = compile_model(model, config)
    
    return model
