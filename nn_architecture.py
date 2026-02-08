"""Simple Keras model architecture (no training)."""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape):
    """Builds and compiles a binary classifier model.

    Args:
        input_shape: Tuple describing the input feature shape, e.g. (num_features,).

    Returns:
        A compiled keras.Model instance.
    """

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Dense(10, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    # Example usage; adjust input_shape to match your data.
    build_model((10,))
