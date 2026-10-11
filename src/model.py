"""
LSTM Model Architecture Definition and Verification Module.

This module defines and compiles the genuine Long Short-Term Memory (LSTM)
neural network for EMNIST Balanced handwriting classification into 47 classes.
"""

import sys
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam

from src.config import (
    DENSE_UNITS,
    DROPOUT_RATE,
    INPUT_FEATURES,
    LEARNING_RATE,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    NUM_CLASSES,
    TIME_STEPS,
)


def build_model(
    input_shape: Tuple[int, int] = (TIME_STEPS, INPUT_FEATURES),
    num_classes: int = NUM_CLASSES,
    lstm_units: int = LSTM_UNITS_1,
    dropout_rate: float = DROPOUT_RATE,
    dense_units: int = DENSE_UNITS,
    learning_rate: float = LEARNING_RATE,
    stacked: bool = False,
) -> tf.keras.Model:
    """
    Constructs and compiles an authentic LSTM handwriting recognition model.
    
    Architecture (Baseline Single-Layer LSTM):
    1. Input Layer: Explicit sequence shape (28 time steps, 28 features).
    2. Recurrent Backbone: Genuine tf.keras.layers.LSTM with 128 units.
       Reads the 28 image rows sequentially from top to bottom, accumulating
       contextual stroke dynamics into its internal cell state.
    3. Regularization: Dropout layer (rate=0.2) to mitigate overfitting.
    4. Feature Dense Layer: Fully connected layer with 64 units and ReLU activation.
    5. Classification Output: Dense layer with 47 units and Softmax activation,
       producing a normalized probability distribution over all 47 character classes.
       
    Architectural Decision (Single vs. Stacked LSTM):
    - By default, stacked=False is selected as the simplest defensible architecture.
      With 101,520 training samples on a standard Windows CPU, a single 128-unit LSTM
      provides ample representational capacity (91,695 total parameters) while training
      in roughly half the wall-clock time of a multi-layer stacked recurrent network.
    - If stacked=True is requested, a 2-layer hierarchy (128 -> 64) is constructed.
    
    Args:
        input_shape: 2D tuple (time_steps=28, input_features=28).
        num_classes: Number of target character categories (47).
        lstm_units: Number of recurrent memory units in primary LSTM layer (128).
        dropout_rate: Fraction of input units to drop during training (0.2).
        dense_units: Number of neurons in intermediate feature layer (64).
        learning_rate: Learning rate for Adam optimizer (0.001).
        stacked: Whether to stack a second LSTM layer.
        
    Returns:
        tf.keras.Model: Compiled Keras Sequential model.
    """
    model_layers = [layers.Input(shape=input_shape, name="handwriting_sequence_input")]

    if stacked:
        # First LSTM returns complete sequence for downstream recurrent processing
        model_layers.append(
            layers.LSTM(lstm_units, return_sequences=True, name="lstm_layer_1")
        )
        model_layers.append(layers.Dropout(dropout_rate, name="dropout_lstm_1"))
        model_layers.append(
            layers.LSTM(LSTM_UNITS_2, return_sequences=False, name="lstm_layer_2")
        )
    else:
        # Single-layer LSTM processes sequence and outputs final hidden state vector
        model_layers.append(
            layers.LSTM(lstm_units, return_sequences=False, name="lstm_recurrent_layer")
        )

    # Regularization and Dense classification head
    model_layers.extend([
        layers.Dropout(dropout_rate, name="dropout_regularization"),
        layers.Dense(dense_units, activation="relu", name="dense_feature_layer"),
        layers.Dense(num_classes, activation="softmax", name="classification_output"),
    ])

    model = Sequential(model_layers, name="lstm_handwriting_classifier")

    # Compile with Adam optimizer and sparse categorical crossentropy
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def verify_model_architecture(model: tf.keras.Model = None) -> tf.keras.Model:
    """
    Performs exhaustive architecture verification and a dummy forward pass.
    """
    if model is None:
        model = build_model()

    print("\n" + "=" * 70)
    print("STEP 4: LSTM MODEL ARCHITECTURE INSPECTION & VERIFICATION")
    print("=" * 70)

    # 1. Print Model Summary
    model.summary()

    # 2. Check for Genuine LSTM Layer
    lstm_layers = [layer for layer in model.layers if isinstance(layer, layers.LSTM)]
    assert len(lstm_layers) > 0, "Verification Failed: No tf.keras.layers.LSTM layer found!"
    print(f"\n[CHECK 1] Genuine LSTM layer confirmed: {[l.name for l in lstm_layers]}")

    # 3. Check Input and Output Shapes
    expected_input_shape = (None, TIME_STEPS, INPUT_FEATURES)
    expected_output_shape = (None, NUM_CLASSES)
    assert model.input_shape == expected_input_shape, (
        f"Input shape mismatch: expected {expected_input_shape}, got {model.input_shape}"
    )
    assert model.output_shape == expected_output_shape, (
        f"Output shape mismatch: expected {expected_output_shape}, got {model.output_shape}"
    )
    print(f"[CHECK 2] Model Input Shape  : {model.input_shape} (Compatible with preprocessed (28, 28) tensors)")
    print(f"[CHECK 3] Model Output Shape : {model.output_shape} (Compatible with 47 character classes)")

    # 4. Check Compilation Parameters
    assert model.optimizer is not None, "Optimizer is not set!"
    assert isinstance(model.optimizer, Adam), f"Expected Adam optimizer, got {type(model.optimizer)}"
    assert model.loss == "sparse_categorical_crossentropy", (
        f"Expected sparse_categorical_crossentropy, got {model.loss}"
    )
    print(f"[CHECK 4] Optimizer          : {type(model.optimizer).__name__} (lr={model.optimizer.learning_rate.numpy():.4f})")
    print(f"[CHECK 5] Loss Function      : {model.loss} (Compatible with integer labels 0..46)")

    # 5. Check Parameter Counts
    total_params = model.count_params()
    trainable_params = sum(np.prod(p.shape) for p in model.trainable_weights)
    print(f"[CHECK 6] Total Parameters   : {total_params:,} (All {trainable_params:,} trainable)")

    # 6. Dummy Forward Pass Verification
    dummy_batch_size = 4
    dummy_input = np.random.uniform(0.0, 1.0, size=(dummy_batch_size, TIME_STEPS, INPUT_FEATURES)).astype(np.float32)
    dummy_output = model(dummy_input)

    assert dummy_output.shape == (dummy_batch_size, NUM_CLASSES), (
        f"Forward pass output shape mismatch: expected {(dummy_batch_size, NUM_CLASSES)}, got {dummy_output.shape}"
    )
    row_sums = np.sum(dummy_output.numpy(), axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Softmax probabilities do not sum to 1: {row_sums}"

    print(f"[CHECK 7] Dummy Forward Pass : Succeeded on CPU")
    print(f"          - Batch Input Shape: {dummy_input.shape}")
    print(f"          - Output Tensor    : {dummy_output.shape}")
    print(f"          - Softmax Row Sums : {row_sums} (Confirmed valid probability distributions)")
    print("=" * 70)
    print("[RESULT] Step 4 Acceptance Criteria: ALL VERIFICATION CHECKS PASSED.")
    print("=" * 70)

    return model


if __name__ == "__main__":
    verify_model_architecture()
