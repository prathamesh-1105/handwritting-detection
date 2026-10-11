"""
Model Training and History Tracking Module for LSTM Handwriting Recognition.

This module handles:
1. Loading preprocessed training and validation splits via src.preprocess.
2. Instantiating the compiled LSTM model via src.model.
3. Training with Adam and Sparse Categorical Cross-Entropy across up to 15 epochs.
4. Monitoring performance with ModelCheckpoint (saving best weights) and EarlyStopping.
5. Saving comprehensive training metrics to outputs/metrics/training_history.json.
6. Verifying that the saved checkpoint can be reloaded and produces valid (batch, 47) predictions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    HISTORY_SAVE_PATH,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    PATIENCE,
    RANDOM_SEED,
)
from src.model import build_model
from src.preprocess import prepare_dataset


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE,
    model_save_path: Path = MODEL_SAVE_PATH,
    history_save_path: Path = HISTORY_SAVE_PATH,
) -> Tuple[tf.keras.Model, Dict]:
    """
    Executes the training pipeline for the genuine LSTM handwriting classifier.
    
    Args:
        epochs: Maximum number of training epochs (default: 15).
        batch_size: Mini-batch gradient descent size (default: 128).
        patience: Number of epochs with no validation loss improvement before early stopping (default: 3).
        model_save_path: Destination file for the best Keras model checkpoint.
        history_save_path: Destination JSON file for training metric history.
        
    Returns:
        Tuple[tf.keras.Model, Dict]: The trained/restored Keras model and the history summary dictionary.
    """
    print("\n" + "=" * 70)
    print("STEP 5: LSTM MODEL TRAINING AND HISTORY TRACKING")
    print("=" * 70)

    # 1. Load Preprocessed Data (Integer Labels, Held-out Test Set Excluded)
    print("[1/5] Loading preprocessed training and validation datasets...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test), class_mapping = prepare_dataset(
        one_hot=False, visualize=False
    )

    print(f"      Training subset   : {x_train.shape} images, {y_train.shape} labels")
    print(f"      Validation subset : {x_val.shape} images, {y_val.shape} labels")
    print(f"      Held-out test set : {x_test.shape} preserved exclusively for Step 7")

    # 2. Instantiate Model
    print("\n[2/5] Instantiating compiled LSTM model...")
    model = build_model(
        input_shape=(28, 28),
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE,
    )

    # Ensure output directories exist
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    history_save_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Configure Callbacks
    # ModelCheckpoint saves the best model on disk when val_loss reaches a new minimum
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_save_path),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    # EarlyStopping halts training if validation loss does not improve for `patience` consecutive epochs
    # restore_best_weights=True ensures model in memory reverts to the optimal epoch weights
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    callbacks = [checkpoint_callback, early_stopping_callback]

    print(f"\n[3/5] Starting model training on CPU:")
    print(f"      - Batch Size          : {batch_size}")
    print(f"      - Maximum Epochs      : {epochs}")
    print(f"      - EarlyStopping       : patience={patience}, restore_best_weights=True")
    print(f"      - Checkpoint Path     : {model_save_path}")
    print(f"      - Optimizer           : Adam (lr={LEARNING_RATE})")
    print(f"      - Loss Function       : sparse_categorical_crossentropy")
    print("-" * 70)

    # 4. Train the Model
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    raw_history = history.history
    completed_epochs = len(raw_history["loss"])

    # 5. Determine Best Epoch vs Final Epoch Metrics
    # Best epoch is identified by the minimum validation loss
    best_epoch_idx = int(np.argmin(raw_history["val_loss"]))
    best_epoch_num = best_epoch_idx + 1

    best_metrics = {
        "best_epoch": best_epoch_num,
        "best_val_loss": float(raw_history["val_loss"][best_epoch_idx]),
        "best_val_accuracy": float(raw_history["val_accuracy"][best_epoch_idx]),
        "best_train_loss": float(raw_history["loss"][best_epoch_idx]),
        "best_train_accuracy": float(raw_history["accuracy"][best_epoch_idx]),
        "final_epoch": completed_epochs,
        "final_val_loss": float(raw_history["val_loss"][-1]),
        "final_val_accuracy": float(raw_history["val_accuracy"][-1]),
        "final_train_loss": float(raw_history["loss"][-1]),
        "final_train_accuracy": float(raw_history["accuracy"][-1]),
    }

    # 6. Save Training History to JSON
    print("\n[4/5] Saving complete training history to JSON...")
    history_payload = {
        "configuration": {
            "batch_size": batch_size,
            "max_epochs": epochs,
            "completed_epochs": completed_epochs,
            "learning_rate": LEARNING_RATE,
            "patience": patience,
            "random_seed": RANDOM_SEED,
            "optimizer": "Adam",
            "loss": "sparse_categorical_crossentropy",
            "model_path": str(model_save_path),
        },
        "summary": best_metrics,
        "history": {
            "loss": [float(v) for v in raw_history["loss"]],
            "accuracy": [float(v) for v in raw_history["accuracy"]],
            "val_loss": [float(v) for v in raw_history["val_loss"]],
            "val_accuracy": [float(v) for v in raw_history["val_accuracy"]],
        },
    }

    with open(history_save_path, "w", encoding="utf-8") as f_out:
        json.dump(history_payload, f_out, indent=2)

    print(f"      Training history saved to: {history_save_path}")

    # 7. Reload and Verify Saved Checkpoint
    print("\n[5/5] Reloading and verifying saved model checkpoint...")
    assert model_save_path.exists(), f"Error: Checkpoint file not found at {model_save_path}"
    loaded_model = tf.keras.models.load_model(str(model_save_path))

    # Test prediction shape on small validation sample
    sample_batch = x_val[:4]
    sample_preds = loaded_model.predict(sample_batch, verbose=0)

    assert sample_preds.shape == (4, NUM_CLASSES), (
        f"Reloaded model prediction shape mismatch: expected (4, {NUM_CLASSES}), got {sample_preds.shape}"
    )
    row_sums = np.sum(sample_preds, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Softmax row sums do not equal 1.0: {row_sums}"

    print(f"      - Reloaded model from : {model_save_path}")
    print(f"      - Validation batch test: Input shape {sample_batch.shape} -> Prediction shape {sample_preds.shape}")
    print(f"      - Softmax row sums    : {row_sums} (Confirmed valid probabilities)")
    print(f"      - Sample predictions  : {np.argmax(sample_preds, axis=-1)} | Ground truth: {y_val[:4]}")

    print("\n" + "=" * 70)
    print("TRAINING SUMMARY & PERFORMANCE AUDIT")
    print("=" * 70)
    print(f"Total Epochs Completed : {completed_epochs} of {epochs}")
    print(f"Best Epoch             : Epoch {best_epoch_num}")
    print(f"Best Validation Loss   : {best_metrics['best_val_loss']:.4f}")
    print(f"Best Validation Accuracy: {best_metrics['best_val_accuracy'] * 100:.2f}%")
    print(f"Final Validation Loss  : {best_metrics['final_val_loss']:.4f}")
    print(f"Final Validation Accuracy: {best_metrics['final_val_accuracy'] * 100:.2f}%")
    print(f"Saved Checkpoint Path  : {model_save_path}")
    print(f"Saved History Path     : {history_save_path}")
    print("=" * 70)

    return loaded_model, history_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Handwriting Recognition Model on EMNIST Balanced")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Maximum training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size (default: 128)")
    parser.add_argument("--check", action="store_true", help="Perform sanity check and print config without training")
    args = parser.parse_args()

    if args.check:
        print("\n" + "=" * 70)
        print("[CHECK] Sanity check mode: Verifying model architecture and configuration...")
        print("=" * 70)
        m = build_model()
        print(f"[CHECK] Successfully built model with {m.count_params():,} total parameters.")
        print(f"[CHECK] Target checkpoint: {MODEL_SAVE_PATH}")
        print(f"[CHECK] Target history   : {HISTORY_SAVE_PATH}")
        print("[CHECK] Sanity check completed successfully. (Run without --check to execute full training).")
    else:
        train_model(epochs=args.epochs, batch_size=args.batch_size)
