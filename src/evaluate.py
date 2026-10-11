"""
Step 7: Evaluation on Unseen Test Data Module.

This module handles:
1. Reloading the saved best LSTM model checkpoint from models/lstm_handwriting_model.keras.
2. Loading and preprocessing the official 18,800 held-out EMNIST Balanced test samples.
3. Evaluating test loss and test accuracy without retraining or parameter alteration.
4. Computing scikit-learn macro, weighted, and per-class precision, recall, F1, and support across all 47 classes.
5. Exporting reproducible evaluation artifacts to outputs/metrics/test_evaluation.json and classification_report.csv.
6. Verifying rigorous evaluation integrity assertions.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
import tensorflow as tf

from src.config import (
    METRICS_DIR,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    TEST_SAMPLES_PER_CLASS,
    TEST_SAMPLES_TOTAL,
)
from src.dataset import load_emnist_balanced
from src.preprocess import correct_orientation, encode_labels, normalize_pixels


def load_and_preprocess_test_data() -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Loads raw test data from data/ cache and applies Step 3 preprocessing pipeline.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
            x_test (18800, 28, 28) float32 in [0.0, 1.0],
            y_test (18800,) int32 in [0, 46],
            class_mapping dictionary {cls_idx: char}
    """
    print("[1/5] Loading official test split from cached files...")
    _, (raw_test_x, raw_test_y), class_mapping = load_emnist_balanced()

    print("[2/5] Preprocessing test samples (transposition and [0.0, 1.0] normalization)...")
    oriented_test_x = correct_orientation(raw_test_x)
    x_test = normalize_pixels(oriented_test_x)
    y_test = encode_labels(raw_test_y, one_hot=False)

    return x_test, y_test, class_mapping


def evaluate_test_set(
    model_path: Path = MODEL_SAVE_PATH,
    batch_size: int = 128,
) -> Dict:
    """
    Executes complete, rigorous evaluation of the saved LSTM model on unseen test data.
    
    Args:
        model_path: Path to models/lstm_handwriting_model.keras.
        batch_size: Mini-batch inference size.
        
    Returns:
        Dict: Complete evaluation metrics dictionary.
    """
    print("\n" + "=" * 70)
    print("STEP 7: EVALUATION ON UNSEEN TEST DATA (EMNIST BALANCED)")
    print("=" * 70)

    # 1. Load Data
    x_test, y_test, class_mapping = load_and_preprocess_test_data()

    # 2. Verify Data Assertions
    print("\n[3/5] Validating test set integrity...")
    assert x_test.shape == (TEST_SAMPLES_TOTAL, 28, 28), (
        f"Expected x_test shape ({TEST_SAMPLES_TOTAL}, 28, 28), got {x_test.shape}"
    )
    assert y_test.shape == (TEST_SAMPLES_TOTAL,), (
        f"Expected y_test shape ({TEST_SAMPLES_TOTAL},), got {y_test.shape}"
    )
    assert x_test.dtype == np.float32, f"Expected float32, got {x_test.dtype}"
    assert 0.0 <= x_test.min() and x_test.max() <= 1.0, "Test pixel intensities out of bounds [0.0, 1.0]"
    assert np.all(np.isfinite(x_test)), "Test set contains NaN or Inf values!"
    assert y_test.min() == 0 and y_test.max() == NUM_CLASSES - 1, (
        f"Label range mismatch: min={y_test.min()}, max={y_test.max()}"
    )
    counts = np.bincount(y_test, minlength=NUM_CLASSES)
    assert np.all(counts == TEST_SAMPLES_PER_CLASS), (
        f"Test set class balance violated. Expected {TEST_SAMPLES_PER_CLASS} per class, got {counts}"
    )
    print(f"      - Samples Count     : {len(x_test):,} (Exactly {TEST_SAMPLES_PER_CLASS} per class across {NUM_CLASSES} classes)")
    print(f"      - Tensor Shape      : {x_test.shape} float32 in [{x_test.min():.2f}, {x_test.max():.2f}]")
    print(f"      - Label Range       : [{y_test.min()}, {y_test.max()}] with complete 47-class representation")

    # 3. Reload Model Checkpoint
    print(f"\n[4/5] Reloading trained LSTM checkpoint from: {model_path}")
    assert model_path.exists(), f"Checkpoint not found at: {model_path}"
    model = tf.keras.models.load_model(str(model_path))
    print(f"      - Model Name        : {model.name}")
    print(f"      - Input Layer Shape : {model.input_shape}")
    print(f"      - Output Layer Shape: {model.output_shape}")
    assert model.output_shape == (None, NUM_CLASSES), (
        f"Model output shape mismatch: expected (None, {NUM_CLASSES}), got {model.output_shape}"
    )

    # 4. Model Evaluation & Predictions
    print("\n[5/5] Running inference on held-out test split...")
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    
    pred_probabilities = model.predict(x_test, batch_size=batch_size, verbose=1)
    assert pred_probabilities.shape == (TEST_SAMPLES_TOTAL, NUM_CLASSES), (
        f"Prediction shape mismatch: {pred_probabilities.shape}"
    )
    assert np.all(np.isfinite(pred_probabilities)), "Prediction probabilities contain NaN/Inf!"
    
    y_pred = np.argmax(pred_probabilities, axis=-1)
    assert len(y_pred) == len(y_test), "Prediction length does not match ground truth!"
    assert y_pred.min() >= 0 and y_pred.max() <= NUM_CLASSES - 1, "Predictions contain invalid class indices!"

    # 5. Compute Comprehensive Metrics
    sklearn_acc = accuracy_score(y_test, y_pred)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    weighted_prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    prec_per_class, rec_per_class, f1_per_class, supp_per_class = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    per_class_metrics = []
    for cls_idx in range(NUM_CLASSES):
        char = class_mapping[cls_idx]
        per_class_metrics.append({
            "class_index": int(cls_idx),
            "character": char,
            "precision": float(prec_per_class[cls_idx]),
            "recall": float(rec_per_class[cls_idx]),
            "f1_score": float(f1_per_class[cls_idx]),
            "support": int(supp_per_class[cls_idx]),
        })

    # Prepare complete payload
    results_payload = {
        "metadata": {
            "model_path": str(model_path),
            "test_samples_total": TEST_SAMPLES_TOTAL,
            "test_samples_per_class": TEST_SAMPLES_PER_CLASS,
            "num_classes": NUM_CLASSES,
            "batch_size": batch_size,
            "metric_averaging_methods": ["macro", "weighted"],
        },
        "summary": {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "sklearn_accuracy": float(sklearn_acc),
            "macro_precision": float(macro_prec),
            "macro_recall": float(macro_rec),
            "macro_f1_score": float(macro_f1),
            "weighted_precision": float(weighted_prec),
            "weighted_recall": float(weighted_rec),
            "weighted_f1_score": float(weighted_f1),
        },
        "per_class_metrics": per_class_metrics,
    }

    # 6. Save Artifacts
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = METRICS_DIR / "test_evaluation.json"
    csv_path = METRICS_DIR / "classification_report.csv"

    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(results_payload, f_json, indent=2)
    print(f"\n[ARTIFACT SAVED] JSON metrics saved to: {json_path}")

    # Write classification report to CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["class_index", "character", "precision", "recall", "f1_score", "support"])
        for item in per_class_metrics:
            writer.writerow([
                item["class_index"],
                item["character"],
                f"{item['precision']:.6f}",
                f"{item['recall']:.6f}",
                f"{item['f1_score']:.6f}",
                item["support"],
            ])
        writer.writerow([])
        writer.writerow(["macro_avg", "-", f"{macro_prec:.6f}", f"{macro_rec:.6f}", f"{macro_f1:.6f}", TEST_SAMPLES_TOTAL])
        writer.writerow(["weighted_avg", "-", f"{weighted_prec:.6f}", f"{weighted_rec:.6f}", f"{weighted_f1:.6f}", TEST_SAMPLES_TOTAL])
    print(f"[ARTIFACT SAVED] CSV classification report saved to: {csv_path}")

    # 7. Print Console Summary
    print("\n" + "=" * 70)
    print("TEST EVALUATION PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Total Test Samples Evaluated : {TEST_SAMPLES_TOTAL:,}")
    print(f"Overall Test Loss            : {test_loss:.4f}")
    print(f"Overall Test Accuracy        : {test_acc * 100:.2f}% ({test_acc:.4f})")
    print(f"Macro-Average Precision      : {macro_prec * 100:.2f}% ({macro_prec:.4f})")
    print(f"Macro-Average Recall         : {macro_rec * 100:.2f}% ({macro_rec:.4f})")
    print(f"Macro-Average F1-Score       : {macro_f1 * 100:.2f}% ({macro_f1:.4f})")
    print(f"Weighted-Average Precision   : {weighted_prec * 100:.2f}% ({weighted_prec:.4f})")
    print(f"Weighted-Average Recall      : {weighted_rec * 100:.2f}% ({weighted_rec:.4f})")
    print(f"Weighted-Average F1-Score    : {weighted_f1 * 100:.2f}% ({weighted_f1:.4f})")
    print("=" * 70)

    return results_payload


if __name__ == "__main__":
    evaluate_test_set()
