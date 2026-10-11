"""
Step 8: Inference and Prediction Module for LSTM Handwriting Recognition.

This module provides:
1. HandwritingPredictor class: loads the trained LSTM model and official EMNIST class mapping.
2. Single-image prediction with robust input validation (shapes, pixel bounds, dtypes).
3. Top-k candidate extraction sorted in descending confidence order.
4. Test-sample prediction by index with ground-truth comparison.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.config import (
    DATA_DIR,
    MAPPING_FILENAME,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    TEST_SAMPLES_TOTAL,
)
from src.dataset import load_class_mapping, load_emnist_balanced
from src.preprocess import correct_orientation, encode_labels, normalize_pixels


class HandwritingPredictor:
    """
    Inference engine for genuine LSTM handwriting recognition.
    """

    def __init__(
        self,
        model_path: Path = MODEL_SAVE_PATH,
        mapping_path: Path = DATA_DIR / MAPPING_FILENAME,
    ):
        """
        Initializes the predictor by loading the trained model and official class mapping.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found at: {mapping_path}")

        print(f"[PREDICTOR] Loading trained model from: {model_path}")
        self.model = tf.keras.models.load_model(str(model_path))
        self.class_mapping = load_class_mapping(mapping_path)
        
        # Verify model compatibility
        assert self.model.output_shape == (None, NUM_CLASSES), (
            f"Model output shape mismatch: expected (None, {NUM_CLASSES}), got {self.model.output_shape}"
        )
        assert len(self.class_mapping) == NUM_CLASSES, (
            f"Class mapping mismatch: expected {NUM_CLASSES} classes, got {len(self.class_mapping)}"
        )

        # Lazy cache for test dataset
        self._x_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None

    def _ensure_test_data_loaded(self) -> None:
        """Loads and preprocessed test data once on demand."""
        if self._x_test is None or self._y_test is None:
            print("[PREDICTOR] Loading preprocessed test dataset for index-based inference...")
            _, (raw_test_x, raw_test_y), _ = load_emnist_balanced()
            oriented = correct_orientation(raw_test_x)
            self._x_test = normalize_pixels(oriented)
            self._y_test = encode_labels(raw_test_y, one_hot=False)

    def predict_image(
        self,
        image: np.ndarray,
        is_raw: bool = False,
        top_k: int = 5,
    ) -> Dict:
        """
        Predicts character for a single 28x28 handwriting image.
        
        Args:
            image: NumPy array of shape (28, 28) or (1, 28, 28).
            is_raw: If True, applies batch orientation correction and 255.0 normalization.
                    If False, assumes image is already normalized float32 in [0.0, 1.0].
            top_k: Number of highest-confidence predictions to return (default: 5).
            
        Returns:
            Dict: Prediction results including top prediction, confidence, and top-k candidates.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")

        # Squeeze potential trailing channel dimension: (28, 28, 1) -> (28, 28)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)

        # Check shapes
        if image.shape == (28, 28):
            img_tensor = image[np.newaxis, :, :]  # Shape: (1, 28, 28)
        elif image.shape == (1, 28, 28):
            img_tensor = image
        else:
            raise ValueError(
                f"Invalid image shape: {image.shape}. Expected (28, 28) or (1, 28, 28)."
            )

        # Preprocessing if raw
        if is_raw:
            oriented = correct_orientation(img_tensor)
            norm_img = normalize_pixels(oriented)
        else:
            norm_img = img_tensor.astype(np.float32)

        # Rigorous input assertions
        if not np.all(np.isfinite(norm_img)):
            raise ValueError("Input image contains NaN or Inf values!")

        min_val, max_val = float(norm_img.min()), float(norm_img.max())
        if min_val < -1e-4 or max_val > 1.0001:
            raise ValueError(
                f"Pixel values out of expected range [0.0, 1.0]: min={min_val:.4f}, max={max_val:.4f}. "
                "If passing unnormalized uint8 [0, 255] pixels, set is_raw=True."
            )

        # Model forward inference
        probabilities = self.model.predict(norm_img, verbose=0)[0]
        
        # Verify valid probability distribution
        assert len(probabilities) == NUM_CLASSES, f"Expected {NUM_CLASSES} probabilities."
        assert np.all(np.isfinite(probabilities)), "Model output probabilities contain NaN/Inf!"
        row_sum = float(np.sum(probabilities))
        assert np.isclose(row_sum, 1.0, atol=1e-4), f"Probabilities do not sum to 1.0: sum={row_sum}"

        # Top prediction
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        predicted_char = self.class_mapping[predicted_idx]

        # Top-k candidates
        k = max(1, min(top_k, NUM_CLASSES))
        sorted_indices = np.argsort(probabilities)[::-1][:k]
        top_k_list = []
        for rank, idx in enumerate(sorted_indices, start=1):
            top_k_list.append({
                "rank": rank,
                "class_index": int(idx),
                "character": self.class_mapping[int(idx)],
                "confidence": float(probabilities[idx]),
            })

        return {
            "predicted_class": predicted_idx,
            "predicted_character": predicted_char,
            "confidence": confidence,
            "top_k": top_k_list,
            "all_probabilities": probabilities,
            "preprocessed_image": norm_img[0],
        }

    def predict_test_sample(self, test_index: int, top_k: int = 5) -> Dict:
        """
        Runs inference on a specific sample from the held-out test split.
        
        Args:
            test_index: Zero-indexed integer in [0, 18799].
            top_k: Number of top candidate predictions to return.
            
        Returns:
            Dict: Prediction dictionary with ground-truth labels and correctness status.
        """
        if not (0 <= test_index < TEST_SAMPLES_TOTAL):
            raise IndexError(
                f"Test index {test_index} out of bounds: must be between 0 and {TEST_SAMPLES_TOTAL - 1}."
            )

        self._ensure_test_data_loaded()
        assert self._x_test is not None and self._y_test is not None

        sample_img = self._x_test[test_index]
        true_label = int(self._y_test[test_index])
        true_char = self.class_mapping[true_label]

        result = self.predict_image(sample_img, is_raw=False, top_k=top_k)
        result["test_index"] = test_index
        result["true_class"] = true_label
        result["true_character"] = true_char
        result["is_correct"] = bool(result["predicted_class"] == true_label)

        return result


def main():
    """Quick verification of HandwritingPredictor functionality."""
    predictor = HandwritingPredictor()
    print("\n[VERIFICATION] Running single test-sample prediction (Index 0)...")
    res = predictor.predict_test_sample(test_index=0, top_k=5)
    print(f"  Test Index         : {res['test_index']}")
    print(f"  Ground Truth       : Class {res['true_class']} ('{res['true_character']}')")
    print(f"  Predicted Character: Class {res['predicted_class']} ('{res['predicted_character']}')")
    print(f"  Confidence         : {res['confidence'] * 100:.2f}%")
    print(f"  Correct Match?     : {res['is_correct']}")
    print("  Top 5 Candidates   :")
    for item in res["top_k"]:
        print(f"    #{item['rank']}: Class {item['class_index']:2d} ('{item['character']}') -> {item['confidence']*100:.2f}%")


if __name__ == "__main__":
    main()
