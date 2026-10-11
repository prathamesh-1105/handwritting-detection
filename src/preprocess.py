"""
Preprocessing and Sequential Input Preparation Module for LSTM Handwriting Recognition.

This module handles:
1. Loading raw EMNIST Balanced data via src.dataset without re-downloading.
2. Applying permanent batch transposition to restore upright character orientation.
3. Normalizing pixel values from uint8 [0, 255] to float32 [0.0, 1.0].
4. Structuring 3D sequential tensors suitable for LSTM (time_steps=28, features=28).
5. Performing a reproducible, stratified 90/10 train-validation split (101,520 train, 11,280 val).
6. Label encoding (defaulting to integer sparse categorical labels for memory efficiency).
7. Comprehensive validation checks (shapes, dtypes, ranges, NaN/Inf checks, data isolation, reproducibility).
8. Visualizing verified preprocessed samples saved to outputs/figures/.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    FIGURES_DIR,
    NUM_CLASSES,
    RANDOM_SEED,
    TEST_SAMPLES_PER_CLASS,
    TEST_SAMPLES_TOTAL,
    TIME_STEPS,
    INPUT_FEATURES,
    TRAIN_SAMPLES_TOTAL,
    TRAIN_SUBSET_PER_CLASS,
    TRAIN_SUBSET_SAMPLES,
    VAL_SUBSET_PER_CLASS,
    VAL_SUBSET_SAMPLES,
    VALIDATION_SPLIT,
)
from src.dataset import load_emnist_balanced


def correct_orientation(images: np.ndarray) -> np.ndarray:
    """
    Corrects the native Fortran column-major storage layout of EMNIST.
    
    In raw EMNIST, images are transposed and mirrored sideways. Transposing
    the 2nd and 3rd axes (H and W) across the batch restores natural upright handwriting.
    
    Args:
        images: Array of shape (N, 28, 28).
        
    Returns:
        np.ndarray: Upright images of shape (N, 28, 28).
    """
    # Swap axis 1 (rows) and axis 2 (cols) across all N samples
    return np.transpose(images, (0, 2, 1))


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel intensities from integer [0, 255] to float32 [0.0, 1.0].
    
    Neural networks (especially recurrent architectures like LSTMs) train
    substantially more stably when inputs have bounded floating-point scales.
    
    Args:
        images: Array of shape (N, 28, 28) with dtype uint8.
        
    Returns:
        np.ndarray: Array of shape (N, 28, 28) with dtype float32 in range [0.0, 1.0].
    """
    return (images.astype(np.float32)) / 255.0


def encode_labels(labels: np.ndarray, one_hot: bool = False, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """
    Encodes class labels for deep learning optimization.
    
    Technical Decision:
    - Default (one_hot=False): Keeps integer labels with dtype int32.
      Paired with Keras `sparse_categorical_crossentropy`, this avoids allocating
      large one-hot matrices, saving ~21 MB of RAM on CPU and speeding up training.
    - If one_hot=True: Encodes into binary 2D matrix of shape (N, num_classes)
      for use with standard `categorical_crossentropy`.
      
    Args:
        labels: 1D array of integer class indices.
        one_hot: If True, convert to one-hot encoded matrix.
        num_classes: Total number of target classes (47).
        
    Returns:
        np.ndarray: Integer labels (N,) or one-hot matrix (N, 47).
    """
    if one_hot:
        return np.eye(num_classes, dtype=np.float32)[labels]
    return labels.astype(np.int32)


def split_train_validation(
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    val_ratio: float = VALIDATION_SPLIT,
    random_seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the 112,800 training samples into a 90/10 stratified train-validation split.
    
    Stratification guarantees that every one of the 47 classes maintains exactly
    the same proportional representation in both the training subset (2,160/class)
    and validation subset (240/class).
    
    Returns:
        x_train, y_train, x_val, y_val, train_indices, val_indices
    """
    indices = np.arange(len(x_train_full))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=y_train_full,
    )
    
    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]
    
    return x_train, y_train, x_val, y_val, train_idx, val_idx


def validate_preprocessed_pipeline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, bool]:
    """
    Runs an exhaustive suite of programmatic integrity assertions across preprocessed splits.
    """
    checks = {}

    # 1. Shape and Sample Count Checks
    checks["train_images_shape"] = bool(x_train.shape == (TRAIN_SUBSET_SAMPLES, TIME_STEPS, INPUT_FEATURES))
    checks["val_images_shape"] = bool(x_val.shape == (VAL_SUBSET_SAMPLES, TIME_STEPS, INPUT_FEATURES))
    checks["test_images_shape"] = bool(x_test.shape == (TEST_SAMPLES_TOTAL, TIME_STEPS, INPUT_FEATURES))
    
    # 2. Data Types & Label Encoding Validation
    checks["images_float32"] = bool(
        x_train.dtype == np.float32 and x_val.dtype == np.float32 and x_test.dtype == np.float32
    )

    if y_train.ndim == 1:
        checks["labels_valid_format"] = bool(
            np.issubdtype(y_train.dtype, np.integer) and
            np.issubdtype(y_val.dtype, np.integer) and
            np.issubdtype(y_test.dtype, np.integer)
        )
        checks["labels_in_range_0_46"] = bool(
            y_train.min() == 0 and y_train.max() == num_classes - 1 and
            y_val.min() == 0 and y_val.max() == num_classes - 1 and
            y_test.min() == 0 and y_test.max() == num_classes - 1
        )
        train_int_labels = y_train
        val_int_labels = y_val
        test_int_labels = y_test
    else:
        checks["labels_valid_format"] = bool(
            y_train.shape == (len(x_train), num_classes) and
            y_val.shape == (len(x_val), num_classes) and
            y_test.shape == (len(x_test), num_classes)
        )
        checks["labels_in_range_0_46"] = bool(
            np.all(y_train.sum(axis=1) == 1.0) and
            np.all(y_val.sum(axis=1) == 1.0) and
            np.all(y_test.sum(axis=1) == 1.0)
        )
        train_int_labels = np.argmax(y_train, axis=1)
        val_int_labels = np.argmax(y_val, axis=1)
        test_int_labels = np.argmax(y_test, axis=1)

    # 3. Value Ranges
    checks["train_pixel_range_0_1"] = bool(0.0 <= x_train.min() and x_train.max() <= 1.0)
    checks["val_pixel_range_0_1"] = bool(0.0 <= x_val.min() and x_val.max() <= 1.0)
    checks["test_pixel_range_0_1"] = bool(0.0 <= x_test.min() and x_test.max() <= 1.0)

    # 4. Finite Value Checks (No NaN or Inf)
    checks["no_nans_or_infs"] = bool(
        np.isfinite(x_train).all() and np.isfinite(x_val).all() and np.isfinite(x_test).all()
    )

    # 5. Stratification Counts Across All 47 Classes
    train_counts = np.bincount(train_int_labels, minlength=num_classes)
    val_counts = np.bincount(val_int_labels, minlength=num_classes)
    test_counts = np.bincount(test_int_labels, minlength=num_classes)

    checks["stratified_train_balance"] = bool(np.all(train_counts == TRAIN_SUBSET_PER_CLASS))
    checks["stratified_val_balance"] = bool(np.all(val_counts == VAL_SUBSET_PER_CLASS))
    checks["stratified_test_balance"] = bool(np.all(test_counts == TEST_SAMPLES_PER_CLASS))

    # 6. Data Isolation (Disjoint Train and Validation Indices)
    index_intersection = np.intersect1d(train_idx, val_idx)
    checks["disjoint_train_val_indices"] = bool(len(index_intersection) == 0)
    checks["total_indices_match"] = bool(len(train_idx) + len(val_idx) == TRAIN_SAMPLES_TOTAL)

    # 7. Reproducibility Check
    # Verify that re-running train_test_split on range yields deterministic results
    t_test_1, _ = train_test_split(np.arange(100), test_size=0.1, random_state=RANDOM_SEED)
    t_test_2, _ = train_test_split(np.arange(100), test_size=0.1, random_state=RANDOM_SEED)
    checks["seed_reproducibility"] = bool(np.array_equal(t_test_1, t_test_2))

    return checks


def visualize_preprocessed_samples(
    x_train: np.ndarray,
    y_train: np.ndarray,
    class_mapping: Dict[int, str],
    output_path: Path = FIGURES_DIR / "preprocessed_samples_visualization.png",
    num_classes_to_display: int = 16,
) -> None:
    """
    Renders and saves a visual gallery of preprocessed, normalized, upright character samples.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select 16 representative classes across digits, uppercase, and lowercase
    # e.g., Digits: 0, 3, 5, 8 | Uppercase: A, B, E, G, M, R, Z | Lowercase: a, d, g, n, t
    classes_to_show = [0, 3, 5, 8, 10, 11, 14, 16, 22, 27, 35, 36, 38, 41, 43, 46]
    classes_to_show = classes_to_show[:num_classes_to_display]

    cols = 4
    rows = (len(classes_to_show) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.6 * rows))
    fig.suptitle(
        "EMNIST Balanced Preprocessed Samples (Upright Orientation, Normalized [0.0, 1.0])\n"
        "Sequential Input Tensors: (28 Time Steps × 28 Features)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    for idx, cls_idx in enumerate(classes_to_show):
        r, c = divmod(idx, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        
        # Locate the first sample matching this class in x_train
        sample_loc = np.where(y_train == cls_idx)[0][0]
        sample_img = x_train[sample_loc]
        char_label = class_mapping[cls_idx]
        
        ax.imshow(sample_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"Class {cls_idx}: '{char_label}'\nShape: (28, 28)\nRange: [{sample_img.min():.2f}, {sample_img.max():.2f}]",
            fontsize=9,
        )
        ax.axis("off")

    # Hide any unused subplots
    for idx in range(len(classes_to_show), rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[FIGURE SAVED] Preprocessed visualization saved to: {output_path}")


def prepare_dataset(
    one_hot: bool = False,
    visualize: bool = True,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Dict[int, str],
]:
    """
    Complete end-to-end Step 3 Preprocessing Pipeline.
    
    1. Loads raw EMNIST Balanced arrays from cache.
    2. Corrects image orientation via batch transposition.
    3. Normalizes pixel values from uint8 [0, 255] to float32 [0.0, 1.0].
    4. Encodes class labels (sparse integer by default, or one-hot matrix).
    5. Splits full training dataset into stratified train and validation subsets.
    6. Validates all shapes, dtypes, ranges, and data-isolation constraints.
    7. Optionally exports visualization grid.
    
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test), class_mapping
    """
    print("\n" + "=" * 70)
    print("STEP 3: EMNIST BALANCED DATA PREPROCESSING & SEQUENTIAL PREPARATION")
    print("=" * 70)

    # 1. Load Raw Data via Step 2 Dataset Loader
    print("[1/5] Loading raw EMNIST Balanced datasets from data/ cache...")
    (raw_train_x, raw_train_y), (raw_test_x, raw_test_y), class_mapping = load_emnist_balanced()

    # 2. Correct Orientation (Batch Transposition)
    print("[2/5] Applying batch transposition to restore upright handwriting orientation...")
    oriented_train_x = correct_orientation(raw_train_x)
    oriented_test_x = correct_orientation(raw_test_x)

    # 3. Normalize Pixel Intensities to Float32 [0.0, 1.0]
    print("[3/5] Normalizing pixel intensities to float32 range [0.0, 1.0]...")
    norm_train_x = normalize_pixels(oriented_train_x)
    norm_test_x = normalize_pixels(oriented_test_x)

    # 4. Encode Labels
    print(f"[4/5] Encoding class labels (one_hot={one_hot})...")
    encoded_train_y = encode_labels(raw_train_y, one_hot=one_hot)
    encoded_test_y = encode_labels(raw_test_y, one_hot=one_hot)

    # 5. Create Stratified 90/10 Train-Validation Split
    print("[5/5] Creating reproducible stratified 90/10 training-validation split...")
    x_train, y_train, x_val, y_val, train_idx, val_idx = split_train_validation(
        norm_train_x,
        encoded_train_y,
        val_ratio=VALIDATION_SPLIT,
        random_seed=RANDOM_SEED,
    )
    x_test, y_test = norm_test_x, encoded_test_y

    # 6. Run Exhaustive Validation Suite
    print("\n[VALIDATION] Running programmatic integrity checks...")
    checks = validate_preprocessed_pipeline(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        train_idx,
        val_idx,
        num_classes=NUM_CLASSES,
    )

    all_passed = all(checks.values())
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  - Check [{name}]: {status}")

    if not all_passed:
        failed = [k for k, v in checks.items() if not v]
        raise AssertionError(f"Preprocessing validation failed for checks: {failed}")

    print("\n[SUMMARY] Preprocessing and Sequential Tensor Preparation Complete:")
    print(f"  - Training Subset   : Images {x_train.shape} | Labels {y_train.shape} | Dtype: {x_train.dtype}")
    print(f"  - Validation Subset : Images {x_val.shape}   | Labels {y_val.shape}   | Dtype: {x_val.dtype}")
    print(f"  - Held-out Test Set : Images {x_test.shape}  | Labels {y_test.shape}  | Dtype: {x_test.dtype}")
    print(f"  - Sequential Shape  : 28 Time Steps × 28 Features per Image")
    print(f"  - Pixel Value Range : min={x_train.min():.4f}, max={x_train.max():.4f}")
    print(f"  - Data Isolation    : Train and Validation sample indices are strictly disjoint (0 overlap)")
    print(f"  - Stratification    : Exactly {TRAIN_SUBSET_PER_CLASS} train and {VAL_SUBSET_PER_CLASS} val samples per class across all 47 classes")

    # 7. Render Visualization
    if visualize:
        raw_train_labels = raw_train_y[train_idx]
        visualize_preprocessed_samples(x_train, raw_train_labels, class_mapping)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), class_mapping


if __name__ == "__main__":
    # Execute preprocessing pipeline and validation checks
    (x_tr, y_tr), (x_v, y_v), (x_te, y_te), mapping = prepare_dataset(one_hot=False, visualize=True)
