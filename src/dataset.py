"""
EMNIST Balanced Dataset Loader and Exploration Module.

This module handles:
1. Downloading and caching official EMNIST Balanced gzip files from NIST.
2. Parsing IDX binary formats for images and labels.
3. Loading the official 47-class label mapping.
4. Validating dataset integrity, distributions, shapes, and pixel ranges.
5. Visualizing raw vs. transposed image orientation for deep learning analysis.
"""

import gzip
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import requests

from src.config import (
    DATA_DIR,
    FIGURES_DIR,
    MAPPING_FILENAME,
    NIST_EMNIST_URL,
    NUM_CLASSES,
    TEST_IMAGES_FILENAME,
    TEST_LABELS_FILENAME,
    TEST_SAMPLES_PER_CLASS,
    TEST_SAMPLES_TOTAL,
    TRAIN_IMAGES_FILENAME,
    TRAIN_LABELS_FILENAME,
    TRAIN_SAMPLES_PER_CLASS,
    TRAIN_SAMPLES_TOTAL,
)


def get_required_file_paths(data_dir: Path = DATA_DIR) -> Dict[str, Path]:
    """Returns absolute paths for all required EMNIST Balanced files."""
    return {
        "train_images": data_dir / TRAIN_IMAGES_FILENAME,
        "train_labels": data_dir / TRAIN_LABELS_FILENAME,
        "test_images": data_dir / TEST_IMAGES_FILENAME,
        "test_labels": data_dir / TEST_LABELS_FILENAME,
        "mapping": data_dir / MAPPING_FILENAME,
    }


def download_emnist_balanced(data_dir: Path = DATA_DIR, timeout: int = 60) -> None:
    """
    Downloads and caches the official EMNIST Balanced dataset from NIST.
    
    Uses an efficient HTTP Range request on the official NIST gzip.zip archive
    to stream only the first ~33 MB containing the complete balanced split,
    avoiding downloading the entire 561 MB archive.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    file_paths = get_required_file_paths(data_dir)

    # Check if all files already exist and have non-zero size
    all_exist = all(p.exists() and p.stat().st_size > 0 for p in file_paths.values())
    if all_exist:
        print("[CACHE] All EMNIST Balanced files are already cached in data/. Skipping download.")
        return

    print(f"[DOWNLOAD] Initiating download from official NIST archive:")
    print(f"           Source URL: {NIST_EMNIST_URL}")
    print(f"           Target Directory: {data_dir}")

    # The EMNIST Balanced files reside within the first 33,028,000 bytes of gzip.zip
    range_end = 33028000
    headers = {"Range": f"bytes=0-{range_end}"}

    try:
        response = requests.get(NIST_EMNIST_URL, headers=headers, stream=True, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"Network error while connecting to NIST server: {exc}") from exc

    if response.status_code == 206:
        raw_stream = response.content
        print(f"[DOWNLOAD] Successfully received partial byte stream ({len(raw_stream):,} bytes).")
        extracted_count = _extract_balanced_entries_from_stream(raw_stream, data_dir)
        if extracted_count < 5:
            print(f"[FALLBACK] Incomplete extraction from byte stream ({extracted_count}/5 files). Falling back to full archive download...")
            _download_full_archive_fallback(NIST_EMNIST_URL, data_dir, timeout=timeout)
    else:
        print(f"[FALLBACK] Range request returned status {response.status_code}. Falling back to standard download...")
        _download_full_archive_fallback(NIST_EMNIST_URL, data_dir, timeout=timeout)

    # Validate that all required files now exist
    missing = [name for name, p in file_paths.items() if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Failed to extract required files: {missing}")

    print("[SUCCESS] All EMNIST Balanced files successfully verified on disk.")


def _extract_balanced_entries_from_stream(raw_data: bytes, target_dir: Path) -> int:
    """Extracts balanced split zip entries from a raw ZIP byte stream. Returns count of extracted files."""
    pos = 0
    extracted_count = 0
    target_files = {
        TRAIN_IMAGES_FILENAME,
        TRAIN_LABELS_FILENAME,
        TEST_IMAGES_FILENAME,
        TEST_LABELS_FILENAME,
        MAPPING_FILENAME,
    }

    while pos < len(raw_data) - 30:
        if raw_data[pos : pos + 4] != b"PK\x03\x04":
            break

        header = struct.unpack("<4sHHHHHIIIHH", raw_data[pos : pos + 30])
        method = header[3]
        comp_size = header[6]
        uncomp_size = header[7]
        fname_len = header[8]
        extra_len = header[9]

        fname = raw_data[pos + 30 : pos + 30 + fname_len].decode("utf-8")
        data_start = pos + 30 + fname_len + extra_len
        data_end = data_start + comp_size

        if data_end > len(raw_data):
            break

        comp_data = raw_data[data_start:data_end]
        base_name = os.path.basename(fname)
        if base_name in target_files:
            if method == 8:  # Deflate
                decomp_data = zlib.decompress(comp_data, -zlib.MAX_WBITS)
            elif method == 0:  # Stored (no compression)
                decomp_data = comp_data
            else:
                raise ValueError(f"Unsupported ZIP compression method: {method}")

            dest_file = target_dir / base_name
            with open(dest_file, "wb") as f_out:
                f_out.write(decomp_data)

            extracted_count += 1
            print(f"[EXTRACT] {base_name} ({len(decomp_data):,} bytes)")

        pos = data_end

    return extracted_count


def _download_full_archive_fallback(url: str, target_dir: Path, timeout: int = 120) -> None:
    """Fallback download if HTTP Range request is not honored."""
    import zipfile
    temp_zip = target_dir / "temp_emnist.zip"
    try:
        print(f"[FALLBACK] Downloading full archive to {temp_zip}...")
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(temp_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        target_files = {
            TRAIN_IMAGES_FILENAME,
            TRAIN_LABELS_FILENAME,
            TEST_IMAGES_FILENAME,
            TEST_LABELS_FILENAME,
            MAPPING_FILENAME,
        }

        with zipfile.ZipFile(temp_zip, "r") as zf:
            for member in zf.namelist():
                base_name = os.path.basename(member)
                if base_name in target_files:
                    with zf.open(member) as src, open(target_dir / base_name, "wb") as dst:
                        dst.write(src.read())
                    print(f"[FALLBACK EXTRACT] {base_name}")
    finally:
        if temp_zip.exists():
            temp_zip.unlink()


def parse_idx_images(file_path: Path) -> np.ndarray:
    """
    Parses an IDX3 image file (gzip-compressed or raw).
    
    Header structure:
    - Magic number: 4 bytes (2051 = 0x00000803 for unsigned byte images)
    - Number of images: 4 bytes (uint32 big-endian)
    - Number of rows: 4 bytes (uint32 big-endian = 28)
    - Number of columns: 4 bytes (uint32 big-endian = 28)
    
    Returns:
        np.ndarray: Array of shape (num_images, 28, 28) and dtype uint8.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {file_path}")

    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid IDX3 magic number in {file_path.name}: expected 2051, got {magic}")

        raw_buffer = f.read()
        images = np.frombuffer(raw_buffer, dtype=np.uint8)

        expected_elements = num_images * rows * cols
        if images.size != expected_elements:
            raise ValueError(
                f"Size mismatch in {file_path.name}: expected {expected_elements} elements, got {images.size}"
            )

        return images.reshape(num_images, rows, cols)


def parse_idx_labels(file_path: Path) -> np.ndarray:
    """
    Parses an IDX1 label file (gzip-compressed or raw).
    
    Header structure:
    - Magic number: 4 bytes (2049 = 0x00000801 for unsigned byte 1D vectors)
    - Number of items: 4 bytes (uint32 big-endian)
    
    Returns:
        np.ndarray: 1D array of shape (num_items,) and dtype uint8.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Label file does not exist: {file_path}")

    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rb") as f:
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid IDX1 magic number in {file_path.name}: expected 2049, got {magic}")

        raw_buffer = f.read()
        labels = np.frombuffer(raw_buffer, dtype=np.uint8)

        if labels.size != num_items:
            raise ValueError(f"Size mismatch in {file_path.name}: expected {num_items} labels, got {labels.size}")

        return labels


def load_class_mapping(mapping_path: Path) -> Dict[int, str]:
    """
    Parses the official EMNIST mapping text file.
    
    Each line contains:
        <class_index_int> <ascii_code_int>
        
    Returns:
        Dict[int, str]: Mapping from integer class index to character representation.
    """
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")

    class_to_char = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                cls_idx = int(parts[0])
                ascii_code = int(parts[1])
                class_to_char[cls_idx] = chr(ascii_code)
            else:
                raise ValueError(f"Malformed mapping format at line {line_num}: '{line}'")

    if len(class_to_char) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} classes in mapping, found {len(class_to_char)}")

    return class_to_char


def load_emnist_balanced(data_dir: Path = DATA_DIR) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Dict[int, str]]:
    """
    High-level API to load and verify the EMNIST Balanced dataset.
    
    Ensures files are downloaded and cached, parses IDX images and labels,
    loads the class mapping, and strictly validates counts and types.
    
    Returns:
        (x_train, y_train), (x_test, y_test), class_mapping
    """
    # 1. Download/Verify Cache
    download_emnist_balanced(data_dir)
    file_paths = get_required_file_paths(data_dir)

    # 2. Parse Raw Arrays
    print("\n[PARSING] Reading IDX binary files into NumPy arrays...")
    x_train = parse_idx_images(file_paths["train_images"])
    y_train = parse_idx_labels(file_paths["train_labels"])
    x_test = parse_idx_images(file_paths["test_images"])
    y_test = parse_idx_labels(file_paths["test_labels"])
    class_mapping = load_class_mapping(file_paths["mapping"])

    # 3. Rigorous Count & Shape Assertions
    assert x_train.shape == (TRAIN_SAMPLES_TOTAL, 28, 28), f"Incorrect train images shape: {x_train.shape}"
    assert y_train.shape == (TRAIN_SAMPLES_TOTAL,), f"Incorrect train labels shape: {y_train.shape}"
    assert x_test.shape == (TEST_SAMPLES_TOTAL, 28, 28), f"Incorrect test images shape: {x_test.shape}"
    assert y_test.shape == (TEST_SAMPLES_TOTAL,), f"Incorrect test labels shape: {y_test.shape}"
    assert len(class_mapping) == NUM_CLASSES, f"Incorrect mapping length: {len(class_mapping)}"

    return (x_train, y_train), (x_test, y_test), class_mapping


def explore_and_visualize_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_mapping: Dict[int, str],
    output_figure_path: Path = FIGURES_DIR / "emnist_samples_exploration.png",
) -> None:
    """
    Inspects dataset statistics and generates a visual exploration grid.
    
    Specifically investigates and demonstrates the EMNIST image orientation quirk:
    - Raw format: Fortran column-major orientation (appears transposed sideways).
    - Transposed format: Standard row-major orientation (upright character).
    """
    print("\n" + "=" * 70)
    print("EMNIST BALANCED DATASET EXPLORATION & AUDIT")
    print("=" * 70)

    # 1. Data Shapes and Types
    print(f"Training Images Shape : {x_train.shape} | Dtype: {x_train.dtype}")
    print(f"Training Labels Shape : {y_train.shape} | Dtype: {y_train.dtype}")
    print(f"Test Images Shape     : {x_test.shape}  | Dtype: {x_test.dtype}")
    print(f"Test Labels Shape     : {y_test.shape}  | Dtype: {y_test.dtype}")

    # 2. Pixel Value Range
    print(f"\nPixel Value Range (Train): min={x_train.min()}, max={x_train.max()} (Raw integer grayscale: 0-255)")
    print(f"Pixel Value Range (Test) : min={x_test.min()}, max={x_test.max()}")

    # 3. Label Range and Class Balance
    unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
    unique_test_labels, test_counts = np.unique(y_test, return_counts=True)

    print(f"\nTotal Unique Classes (Train): {len(unique_train_labels)} (Labels: {unique_train_labels.min()} to {unique_train_labels.max()})")
    print(f"Total Unique Classes (Test) : {len(unique_test_labels)} (Labels: {unique_test_labels.min()} to {unique_test_labels.max()})")
    print(f"Train Samples per Class     : exactly {train_counts[0]} for all classes? {bool(np.all(train_counts == TRAIN_SAMPLES_PER_CLASS))}")
    print(f"Test Samples per Class      : exactly {test_counts[0]} for all classes? {bool(np.all(test_counts == TEST_SAMPLES_PER_CLASS))}")

    # 4. Official Class Mapping Table
    print("\nOfficial 47-Class Mapping Table:")
    digits = [f"{i}: '{class_mapping[i]}'" for i in range(10)]
    uppercase = [f"{i}: '{class_mapping[i]}'" for i in range(10, 36)]
    lowercase = [f"{i}: '{class_mapping[i]}'" for i in range(36, 47)]
    print(f"  Digits (0-9)    [10]: {', '.join(digits)}")
    print(f"  Uppercase (A-Z) [26]: {', '.join(uppercase)}")
    print(f"  Lowercase       [11]: {', '.join(lowercase)}")

    # 5. Orientation Documentation
    print("\n" + "-" * 70)
    print("IMAGE ORIENTATION INVESTIGATION (Objective 10)")
    print("-" * 70)
    print("Observation: EMNIST characters in their raw IDX format are stored in column-major")
    print("order (Fortran array layout). When loaded into NumPy in default row-major order,")
    print("images appear transposed (rotated 90 degrees anti-clockwise and flipped horizontally).")
    print("Fix verified: Applying array transposition (img.T or np.transpose(img, (0, 2, 1)))")
    print("restores the upright natural handwriting orientation.")
    print("Note: In accordance with Step 2 objectives, images are preserved in their raw format")
    print("for exploration; the definitive transformation will be applied during Step 3 preprocessing.")

    # 6. Generate Representative Exploration Figure
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select 8 diverse character classes for demonstration
    sample_classes = [0, 3, 7, 10, 11, 23, 36, 41]  # 0, 3, 7, A, B, N, a, g
    num_samples = len(sample_classes)

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 2.2 * num_samples))
    fig.suptitle(
        "EMNIST Balanced: Raw vs. Transposed Orientation Comparison\n(Investigating Fortran Column-Major Storage)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    for i, cls_idx in enumerate(sample_classes):
        # Pick the first matching sample from training set
        sample_loc = np.where(y_train == cls_idx)[0][0]
        raw_img = x_train[sample_loc]
        transposed_img = raw_img.T
        char = class_mapping[cls_idx]

        # Column 1: Raw
        ax_raw = axes[i, 0]
        ax_raw.imshow(raw_img, cmap="gray")
        ax_raw.set_title(f"Class {cls_idx} ('{char}') - RAW", fontsize=10)
        ax_raw.axis("off")

        # Column 2: Transposed
        ax_tr = axes[i, 1]
        ax_tr.imshow(transposed_img, cmap="gray")
        ax_tr.set_title(f"Class {cls_idx} ('{char}') - TRANSPOSED (Upright)", fontsize=10, color="darkgreen", fontweight="bold")
        ax_tr.axis("off")

    plt.tight_layout()
    plt.savefig(output_figure_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[FIGURE SAVED] Exploration plot saved to: {output_figure_path}")


if __name__ == "__main__":
    # Execute full Step 2 loading and exploration pipeline
    (train_x, train_y), (test_x, test_y), mapping = load_emnist_balanced()
    explore_and_visualize_dataset(train_x, train_y, test_x, test_y, mapping)
