"""
Step 8: Handwriting Prediction Demonstration Module.

This script:
1. Selects a reproducible subset of unseen test samples using a fixed random seed.
2. Runs inference using HandwritingPredictor.
3. Generates a clear, publication-quality visual grid with true vs. predicted labels and confidence.
4. Saves the visualization to outputs/figures/handwriting_prediction_demo.png.
5. Performs rigorous programmatic verification of all prediction outputs.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR, RANDOM_SEED, TEST_SAMPLES_TOTAL
from src.predict import HandwritingPredictor


def run_prediction_demonstration(
    num_samples: int = 16,
    random_seed: int = RANDOM_SEED,
    save_path: Path = FIGURES_DIR / "handwriting_prediction_demo.png",
) -> List[Dict]:
    """
    Executes reproducible visual demonstration of handwriting predictions on unseen test data.
    
    Args:
        num_samples: Total number of samples to demonstrate (default: 16 in a 4x4 grid).
        random_seed: Seed for reproducible sample selection (default: 42).
        save_path: Output PNG filepath for visualization.
        
    Returns:
        List[Dict]: List of prediction outcome dictionaries.
    """
    print("\n" + "=" * 70)
    print("STEP 8: HANDWRITING PREDICTION DEMONSTRATION ON UNSEEN TEST SAMPLES")
    print("=" * 70)

    # 1. Initialize Predictor
    predictor = HandwritingPredictor()

    # 2. Select Reproducible Test Sample Indices
    rng = np.random.default_rng(random_seed)
    selected_indices = rng.choice(TEST_SAMPLES_TOTAL, size=num_samples, replace=False)
    print(f"\n[1/4] Selected {num_samples} reproducible test sample indices (seed={random_seed}):")
    print(f"      Indices: {list(selected_indices)}")

    # 3. Perform Inference on Selected Samples
    print("\n[2/4] Running model predictions and extracting top-5 candidate rankings...")
    results = []
    for idx in selected_indices:
        res = predictor.predict_test_sample(int(idx), top_k=5)
        results.append(res)

    # 4. Programmatic Verification Checks
    print("\n[3/4] Verifying prediction integrity assertions...")
    for i, res in enumerate(results):
        # Valid index
        assert 0 <= res["predicted_class"] <= 46, f"Invalid predicted class index: {res['predicted_class']}"
        assert 0 <= res["true_class"] <= 46, f"Invalid true class index: {res['true_class']}"
        # Confidence bounds
        conf = res["confidence"]
        assert np.isfinite(conf), "Confidence is not finite!"
        assert 0.0 <= conf <= 1.0, f"Confidence out of bounds: {conf}"
        # Top-5 sorted descending
        top_k = res["top_k"]
        assert len(top_k) == 5, f"Expected 5 top candidates, got {len(top_k)}"
        top_confs = [item["confidence"] for item in top_k]
        for k_idx in range(len(top_confs) - 1):
            assert top_confs[k_idx] >= top_confs[k_idx + 1] - 1e-6, (
                f"Top-k confidences not in descending order: {top_confs}"
            )
        # Ground truth consistency
        assert res["true_character"] == predictor.class_mapping[res["true_class"]], "Ground truth mapping mismatch!"
        assert res["predicted_character"] == predictor.class_mapping[res["predicted_class"]], "Predicted mapping mismatch!"

    print("      - All 16 predictions passed class-range checks [0, 46].")
    print("      - All confidence values verified within [0.0, 1.0].")
    print("      - All top-5 candidate lists strictly verified in descending order.")
    print("      - Ground truth labels verified against official EMNIST class mapping.")

    # 5. Generate Visual Demonstration Figure
    print(f"\n[4/4] Generating visual prediction gallery at: {save_path}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(13, 14), dpi=300)

    correct_count = 0
    for idx, (res, ax) in enumerate(zip(results, axes.flat)):
        img = res["preprocessed_image"]
        true_char = res["true_character"]
        pred_char = res["predicted_character"]
        conf = res["confidence"]
        is_correct = res["is_correct"]

        if is_correct:
            correct_count += 1
            status_text = "CORRECT"
            title_color = "#1b5e20"  # Dark green
            box_color = "#2e7d32"
            symbol = "[OK]"
        else:
            status_text = "MISMATCH"
            title_color = "#b71c1c"  # Dark red
            box_color = "#c62828"
            symbol = "[MISMATCH]"

        # Show normalized handwriting image
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])

        # Colored border indicating prediction outcome
        for spine in ax.spines.values():
            spine.set_edgecolor(box_color)
            spine.set_linewidth(2.5)

        # Title formatting with true vs pred and confidence
        title_str = (
            f"Sample #{res['test_index']}\n"
            f"True: '{true_char}'  |  Pred: '{pred_char}' {symbol}\n"
            f"Conf: {conf * 100:.1f}% ({status_text})"
        )
        ax.set_title(title_str, fontsize=10, fontweight="bold", color=title_color, pad=6)

    plt.suptitle(
        f"EMNIST Balanced LSTM Handwriting Recognition — Unseen Test Demo (Step 8)\n"
        f"Subset Accuracy: {correct_count}/{num_samples} ({correct_count / num_samples * 100:.1f}%) "
        f"[Seed={random_seed}, Evaluated on Saved Checkpoint]",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    assert save_path.exists(), f"Failed to save figure at: {save_path}"
    print(f"      - Figure successfully saved ({save_path.stat().st_size:,} bytes).")

    # 6. Print Console Table
    print("\n" + "=" * 70)
    print("DEMO SAMPLES PREDICTION SUMMARY TABLE")
    print("=" * 70)
    header = f"{'Sample':<8} | {'True Char':<10} | {'Pred Char':<10} | {'Conf':<9} | {'Status':<12} | {'Top-3 Candidates'}"
    print(header)
    print("-" * 70)
    for res in results:
        top_str = ", ".join([f"'{c['character']}' ({c['confidence']*100:.1f}%)" for c in res["top_k"][:3]])
        status = "[CORRECT]" if res["is_correct"] else "[MISMATCH]"
        print(
            f"#{res['test_index']:<7} | "
            f"'{res['true_character']}' (cls {res['true_class']:<2}) | "
            f"'{res['predicted_character']}' (cls {res['predicted_class']:<2}) | "
            f"{res['confidence']*100:5.1f}%   | "
            f"{status:<12} | "
            f"{top_str}"
        )
    print("-" * 70)
    print(f"Demo Subset Accuracy : {correct_count}/{num_samples} ({correct_count / num_samples * 100:.1f}%)")
    print(f"Saved Figure Path    : {save_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_prediction_demonstration()
