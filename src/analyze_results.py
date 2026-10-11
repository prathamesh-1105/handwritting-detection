"""
Step 9: Visualization, Result Analysis, and Error Inspection Module.

This module handles:
1. Generating the 47x47 Confusion Matrix and saving both CSV and publication-quality figure.
2. Extracting and ranking the most frequent true-to-predicted misclassification pairs.
3. Ranking the 5 strongest and 5 weakest classes by F1-score with full per-class metrics.
4. Visualizing a reproducible 4x4 gallery of actual misclassified test samples with top-3 candidate rankings.
5. Saving all reproducible metrics to outputs/metrics/ and figures to outputs/figures/.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    RANDOM_SEED,
    TEST_SAMPLES_PER_CLASS,
    TEST_SAMPLES_TOTAL,
)
from src.evaluate import load_and_preprocess_test_data


def generate_confusion_matrix_and_predictions(
    model_path: Path = MODEL_SAVE_PATH,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Evaluates the model on the full test set to obtain predictions and confusion matrix.
    
    Returns:
        cm (47, 47) raw count matrix,
        cm_norm (47, 47) row-normalized percentage matrix,
        y_test (18800,) ground truth,
        y_pred (18800,) predictions,
        class_mapping dictionary
    """
    print("[1/5] Loading test data and saved LSTM model...")
    x_test, y_test, class_mapping = load_and_preprocess_test_data()

    assert model_path.exists(), f"Model not found at: {model_path}"
    model = tf.keras.models.load_model(str(model_path))

    print("[2/5] Running inference on 18,800 test samples to obtain prediction distributions...")
    probs = model.predict(x_test, batch_size=batch_size, verbose=1)
    y_pred = np.argmax(probs, axis=-1)

    assert len(y_test) == len(y_pred) == TEST_SAMPLES_TOTAL
    assert np.all((y_pred >= 0) & (y_pred < NUM_CLASSES))

    # Compute raw and normalized confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))
    # Row-normalize: divide each row by the true class sample count (400)
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

    return cm, cm_norm, y_test, y_pred, probs, x_test, class_mapping


def plot_and_save_confusion_matrix(
    cm_norm: np.ndarray,
    class_mapping: Dict[int, str],
    save_path: Path = FIGURES_DIR / "confusion_matrix.png",
) -> Path:
    """
    Plots and exports a 47x47 normalized confusion matrix heatmap.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    class_labels = [f"'{class_mapping[i]}'" for i in range(NUM_CLASSES)]

    plt.figure(figsize=(18, 16), dpi=300)
    ax = sns.heatmap(
        cm_norm * 100.0,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Accuracy / Confusion Rate (%)", "shrink": 0.8},
        linewidths=0.2,
        linecolor="#e0e0e0",
        vmin=0.0,
        vmax=100.0,
    )

    plt.title(
        "EMNIST Balanced LSTM Handwriting Recognition — Normalized Confusion Matrix (47 Classes)\n"
        "Values represent row-normalized percentages (%) per true class",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    plt.xlabel("Predicted Character Class", fontsize=12, fontweight="bold", labelpad=10)
    plt.ylabel("True Ground-Truth Character Class", fontsize=12, fontweight="bold", labelpad=10)

    # Rotate ticks for legibility
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    assert save_path.exists(), f"Failed to save confusion matrix at: {save_path}"
    print(f"[ARTIFACT SAVED] Confusion matrix figure saved to: {save_path}")
    return save_path


def export_confusion_matrix_csv(
    cm: np.ndarray,
    class_mapping: Dict[int, str],
    save_path: Path = METRICS_DIR / "confusion_matrix.csv",
) -> Path:
    """
    Saves the 47x47 raw count confusion matrix to CSV.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    class_labels = [class_mapping[i] for i in range(NUM_CLASSES)]

    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_char / pred_char"] + class_labels)
        for i in range(NUM_CLASSES):
            row_char = class_mapping[i]
            writer.writerow([row_char] + list(cm[i]))

    assert save_path.exists(), f"Failed to save confusion matrix CSV at: {save_path}"
    print(f"[ARTIFACT SAVED] Confusion matrix CSV saved to: {save_path}")
    return save_path


def extract_top_confusions(
    cm: np.ndarray,
    class_mapping: Dict[int, str],
    top_n: int = 20,
    save_path: Path = METRICS_DIR / "top_confusions.json",
) -> List[Dict]:
    """
    Extracts the most frequent off-diagonal misclassification pairs.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    confusions = []

    for true_idx in range(NUM_CLASSES):
        for pred_idx in range(NUM_CLASSES):
            if true_idx != pred_idx:
                count = int(cm[true_idx, pred_idx])
                if count > 0:
                    pct = (count / TEST_SAMPLES_PER_CLASS) * 100.0
                    confusions.append({
                        "true_class_index": true_idx,
                        "true_character": class_mapping[true_idx],
                        "predicted_class_index": pred_idx,
                        "predicted_character": class_mapping[pred_idx],
                        "misclassified_count": count,
                        "total_true_samples": TEST_SAMPLES_PER_CLASS,
                        "percentage_of_true_class": round(pct, 2),
                    })

    # Sort descending by count
    confusions.sort(key=lambda x: x["misclassified_count"], reverse=True)
    top_list = confusions[:top_n]

    payload = {
        "metadata": {
            "total_test_samples": TEST_SAMPLES_TOTAL,
            "samples_per_class": TEST_SAMPLES_PER_CLASS,
            "total_misclassifications": int(np.sum(cm) - np.trace(cm)),
            "top_n_extracted": len(top_list),
        },
        "top_confusions": top_list,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    assert save_path.exists(), f"Failed to save top confusions at: {save_path}"
    print(f"[ARTIFACT SAVED] Top confusions JSON saved to: {save_path}")
    return top_list


def analyze_strongest_weakest_classes(
    eval_json_path: Path = METRICS_DIR / "test_evaluation.json",
    save_path: Path = METRICS_DIR / "class_performance_summary.json",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Identifies the top-5 and bottom-5 classes ranked by F1-score.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_class = data["per_class_metrics"]
    # Sort descending by F1-score
    sorted_by_f1 = sorted(per_class, key=lambda x: x["f1_score"], reverse=True)

    strongest_5 = sorted_by_f1[:5]
    weakest_5 = sorted_by_f1[-5:][::-1]  # Ranked from lowest to 5th lowest

    payload = {
        "strongest_classes": strongest_5,
        "weakest_classes": weakest_5,
        "full_f1_ranking": sorted_by_f1,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    assert save_path.exists(), f"Failed to save class performance summary at: {save_path}"
    print(f"[ARTIFACT SAVED] Class performance summary saved to: {save_path}")
    return strongest_5, weakest_5


def plot_and_save_misclassified_examples(
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_mapping: Dict[int, str],
    num_samples: int = 16,
    random_seed: int = RANDOM_SEED,
    save_path: Path = FIGURES_DIR / "misclassified_examples.png",
) -> Path:
    """
    Selects reproducible misclassified test samples and generates a 4x4 visual gallery.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    misclassified_indices = np.where(y_test != y_pred)[0]
    total_mistakes = len(misclassified_indices)
    print(f"\n[4/5] Total misclassified test samples: {total_mistakes:,} / {TEST_SAMPLES_TOTAL:,} ({total_mistakes/TEST_SAMPLES_TOTAL*100:.2f}%)")

    rng = np.random.default_rng(random_seed)
    selected_indices = rng.choice(misclassified_indices, size=num_samples, replace=False)

    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(13, 14), dpi=300)

    for idx, (sample_idx, ax) in enumerate(zip(selected_indices, axes.flat)):
        img = x_test[sample_idx]
        true_cls = int(y_test[sample_idx])
        pred_cls = int(y_pred[sample_idx])
        true_char = class_mapping[true_cls]
        pred_char = class_mapping[pred_cls]
        conf = float(probs[sample_idx, pred_cls])

        # Get top-3 candidate probabilities
        sample_probs = probs[sample_idx]
        top3_indices = np.argsort(sample_probs)[::-1][:3]
        top3_str = ", ".join([f"'{class_mapping[k]}': {sample_probs[k]*100:.0f}%" for k in top3_indices])

        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])

        # Red border for misclassifications
        for spine in ax.spines.values():
            spine.set_edgecolor("#c62828")
            spine.set_linewidth(2.5)

        title_str = (
            f"Sample #{sample_idx}\n"
            f"True: '{true_char}'  |  Pred: '{pred_char}' [MISMATCH]\n"
            f"Conf: {conf * 100:.1f}%\n"
            f"Top 3: {top3_str}"
        )
        ax.set_title(title_str, fontsize=8.5, fontweight="bold", color="#b71c1c", pad=5)

    plt.suptitle(
        f"EMNIST Balanced LSTM Handwriting Recognition — Representative Misclassifications (Step 9)\n"
        f"Selected from {total_mistakes:,} Mistakes [Seed={random_seed}, Evaluated on Saved Checkpoint]",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    assert save_path.exists(), f"Failed to save misclassified gallery at: {save_path}"
    print(f"[ARTIFACT SAVED] Misclassified examples gallery saved to: {save_path}")
    return save_path


def run_step9_analysis():
    """
    Executes complete Step 9 Visualization, Result Analysis, and Error Inspection pipeline.
    """
    print("\n" + "=" * 70)
    print("STEP 9: VISUALIZATION, RESULT ANALYSIS, AND ERROR INSPECTION")
    print("=" * 70)

    # 1. Compute Confusion Matrix and Predictions
    cm, cm_norm, y_test, y_pred, probs, x_test, class_mapping = generate_confusion_matrix_and_predictions()

    # 2. Export Confusion Matrix artifacts
    cm_fig_path = plot_and_save_confusion_matrix(cm_norm, class_mapping)
    cm_csv_path = export_confusion_matrix_csv(cm, class_mapping)

    # 3. Extract Top Confusions
    print("\n[3/5] Extracting most frequent misclassification pairs...")
    top_confusions = extract_top_confusions(cm, class_mapping, top_n=10)

    # 4. Analyze Strongest and Weakest Classes
    strongest_5, weakest_5 = analyze_strongest_weakest_classes()

    # 5. Plot Misclassified Examples Gallery
    misc_fig_path = plot_and_save_misclassified_examples(
        x_test, y_test, y_pred, probs, class_mapping, num_samples=16
    )

    # 6. Print Comprehensive Summary
    print("\n" + "=" * 70)
    print("STEP 9 RESULTS SUMMARY")
    print("=" * 70)
    print("\n[TOP 5 STRONGEST CLASSES BY F1-SCORE]")
    print(f"{'Class':<8} | {'Char':<6} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11} | {'Support'}")
    print("-" * 65)
    for c in strongest_5:
        print(f"Class {c['class_index']:<2d} | '{c['character']}'  | {c['precision']*100:6.2f}%     | {c['recall']*100:6.2f}%     | {c['f1_score']*100:6.2f}%     | {c['support']}")

    print("\n[TOP 5 WEAKEST CLASSES BY F1-SCORE]")
    print(f"{'Class':<8} | {'Char':<6} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11} | {'Support'}")
    print("-" * 65)
    for c in weakest_5:
        print(f"Class {c['class_index']:<2d} | '{c['character']}'  | {c['precision']*100:6.2f}%     | {c['recall']*100:6.2f}%     | {c['f1_score']*100:6.2f}%     | {c['support']}")

    print("\n[TOP 10 MOST FREQUENT MISCLASSIFICATION PAIRS]")
    print(f"{'True Char':<12} -> {'Pred Char':<12} | {'Count':<8} | {'% of True Class'}")
    print("-" * 55)
    for pair in top_confusions:
        t_str = f"'{pair['true_character']}' (cls {pair['true_class_index']:<2d})"
        p_str = f"'{pair['predicted_character']}' (cls {pair['predicted_class_index']:<2d})"
        print(f"{t_str:<12} -> {p_str:<12} | {pair['misclassified_count']:<8} | {pair['percentage_of_true_class']:5.2f}%")

    print("\n" + "=" * 70)
    print("STEP 9 EXECUTION COMPLETE: ALL ARTIFACTS VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    run_step9_analysis()
