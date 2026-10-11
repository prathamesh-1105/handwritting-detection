"""
Step 6: Validation, Hyperparameter Review, and Overfitting Analysis Module.

This module performs:
1. Direct ingestion and verification of training metrics from outputs/metrics/training_history.json.
2. Learning curve plotting (loss and accuracy subplots) saved to outputs/figures/training_history_curves.png.
3. Quantitative overfitting analysis (generalization gaps, epoch-to-epoch loss deltas).
4. Inspection of hyperparameter configurations.
"""

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

from src.config import (
    BATCH_SIZE,
    DENSE_UNITS,
    DROPOUT_RATE,
    EPOCHS,
    FIGURES_DIR,
    HISTORY_SAVE_PATH,
    LEARNING_RATE,
    LSTM_UNITS_1,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    PATIENCE,
    RANDOM_SEED,
)


def load_and_validate_history(history_path: Path = HISTORY_SAVE_PATH) -> Dict:
    """
    Loads and validates the training history JSON file.
    
    Args:
        history_path: Path to outputs/metrics/training_history.json.
        
    Returns:
        Dict: Parsed JSON data.
    """
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found at: {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hist = data.get("history", {})
    required_keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
    for k in required_keys:
        if k not in hist:
            raise KeyError(f"Missing required metric key '{k}' in history JSON.")

    lengths = {k: len(hist[k]) for k in required_keys}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent metric array lengths in history: {lengths}")

    return data


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: Path = FIGURES_DIR / "training_history_curves.png",
) -> Path:
    """
    Generates and saves a two-subplot figure of training and validation loss/accuracy.
    
    Args:
        history: Dictionary containing loss, accuracy, val_loss, and val_accuracy lists.
        save_path: Destination PNG filepath.
        
    Returns:
        Path: Verified output filepath.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history["loss"]) + 1)
    best_epoch_idx = int(np.argmin(history["val_loss"]))
    best_epoch = best_epoch_idx + 1

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # ---------------- Subplot 1: Loss Curves ----------------
    ax_loss.plot(
        epochs,
        history["loss"],
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="Training Loss",
    )
    ax_loss.plot(
        epochs,
        history["val_loss"],
        color="#d62728",
        linewidth=2.2,
        marker="s",
        markersize=4.5,
        label="Validation Loss",
    )
    # Highlight best epoch
    ax_loss.axvline(
        x=best_epoch,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Best Checkpoint (Epoch {best_epoch})",
    )
    ax_loss.scatter(
        [best_epoch],
        [history["val_loss"][best_epoch_idx]],
        color="#2ca02c",
        s=80,
        zorder=5,
    )
    ax_loss.set_title("Training and Validation Loss vs. Epoch", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss (Sparse Categorical Cross-Entropy)", fontsize=11)
    ax_loss.set_xticks(epochs)
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(loc="upper right", frameon=True, fontsize=10)

    # ---------------- Subplot 2: Accuracy Curves ----------------
    ax_acc.plot(
        epochs,
        [acc * 100 for acc in history["accuracy"]],
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="Training Accuracy",
    )
    ax_acc.plot(
        epochs,
        [acc * 100 for acc in history["val_accuracy"]],
        color="#2ca02c",
        linewidth=2.2,
        marker="s",
        markersize=4.5,
        label="Validation Accuracy",
    )
    # Highlight best epoch
    ax_acc.axvline(
        x=best_epoch,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label=f"Best Checkpoint (Epoch {best_epoch})",
    )
    ax_acc.set_title("Training and Validation Accuracy vs. Epoch", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_xticks(epochs)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(loc="lower right", frameon=True, fontsize=10)

    plt.suptitle(
        "EMNIST Balanced LSTM Handwriting Recognition — Learning Curves (Step 6)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    if not save_path.exists():
        raise RuntimeError(f"Failed to generate figure at {save_path}")

    return save_path


def analyze_overfitting(history: Dict[str, List[float]]) -> Dict:
    """
    Performs quantitative overfitting and convergence analysis.
    
    Args:
        history: Metric history dictionary.
        
    Returns:
        Dict: Computed statistical metrics.
    """
    losses = history["loss"]
    accuracies = history["accuracy"]
    val_losses = history["val_loss"]
    val_accuracies = history["val_accuracy"]
    total_epochs = len(losses)

    # Best epoch by minimum validation loss
    best_idx = int(np.argmin(val_losses))
    best_epoch = best_idx + 1

    # Accuracy gaps (Train - Val)
    gap_best = accuracies[best_idx] - val_accuracies[best_idx]
    gap_final = accuracies[-1] - val_accuracies[-1]

    # Epochs where val_loss increased from previous epoch
    val_loss_increases = []
    for i in range(1, total_epochs):
        delta = val_losses[i] - val_losses[i - 1]
        if delta > 0:
            val_loss_increases.append({
                "epoch": i + 1,
                "prev_epoch": i,
                "prev_val_loss": val_losses[i - 1],
                "curr_val_loss": val_losses[i],
                "delta": delta,
                "percentage_increase": (delta / val_losses[i - 1]) * 100,
            })

    # Epochs where val_accuracy dropped from previous epoch
    val_acc_drops = []
    for i in range(1, total_epochs):
        delta = val_accuracies[i] - val_accuracies[i - 1]
        if delta < 0:
            val_acc_drops.append({
                "epoch": i + 1,
                "prev_epoch": i,
                "prev_val_acc": val_accuracies[i - 1],
                "curr_val_acc": val_accuracies[i],
                "delta": delta,
            })

    return {
        "total_epochs": total_epochs,
        "best_epoch": best_epoch,
        "best_val_loss": val_losses[best_idx],
        "best_val_accuracy": val_accuracies[best_idx],
        "best_train_loss": losses[best_idx],
        "best_train_accuracy": accuracies[best_idx],
        "gap_best_epoch": gap_best,
        "final_epoch": total_epochs,
        "final_val_loss": val_losses[-1],
        "final_val_accuracy": val_accuracies[-1],
        "final_train_loss": losses[-1],
        "final_train_accuracy": accuracies[-1],
        "gap_final_epoch": gap_final,
        "val_loss_increases": val_loss_increases,
        "val_acc_drops": val_acc_drops,
    }


def run_step6_analysis() -> Tuple[Dict, Path, Dict]:
    """
    Executes the complete Step 6 validation and overfitting analysis pipeline.
    
    Returns:
        Tuple: (history_data, figure_path, analysis_results)
    """
    print("\n" + "=" * 70)
    print("STEP 6: VALIDATION, HYPERPARAMETER REVIEW, AND OVERFITTING ANALYSIS")
    print("=" * 70)

    # 1. Load and validate history
    print("[1/4] Loading and validating training history...")
    data = load_and_validate_history()
    hist = data["history"]
    num_epochs = len(hist["loss"])
    print(f"      Verified metric arrays: 4 keys, {num_epochs} entries each.")

    # 2. Analyze overfitting
    print("\n[2/4] Performing quantitative convergence and gap analysis...")
    analysis = analyze_overfitting(hist)
    print(f"      - Best Epoch (min val_loss)     : Epoch {analysis['best_epoch']}")
    print(f"      - Minimum Validation Loss       : {analysis['best_val_loss']:.6f}")
    print(f"      - Validation Accuracy at Best   : {analysis['best_val_accuracy']*100:.2f}%")
    print(f"      - Training Accuracy at Best     : {analysis['best_train_accuracy']*100:.2f}%")
    print(f"      - Accuracy Gap at Best Epoch    : {analysis['gap_best_epoch']*100:.4f}%")
    print(f"      - Final Epoch (Epoch 15) Gap    : {analysis['gap_final_epoch']*100:.4f}%")
    print(f"      - Epochs with Val Loss Increase : {[item['epoch'] for item in analysis['val_loss_increases']]}")

    # 3. Generate learning curve visualization
    print("\n[3/4] Plotting and saving learning curves...")
    figure_path = plot_learning_curves(hist)
    print(f"      - Learning curve figure saved to: {figure_path}")
    print(f"      - Figure file size              : {figure_path.stat().st_size} bytes")

    # 4. Hyperparameter Summary
    print("\n[4/4] Reviewing baseline architecture hyperparameters...")
    print(f"      - LSTM Units                    : {LSTM_UNITS_1}")
    print(f"      - Dropout Rate                  : {DROPOUT_RATE}")
    print(f"      - Dense Head Units              : {DENSE_UNITS}")
    print(f"      - Learning Rate                 : {LEARNING_RATE} (Adam)")
    print(f"      - Batch Size                    : {BATCH_SIZE}")
    print(f"      - Maximum Epochs                : {EPOCHS}")
    print(f"      - EarlyStopping Patience        : {PATIENCE} (monitor='val_loss', mode='min')")
    print(f"      - ModelCheckpoint Monitor       : val_loss (mode='min', save_best_only=True)")

    print("\n" + "=" * 70)
    print("STEP 6 ANALYSIS COMPLETE: ALL AUDIT CHECKS PASSED")
    print("=" * 70)

    return data, figure_path, analysis


if __name__ == "__main__":
    run_step6_analysis()
