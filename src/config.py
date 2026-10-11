"""
Configuration and Hyperparameters for LSTM Handwriting Recognition Project.
"""

from pathlib import Path

# Base Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# Ensure essential project directories exist
for directory in [DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Official EMNIST Remote Source & Local Filenames
NIST_EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
TRAIN_IMAGES_FILENAME = "emnist-balanced-train-images-idx3-ubyte.gz"
TRAIN_LABELS_FILENAME = "emnist-balanced-train-labels-idx1-ubyte.gz"
TEST_IMAGES_FILENAME = "emnist-balanced-test-images-idx3-ubyte.gz"
TEST_LABELS_FILENAME = "emnist-balanced-test-labels-idx1-ubyte.gz"
MAPPING_FILENAME = "emnist-balanced-mapping.txt"

# Dataset Configuration: EMNIST Balanced Split
# The 'balanced' split contains 47 balanced classes:
# - 10 digits (0-9)
# - 37 letters (uppercase and lowercase merged for characters with similar shapes: C, I, J, K, L, M, O, P, S, U, V, W, X, Y, Z)
DATASET_NAME = "balanced"
NUM_CLASSES = 47
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28

# Exact Sample Counts for EMNIST Balanced Split
TRAIN_SAMPLES_PER_CLASS = 2400
TEST_SAMPLES_PER_CLASS = 400
TOTAL_SAMPLES_PER_CLASS = 2800   # 2,400 train + 400 test

TRAIN_SAMPLES_TOTAL = 112800     # 2,400 * 47
TEST_SAMPLES_TOTAL = 18800       # 400 * 47
TOTAL_SAMPLES = 131600           # 112,800 + 18,800

# Stratified Train/Validation Split Counts (90% Train, 10% Validation)
TRAIN_SUBSET_SAMPLES = 101520    # 112,800 * 0.9 (2,160 per class)
VAL_SUBSET_SAMPLES = 11280       # 112,800 * 0.1 (240 per class)
TRAIN_SUBSET_PER_CLASS = 2160
VAL_SUBSET_PER_CLASS = 240

# Sequence Configuration for LSTM
# An image of 28x28 pixels is treated as a sequence of rows:
# - Time Steps: 28 (sequential rows read from top to bottom)
# - Features per Time Step: 28 (pixel intensity values across each row)
TIME_STEPS = 28
INPUT_FEATURES = 28

# Baseline LSTM Model Architecture Configuration
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.2
DENSE_UNITS = 64

# Training Configuration
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.1
PATIENCE = 3

# Saved Artifact Paths
MODEL_SAVE_PATH = MODELS_DIR / "lstm_handwriting_model.keras"
HISTORY_SAVE_PATH = METRICS_DIR / "training_history.json"
