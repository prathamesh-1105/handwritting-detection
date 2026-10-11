# Project Progress Tracker: Design and Implement an LSTM Model for Handwriting Recognition

**Student / Academic Assignment:** Handwriting Recognition using Long Short-Term Memory (LSTM) Networks  
**Framework:** Python 3.11, TensorFlow / Keras, NumPy, Matplotlib, Scikit-Learn, Seaborn  
**Target Dataset:** EMNIST (Extended MNIST - Balanced Split: 47 classes)  
**Execution Environment:** Native Windows CPU (`venv` virtual environment; no GPU or paid APIs required)  

---

## 10-Step Project Execution Roadmap

| Step | Title | Status | Primary Output / Artifact |
|:---:|:---|:---:|:---|
| **1** | Project setup, requirements, dataset selection, and project structure | **COMPLETED & AUDITED** | Isolated `venv`, `requirements.txt`, `src/config.py`, verified directory tree |
| **2** | Dataset loading and exploration | **COMPLETED & AUDITED** | Modular loader in `src/dataset.py`, cached files in `data/`, exploratory visualization in `outputs/figures/` |
| **3** | Data preprocessing and sequential input preparation for LSTM | **COMPLETED & AUDITED** | Preprocessing module in `src/preprocess.py`, normalized tensors `(N, 28, 28)`, gallery in `outputs/figures/` |
| **4** | LSTM model architecture design and implementation | **COMPLETED & AUDITED** | Model definition in `src/model.py`, genuine Keras `LSTM`, verified summary (91,695 params) |
| **5** | Model training and training-history tracking | **COMPLETED & AUDITED** | Training module in `src/train.py`, checkpoint `models/lstm_handwriting_model.keras`, history JSON |
| **6** | Validation, hyperparameter review, and overfitting checks | **COMPLETED & AUDITED** | Validation analysis script `src/validation_analysis.py`, curves in `outputs/figures/training_history_curves.png` |
| **7** | Evaluation on unseen test data using appropriate metrics | **COMPLETED & AUDITED** | Evaluation module `src/evaluate.py`, artifacts `outputs/metrics/test_evaluation.json`, `classification_report.csv` |
| **8** | Handwriting prediction demonstrations on unseen samples | **COMPLETED & AUDITED** | Prediction engine `src/predict.py`, visual demo `src/demo_predictions.py`, gallery `outputs/figures/handwriting_prediction_demo.png` |
| **9** | Visualization, result analysis, and error inspection | **COMPLETED & AUDITED** | Confusion matrix heatmap, misclassified grid, top confusions, and class performance summaries |
| **10**| Final cleanup, reproducibility checks, README, practical report, and viva preparation | **COMPLETED & AUDITED** | Comprehensive `README.md`, lab report `PROJECT_REPORT.md`, viva guide `VIVA_QUESTIONS.md`, all verified |

---

## Step-by-Step Detailed Status

### Step 1: Project Setup, Requirements, Dataset Selection, and Project Structure
- **Objective:** Establish an isolated Python environment, define dependency requirements, select the appropriate handwriting dataset and recognition task, create modular folder hierarchy, and verify toolchains with reproducible tests.
- **Status:** **COMPLETED & AUDITED**
- **Verified Directory Structure:**
  ```
  c:\Users\Prathamesh\Desktop\dl hand\
  ├── data/                    # Output directory for raw/processed dataset files
  ├── models/                  # Output directory for saved model weights (.keras)
  ├── outputs/
  │   ├── figures/             # Output directory for plots and visual demo images
  │   └── metrics/             # Output directory for evaluation logs and matrices
  ├── src/
  │   ├── __init__.py          # Package marker
  │   ├── config.py            # Central project parameters and verified paths
  │   └── dataset.py           # Modular dataset loader & exploration
  ├── venv/                    # Isolated Python 3.11.9 virtual environment
  ├── requirements.txt         # Verified, minimal dependency specification
  └── PROJECT_PROGRESS.md      # Comprehensive step-by-step progress and audit tracker
  ```
- **Technical Decisions & Justifications:**
  - *Dataset Selection:* Selected **EMNIST (Balanced Split)** over standard MNIST (digits only) and IAM Handwriting (multi-GB, requires CTC decoding and GPU).
  - *Exact Sample Distribution:*
    - **Classes:** 47 classes (10 digits `0-9` + 37 balanced upper/lowercase letters).
    - **Training set:** 112,800 samples (exactly 2,400 samples per class).
    - **Test set:** 18,800 samples (exactly 400 samples per class).
    - **Total dataset:** 131,600 samples (exactly 2,800 samples per class).
  - *Sequential Input Formulation for LSTM:*
    - An LSTM requires sequential data `(batch_size, time_steps, features_per_step)`.
    - Each 28×28 grayscale character image will be treated as a sequence of **28 time steps** (rows read sequentially from top to bottom), each with a feature vector of **28 pixel values** ($X \in \mathbb{R}^{28 \times 28}$).
  - *Dependency Audit & EMNIST Package Decision:*
    - The PyPI third-party package `emnist` was tested and discovered to be broken: it attempts to download from an old Google Drive link returning an HTML virus-scan page (105 KB), resulting in `zipfile.BadZipFile`.
    - Removed `emnist` from `requirements.txt` and uninstalled it from `venv`. The dataset loader uses Python standard `requests`, `zlib`, `struct`, and `gzip`.
- **Verified Test Results:**
  - `python --version` $\rightarrow$ `Python 3.11.9`
  - Virtual Environment: Verified at `.\venv\Scripts\python.exe`
  - TensorFlow Version: `2.21.0` (Runs in CPU mode natively on Windows; `GPUs Available: []`)
  - Keras Version: `3.15.1` (LSTM layer verified: `keras.src.layers.rnn.lstm.LSTM`)
  - NumPy Version: `2.4.6`
  - Matplotlib Version: `3.11.2`
  - Scikit-Learn Version: `1.9.1`
  - Seaborn Version: `0.13.2`
  - Requests Version: `2.34.2`

---

### Step 2: Dataset Loading and Exploration
- **Objective:** Locate a reliable, official public source for EMNIST Balanced, implement a modular loader in `src/dataset.py`, download and cache binary files inside `data/`, parse IDX formats into NumPy arrays, verify dimensions, data types, pixel ranges, exact class balance, official mapping, and investigate image orientation.
- **Status:** **COMPLETED & AUDITED**
- **Implementation & Source Details:**
  - *Source URL:* Official NIST biometrics repository archive: `https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip`
  - *Download Strategy:* Implemented an optimized HTTP Range stream (bytes 0 to 33,028,000) that fetches only the 5 Balanced split files (~33 MB) in seconds, without downloading the entire 561 MB multi-split archive. Built-in automatic fallback to full zip download if Range is unavailable or if fewer than 5 files are parsed from the stream, with automated temporary file cleanup.
  - *Repeatable Caching:* Checks `data/` before connecting to network; skips downloading if valid files exist.
  - *Cached Files in [`data/`](file:///c:/Users/Prathamesh/Desktop/dl%20hand/data):*
    - `emnist-balanced-mapping.txt` (328 bytes)
    - `emnist-balanced-test-images-idx3-ubyte.gz` (4,714,707 bytes)
    - `emnist-balanced-test-labels-idx1-ubyte.gz` (13,504 bytes)
    - `emnist-balanced-train-images-idx3-ubyte.gz` (28,207,740 bytes)
    - `emnist-balanced-train-labels-idx1-ubyte.gz` (80,618 bytes)
- **Verified Dataset Statistics:**
  - **Training Images:** Shape `(112800, 28, 28)` | Dtype: `uint8` | Pixel Range: `[0, 255]`
  - **Training Labels:** Shape `(112800,)` | Dtype: `uint8` | Label Range: `[0, 46]`
  - **Test Images:** Shape `(18800, 28, 28)` | Dtype: `uint8` | Pixel Range: `[0, 255]`
  - **Test Labels:** Shape `(18800,)` | Dtype: `uint8` | Label Range: `[0, 46]`
  - **Class Distribution:** Exactly 2,400 training samples and 400 test samples across all 47 classes (`np.all(counts == 2400)` $\rightarrow$ `True`; `np.all(counts == 400)` $\rightarrow$ `True`).
- **Verified Official Class Mapping:**
  - 47 total classes:
    - Indices 0–9: Digits `'0'` through `'9'`
    - Indices 10–35: Uppercase letters `'A'` through `'Z'`
    - Indices 36–46: Lowercase letters `'a'`, `'b'`, `'d'`, `'e'`, `'f'`, `'g'`, `'h'`, `'n'`, `'q'`, `'r'`, `'t'` (11 lowercase letters distinct from uppercase; remaining 15 characters are merged into uppercase to avoid ambiguous dual ground-truth).
- **Image Orientation Investigation (Objective 10):**
  - **Finding:** Raw EMNIST IDX images are stored in Fortran column-major order. When read directly into row-major NumPy arrays, characters appear rotated 90° counter-clockwise and horizontally mirrored (lying sideways).
  - **Verified Solution:** Array transposition (`img.T` or `np.transpose(img, (0, 2, 1))`) restores natural upright handwriting.
  - **Protocol:** Images remain in raw format during Step 2; definitive batch transposition will be applied during Step 3 preprocessing.
- **Visual Artifact Generated:**
  - Saved: [`outputs/figures/emnist_samples_exploration.png`](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/emnist_samples_exploration.png) (178,141 bytes), showing side-by-side Raw vs. Transposed comparisons across 8 representative character classes.

---

### Step 3: Data Preprocessing and Preparation of Sequential Inputs
- **Objective:** Apply permanent batch transposition to orient all images upright, normalize pixel values from `uint8` [0, 255] to `float32` [0.0, 1.0], structure 3D sequential tensors `(N, time_steps=28, input_features=28)` suitable for LSTM, construct a reproducible stratified 90/10 train-validation split (101,520 train, 11,280 val), encode labels, run an exhaustive integrity test suite, and export a visualization of preprocessed samples.
- **Status:** **COMPLETED & AUDITED**
- **Implementation & Preprocessing Details:**
  - *Module Created:* [`src/preprocess.py`](file:///c:/Users/Prathamesh/Desktop/dl%20hand/src/preprocess.py)
  - *Orientation Correction:* Batch array transposition `np.transpose(images, (0, 2, 1))` applied across training and test splits, flipping the native Fortran column-major storage to upright row-major orientation.
  - *Pixel Intensity Normalization:* Cast to `np.float32` and scaled by `1.0 / 255.0`, yielding floating-point tensors bounded in $[0.0000, 1.0000]$.
  - *Sequential Tensor Formulation for LSTM:*
    - Each 28×28 image represents **28 sequential time steps** (rows read top-to-bottom), with each time step containing a feature vector of **28 pixel intensities** ($X \in \mathbb{R}^{28 \times 28}$).
    - Training subset tensor shape: `(101520, 28, 28)`, `float32` (~318 MB in RAM).
    - Validation subset tensor shape: `(11280, 28, 28)`, `float32` (~35 MB in RAM).
    - Held-out test set tensor shape: `(18800, 28, 28)`, `float32` (~59 MB in RAM).
  - *Stratified 90/10 Train-Validation Split:*
    - Derived using `sklearn.model_selection.train_test_split` with `random_state=42` and `stratify=y_train_full`.
    - Train subset: Exactly **101,520 samples** ($112,800 \times 0.90$).
    - Validation subset: Exactly **11,280 samples** ($112,800 \times 0.10$).
    - Preserved test set: Exactly **18,800 samples** held out untouched for Step 7 evaluation.
    - Class balance: Exactly **2,160 training samples** ($2,400 \times 0.9$) and **240 validation samples** ($2,400 \times 0.1$) across each and every one of the 47 classes.
  - *Label Encoding Strategy:*
    - **Default:** Sparse integer encoding (`int32`, values $0 \dots 46$), aligned with `sparse_categorical_crossentropy`. Saves memory on CPU (~112 KB vs. ~21 MB for one-hot matrices) and accelerates loss computation.
    - **Alternative Option:** Verified `one_hot=True` mode produces exact binary indicator matrices of shape `(N, 47)` with unit row sums.
  - *Memory Optimization:*
    - Memory usage is strictly controlled: vector operations on NumPy arrays avoid unnecessary intermediate copies, easily fitting within standard laptop RAM without GPU requirements.
- **Verified Programmatic Assertions (15/15 Passed):**
  - Array shapes: `(101520, 28, 28)`, `(11280, 28, 28)`, `(18800, 28, 28)` $\rightarrow$ **PASS**
  - Floating-point dtype: `float32` $\rightarrow$ **PASS**
  - Pixel value range: $[0.0000, 1.0000]$ across all splits $\rightarrow$ **PASS**
  - Target label bounds: $0 \le y \le 46$ across all splits $\rightarrow$ **PASS**
  - Non-finite check: Zero NaNs and zero Infs (`np.isfinite().all() == True`) $\rightarrow$ **PASS**
  - Class balance: Exactly 2,160 train, 240 val, 400 test samples per class $\rightarrow$ **PASS**
  - Data isolation: Disjoint index check ($I_{\text{train}} \cap I_{\text{val}} = \emptyset$) $\rightarrow$ **PASS**
  - Total index conservation: $|I_{\text{train}}| + |I_{\text{val}}| = 112,800$ $\rightarrow$ **PASS**
  - Split reproducibility: Seed 42 produces identical splits $\rightarrow$ **PASS**
- **Visual Artifact Generated:**
  - Saved: [`outputs/figures/preprocessed_samples_visualization.png`](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/preprocessed_samples_visualization.png) (146,501 bytes), exhibiting 16 upright, normalized character samples with exact shapes and pixel ranges.

---

### Step 4: LSTM Model Architecture Design and Implementation
- **Objective:** Construct a genuine Recurrent Neural Network using TensorFlow/Keras `LSTM` layers, configure explicit sequential input dimensions `(28, 28)`, implement a classification head with Softmax for 47 classes, compile with Adam and sparse categorical cross-entropy, inspect `model.summary()`, and run a CPU dummy forward pass without training the network.
- **Status:** **COMPLETED & AUDITED**
- **Implementation & Architecture Details:**
  - *Module Created:* [`src/model.py`](file:///c:/Users/Prathamesh/Desktop/dl%20hand/src/model.py)
  - *Input Layer:* `layers.Input(shape=(28, 28), name="handwriting_sequence_input")` representing 28 sequential row time steps $\times$ 28 horizontal pixel features.
  - *Recurrent Backbone:* Genuine `tf.keras.layers.LSTM(128, name="lstm_recurrent_layer")` processing the sequential row dynamics into a 128-dimensional hidden state ($80,384$ parameters).
  - *Regularization:* `layers.Dropout(0.2, name="dropout_regularization")` active during training to prevent co-adaptation of recurrent features.
  - *Dense Hidden Layer:* `layers.Dense(64, activation="relu", name="dense_feature_layer")` ($8,256$ parameters).
  - *Classification Output Layer:* `layers.Dense(47, activation="softmax", name="classification_output")` ($3,055$ parameters), yielding normalized class posterior probabilities $\sum_{k=0}^{46} P(y=k \mid X) = 1.0$.
  - *Compilation:* Optimizer `Adam(learning_rate=0.001)`, loss `'sparse_categorical_crossentropy'`, metrics `['accuracy']`.
  - *Parameter Count Breakdown:*
    - LSTM layer: $4 \times [(28 + 128) \times 128 + 128] = 80,384$
    - Dense feature layer: $128 \times 64 + 64 = 8,256$
    - Output layer: $64 \times 47 + 47 = 3,055$
    - Total parameters: **91,695 (all trainable, 358.18 KB)**.
  - *Architectural Justification:* Single 128-unit LSTM provides optimal CPU throughput (~25–35s/epoch on 101,520 samples) while capturing full sequence dynamics, minimizing memory pressure on standard local hardware.
- **Verified Programmatic Assertions:**
  - Genuine `tf.keras.layers.LSTM` confirmed in model layer graph $\rightarrow$ **PASS**
  - Input shape: `(None, 28, 28)` matches preprocessed tensors $\rightarrow$ **PASS**
  - Output shape: `(None, 47)` matches 47 EMNIST Balanced classes $\rightarrow$ **PASS**
  - Optimizer verified: `Adam` ($lr = 0.001$) $\rightarrow$ **PASS**
  - Loss verified: `'sparse_categorical_crossentropy'` (compatible with integer targets $[0, 46]$) $\rightarrow$ **PASS**
  - Parameter count: Exactly 91,695 trainable parameters verified $\rightarrow$ **PASS**
  - Dummy forward pass: Batch of shape `(4, 28, 28)` yielded output `(4, 47)` with valid probability distributions (row sums $[1.0000, 1.0000, 1.0000, 1.0000]$) $\rightarrow$ **PASS**
  - Boundary: Zero model training has occurred $\rightarrow$ **PASS**

---

### Step 5: Model Training and Training-History Tracking
- **Objective:** Train the compiled LSTM model on the 101,520 training images and validate on the 11,280 validation images across 15 epochs, tracking training history (loss, accuracy, val_loss, val_accuracy) with checkpoints.
- **Status:** **COMPLETED & AUDITED**
- **Verified Implementation & Artifacts:**
  - Modular training script implemented in `src/train.py` with reproducible seed (`42`), stratified data loading (`prepare_dataset(one_hot=False)`), and callbacks (`ModelCheckpoint`, `EarlyStopping`).
  - Model checkpoint saved: `models/lstm_handwriting_model.keras` (1,131,849 bytes). Reloaded and validated on CPU (input `(None, 28, 28)`, output `(None, 47)`, valid softmax distribution, row sums $\approx 1.0$).
  - Training history serialized: `outputs/metrics/training_history.json` (2,530 bytes). All metric arrays (`loss`, `accuracy`, `val_loss`, `val_accuracy`) strictly verified with length 15.
  - Training execution: All 15 planned epochs completed naturally (no early termination; EarlyStopping `patience=3` was not reached since val_loss improved through Epoch 14).
  - **Best Checkpoint (Epoch 14):**
    - `val_loss`: 0.370830 (minimum across entire run)
    - `val_accuracy`: 86.66% (0.866578)
    - `loss`: 0.304727
    - `accuracy`: 88.51% (0.885097)
  - **Final Epoch (Epoch 15):**
    - `val_loss`: 0.373850
    - `val_accuracy`: 86.91% (0.869060)
    - `loss`: 0.295444
    - `accuracy`: 88.90% (0.888997)
  - Best model weights restored via `restore_best_weights=True` to Epoch 14 parameters upon training completion.
  - **Audit Metric Reconciliation:** Ground-truth Epoch 1 in `training_history.json` and console logs recorded `loss: 1.435611`, `accuracy: 58.25%`, `val_loss: 0.718148` (checkpoint rounded: `0.71815`), `val_accuracy: 76.50%`. The earlier live training mention of `1.3402 / 62.43% / 78.47%` was an uncommitted/approximate transcription; raw ground truth from Keras `model.fit()` is permanently recorded.
  - Held-out test set (18,800 samples) strictly preserved and untouched for Step 7.

---

### Step 6: Validation, Hyperparameter Review, and Overfitting Checks
- **Objective:** Compare training vs. validation curves, analyze gap between train and validation accuracy, and review hyperparameter choices.
- **Status:** **COMPLETED & AUDITED**
- **Verified Implementation & Artifacts:**
  - Automated analysis and plotting pipeline implemented in `src/validation_analysis.py`.
  - Learning curves generated and saved: `outputs/figures/training_history_curves.png` (347,845 bytes). Features two subplots (Loss vs Epoch and Accuracy vs Epoch) with clear labels, gridlines, legends, and explicit dashed markers indicating Best Checkpoint (Epoch 14).
  - **Quantitative Overfitting & Gap Analysis:**
    - **Best Checkpoint (Epoch 14):** Training Accuracy = 88.5097%, Validation Accuracy = 86.6578% $\rightarrow$ Generalization Gap = **+1.8519%** (0.018519).
    - **Final Epoch (Epoch 15):** Training Accuracy = 88.8997%, Validation Accuracy = 86.9060% $\rightarrow$ Generalization Gap = **+1.9937%** (0.019937).
    - **Validation Loss Dynamics:** Validation loss decreased smoothly across 12 epochs and increased in only 2 epochs:
      - Epoch 10: $+0.000908$ increase ($0.383347 \rightarrow 0.384255$, minor stochastic batch noise; loss promptly resumed dropping in Epoch 11).
      - Epoch 15: $+0.003020$ increase ($0.370830 \rightarrow 0.373850$, onset of training saturation/plateauing).
  - **Overfitting Assessment:** No severe overfitting or underfitting present. The narrow generalization gap (<2.0%) confirms effective regularization via `Dropout(0.2)` and the Adam optimizer. The slight uptick at Epoch 15 illustrates why `ModelCheckpoint(save_best_only=True)` and `EarlyStopping(restore_best_weights=True)` were employed, as they safely retained the optimal weights from Epoch 14 (`val_loss = 0.370830`).
  - **Hyperparameter Review:** Evaluated baseline architecture (`LSTM(128) -> Dropout(0.2) -> Dense(64, relu) -> Dense(47, softmax)`, `Adam(0.001)`, `batch_size=128`, `patience=3`). Confirmed reasonable, stable, and computationally feasible for standard Windows CPU execution without GPU dependency.
  - Boundary: Held-out test set (18,800 samples) remains completely unaccessed and preserved exclusively for Step 7.

---

### Step 7: Evaluation on Unseen Test Data Using Appropriate Metrics
- **Objective:** Evaluate the trained LSTM model on the held-out test split, calculating overall test accuracy, test loss, precision, recall, and F1-score across all 47 classes.
- **Status:** **COMPLETED & AUDITED**
- **Verified Implementation & Artifacts:**
  - Modular evaluation pipeline implemented in `src/evaluate.py`.
  - Checkpoint loaded: `models/lstm_handwriting_model.keras` without retraining or parameter alteration.
  - Held-out test set evaluated: exactly 18,800 images (400 per class across all 47 classes).
  - Saved reproducible metrics: `outputs/metrics/test_evaluation.json` (9,723 bytes) and `outputs/metrics/classification_report.csv` (1,883 bytes).
  - **Key Computed Test Metrics:**
    - Test Loss: `0.3901` (`0.390104`)
    - Overall Test Accuracy: **86.53%** (`0.865319`)
    - Macro Precision: **86.92%** (`0.869225`)
    - Macro Recall: **86.53%** (`0.865319`)
    - Macro F1-Score: **86.44%** (`0.864424`)
    - Weighted Precision: **86.92%** (`0.869225`)
    - Weighted Recall: **86.53%** (`0.865319`)
    - Weighted F1-Score: **86.44%** (`0.864424`)
  - **Class-Level Observations:** Highest F1-scores achieved on distinctive characters like 'W' (97.74%), '3' (97.51%), 'A' (96.71%), '7' (96.40%), 'P' (96.21%). Expected orthographic handwriting confusions observed on visually ambiguous pairs ('1' vs 'I' vs 'L', '0' vs 'O', 'q' vs '9' vs 'g').
  - Evaluation Integrity: 18,800 predictions verified within valid class bounds $[0, 46]$. Zero test samples were used for model fitting or checkpoint selection.

---

### Step 8: Handwriting Prediction Demonstrations on Unseen Samples
- **Objective:** Build an inference module that takes unseen test images, feeds them row-by-row into the LSTM, outputs predicted class probabilities, and compares predictions against ground truth.
- **Status:** **COMPLETED & AUDITED**
- **Verified Implementation & Artifacts:**
  - Inference engine implemented in `src/predict.py`: features `HandwritingPredictor` class with strict shape checks `(28, 28)` or `(1, 28, 28)`, pixel validation $[0.0, 1.0]$, top-k candidate ranking sorted descending, and index-based test sample prediction.
  - Demonstration script implemented in `src/demo_predictions.py`: selects 16 reproducible test samples via fixed random seed (`42`), queries predictor, and saves a 4x4 visual gallery.
  - Visual gallery artifact: `outputs/figures/handwriting_prediction_demo.png` (442,230 bytes). Displays original handwriting images, ground-truth characters, predicted characters, confidence percentages, and color-coded status boxes (green for correct, red for mismatch).
  - **Demonstration Metrics & Verification:**
    - Evaluated 16 test samples across multiple classes ('7', 'J', 'Q', 'D', 'I', 'Y', 'C', '8', 'T', 'H', 'O', '9').
    - Demo subset accuracy: **15/16 (93.8%)** on seed 42.
    - Verified orthographic ambiguity: Sample #13104 (a single straight vertical stroke) had ground truth 'I' (class 18) and predicted '1' (class 1) with 49.4% confidence (top candidates: '1': 49.4%, 'L': 34.7%, 'I': 15.5%), validating genuine non-fabricated behavior.
    - All predictions verified within valid class bounds $[0, 46]$ with finite probabilities summing to 1.0.
  - Boundary: Zero model retraining or fine-tuning was performed. Checkpoint weights and Step 7 evaluation metrics remain completely untouched.

---

### Step 9: Visualization, Result Analysis, and Error Inspection
- **Objective:** Generate reproducible visual artifacts explaining LSTM classification performance across the 47 EMNIST Balanced classes, calculate the 47x47 confusion matrix, isolate the strongest/weakest classes by F1-score, compute exact pairwise confusion counts, and visualize actual misclassified test samples with candidate confidence breakdowns.
- **Status:** **COMPLETED & AUDITED**
- **Verified Implementation & Artifacts:**
  - Modular analysis module: `src/analyze_results.py` executing prediction generation, matrix calculation, top-confusion aggregation, ranking summaries, and figure creation.
  - 47x47 Confusion Matrix Heatmap: `outputs/figures/confusion_matrix.png` (272,735 bytes). Displays row-normalized true-class percentages with character ticks (`0-9`, `A-Z`, `a-z` subset), percentage colorbar, and labeled axes.
  - Misclassified Examples Gallery: `outputs/figures/misclassified_examples.png` (463,384 bytes). 4x4 reproducible grid of 16 actual misclassified test images showing true character, predicted character, model confidence, top-3 candidates, and original stroke orientation.
  - Raw Confusion Matrix Table: `outputs/metrics/confusion_matrix.csv` (4,817 bytes). Exact integer counts for all 47x47 class pairs.
  - Top Confusion Rankings: `outputs/metrics/top_confusions.json` (2,738 bytes). Contains top 10 misclassification pairs with count and true-class percentage.
  - Class Performance Summary: `outputs/metrics/class_performance_summary.json` (10,949 bytes). Contains strongest classes, weakest classes, and complete 47-class ranking sorted by F1-score.
- **Quantitative Audit Findings:**
  - **Top 5 Strongest Classes (Highest F1-Score):**
    1. `'W'` (Class 32): Precision = 98.23%, Recall = 97.25%, **F1 = 97.74%**, Support = 400
    2. `'3'` (Class 3): Precision = 97.03%, Recall = 98.00%, **F1 = 97.51%**, Support = 400
    3. `'A'` (Class 10): Precision = 97.95%, Recall = 95.50%, **F1 = 96.71%**, Support = 400
    4. `'7'` (Class 7): Precision = 95.80%, Recall = 97.00%, **F1 = 96.40%**, Support = 400
    5. `'P'` (Class 25): Precision = 97.19%, Recall = 95.25%, **F1 = 96.21%**, Support = 400
  - **Top 5 Weakest Classes (Lowest F1-Score):**
    1. `'L'` (Class 21): Precision = 67.79%, Recall = 45.25%, **F1 = 54.27%**, Support = 400
    2. `'f'` (Class 40): Precision = 66.67%, Recall = 51.00%, **F1 = 57.79%**, Support = 400
    3. `'q'` (Class 44): Precision = 60.77%, Recall = 59.25%, **F1 = 60.00%**, Support = 400
    4. `'1'` (Class 1): Precision = 49.09%, Recall = 80.50%, **F1 = 60.98%**, Support = 400
    5. `'I'` (Class 18): Precision = 73.49%, Recall = 54.75%, **F1 = 62.75%**, Support = 400
  - **Top 5 Most Frequent Actual Confusions:**
    1. True `'L'` $\rightarrow$ Pred `'1'`: 184 mistakes (46.00% of true 'L')
    2. True `'f'` $\rightarrow$ Pred `'F'`: 167 mistakes (41.75% of true 'f')
    3. True `'I'` $\rightarrow$ Pred `'1'`: 142 mistakes (35.50% of true 'I')
    4. True `'0'` $\rightarrow$ Pred `'O'`: 114 mistakes (28.50% of true '0')
    5. True `'O'` $\rightarrow$ Pred `'0'`: 111 mistakes (27.75% of true 'O')
- **Boundary Verification:** Model checkpoint `models/lstm_handwriting_model.keras` remained completely unchanged (zero retraining/fine-tuning). Official test split (18,800 images) and official class mapping were preserved with 100% integrity. Step 10 remains strictly PENDING.

---

### Step 10: Final Cleanup, Reproducibility Checks, README, Practical Report, and Viva Preparation
- **Objective:** Verify end-to-end reproducibility, clean unused temporary files, write a structured `README.md`, formulate a college lab report `PROJECT_REPORT.md`, compile viva questions in `VIVA_QUESTIONS.md`, and audit project integrity.
- **Status:** **COMPLETED & AUDITED**
- **Verified Implementation & Deliverables:**
  - Comprehensive Project Documentation: `README.md` (15 structured sections including problem statement, sequential sequence formulation, exact measured results, architecture parameter derivations, and setup commands).
  - Academic Practical Laboratory Report: `PROJECT_REPORT.md` (B.Tech Computer Engineering format complete with aim, theory of LSTM gates, equations, ASCII architecture schematics, algorithm pseudocode, confusion matrix analysis, and bibliography).
  - Viva and Examination Preparation Guide: `VIVA_QUESTIONS.md` (16 exam-ready Q&As explaining cell state mechanics, pixel normalization, vanishing gradients, Softmax, Macro vs. Weighted F1 equivalence, and specific character confusions with project metrics).
  - CLI Script Robustness: Added explicit `sys.path` initialization across modules (`src/dataset.py`, `src/preprocess.py`, `src/model.py`, `src/train.py`) and implemented `--check` dry-run mode in `src/train.py` allowing safe verification without retraining.
  - Reproducibility Checks: Verified dataset loading (`python -c "from src.dataset import load_emnist_balanced..."`), training sanity check (`python src/train.py --check`), prediction demonstration (`python src/demo_predictions.py`), and evaluation/analysis imports.
  - Checkpoint and Metric Integrity: Verified that `models/lstm_handwriting_model.keras` (1,131,849 bytes) remained strictly untouched. All 6 metric files parse cleanly and agree across all reports.
  - Complete 10-Step Roadmap: All 10 project milestones are successfully completed and verified.
