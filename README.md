# Design and Implementation of an LSTM Model for Handwriting Recognition

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.21.0-orange.svg)](https://tensorflow.org/)
[![Dataset EMNIST](https://img.shields.io/badge/Dataset-EMNIST%20Balanced%20(47%20Classes)-green.svg)](https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-86.53%25-brightgreen.svg)]()
[![Macro F1-Score](https://img.shields.io/badge/Macro%20F1--Score-86.44%25-brightgreen.svg)]()

An academic deep learning project demonstrating character-level handwriting classification using a genuine Long Short-Term Memory (LSTM) recurrent neural network trained on the official NIST EMNIST Balanced dataset.

---

## 1. Project Title & Objective

- **Title:** Design and Implementation of an LSTM Model for Handwriting Recognition
- **Academic Context:** B.Tech Computer Engineering / Deep Learning Practical Assignment
- **Primary Objective:** Build, train, evaluate, and interpret a CPU-compatible recurrent neural network using Long Short-Term Memory (LSTM) cells to accurately classify handwritten alphanumeric characters across 47 balanced classes without relying on Convolutional Neural Networks (CNNs).

---

## 2. Problem Statement

Handwritten character recognition is inherently challenging because human writing displays immense variability in stroke width, slant, curvature, loop size, and personal writing style. While 2D Convolutional Neural Networks (CNNs) are the traditional choice for static computer vision tasks, recurrent architectures like LSTMs provide a compelling sequential alternative.

The challenge is to treat each handwritten character image not as a static spatial grid, but as an ordered sequence of horizontal scanlines read top-to-bottom, evaluating whether an LSTM can capture the temporal dynamics and structural transitions across consecutive strokes to perform multi-class recognition across 47 alphanumeric categories.

---

## 3. Dataset Source, Structure & Distribution

The project uses the official **EMNIST Balanced** split published by the National Institute of Standards and Technology (NIST):

- **Remote Archive:** `https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip`
- **Total Classes:** Exactly **47 balanced classes**
  - **10 Digits:** `0, 1, 2, 3, 4, 5, 6, 7, 8, 9` (Class Indices: 0–9)
  - **26 Uppercase Letters:** `A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z` (Class Indices: 10–35)
  - **11 Lowercase Letters:** `a, b, d, e, f, g, h, n, q, r, t` (Class Indices: 36–46)  
    *(Note: Letters with identical uppercase and lowercase shapes such as C/c, O/o, S/s, U/u, V/v, W/w, X/x, Z/z are merged into single classes in the official balanced split).*
- **Dataset Partitioning:**
  - **Full Training Split:** 112,800 images (exactly 2,400 samples per class)
  - **Stratified Training Subset (90%):** 101,520 images (2,160 samples per class)
  - **Stratified Validation Subset (10%):** 11,280 images (240 samples per class)
  - **Held-out Official Test Split:** 18,800 images (exactly 400 samples per class)

---

## 4. Why LSTM for Image Classification?

Standard Feedforward Neural Networks (MLPs) treat image pixels as independent flat vectors, discarding geometric continuity. While CNNs apply 2D spatial convolutions, **Long Short-Term Memory (LSTM)** networks offer a sequence-based perspective:

1. **Temporal Modeling of Handwriting:** In human writing, characters are drawn sequentially over time. By scanning an image row-by-row, the LSTM simulates a writing sensor observing pen trajectories unfolding.
2. **Mitigating Vanishing Gradients:** Standard SimpleRNNs fail over 28 sequential steps due to exponential gradient decay. LSTMs maintain an additive **Cell State ($C_t$)** governed by input, forget, and output gating mechanisms, preserving long-range dependencies from the character's top strokes to its base.
3. **Parametric Compactness:** The LSTM baseline achieves 86.53% test accuracy with only **91,695 total parameters**, requiring zero specialized GPU hardware.

---

## 5. Sequential Input Transformation: Converting $28 \times 28$ Images into Time Steps

Each character image has dimensions of $28 \times 28$ grayscale pixels. Rather than flattening this into a 784-dimensional vector, the image is structured as a 2D temporal sequence:

$$\text{Input Tensor Shape: } (N, 28, 28)$$

- **Time Steps ($T = 28$):** The vertical scanlines (rows) ordered sequentially from row $0$ (top of the character) to row $27$ (bottom of the character).
- **Features per Time Step ($D = 28$):** The 28 horizontal pixel intensities along each row.
- **Sequential Flow:** At time step $t=1$, the LSTM inspects the top stroke; at $t=14$, it processes central crossbars and loops; at $t=28$, it inspects terminal descenders and base flourishes.

```
Row 0  (t=1)  : [ 0.0, 0.0, ..., 0.0 ]  --> LSTM Step 1  --> Hidden State h_1
Row 1  (t=2)  : [ 0.0, 0.2, ..., 0.0 ]  --> LSTM Step 2  --> Hidden State h_2
...
Row 27 (t=28) : [ 0.0, 0.0, ..., 0.0 ]  --> LSTM Step 28 --> Final Hidden State h_28 --> Dense Layers
```

---

## 6. Model Architecture & Parameter Count

The model is built using TensorFlow/Keras and defined in [src/model.py](file:///c:/Users/Prathamesh/Desktop/dl%20hand/src/model.py):

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 128)               80,384    
 dropout (Dropout)           (None, 128)               0         
 dense (Dense)               (None, 64)                8,256     
 dense_1 (Dense)             (None, 47)                3,055     
=================================================================
Total params: 91,695 (358.18 KB)
Trainable params: 91,695 (358.18 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

### Mathematical Parameter Derivation:
1. **LSTM Layer (128 units, input dimension 28):**
   $$4 \times \left[(d + h) \times h + h\right] = 4 \times \left[(28 + 128) \times 128 + 128\right] = 4 \times [19,968 + 128] = \mathbf{80,384}$$
2. **Dropout Layer (rate = 0.2):** $\mathbf{0}$ parameters.
3. **Dense Hidden Layer (64 units, ReLU):**
   $$(128 \times 64) + 64 = 8,192 + 64 = \mathbf{8,256}$$
4. **Softmax Output Layer (47 units, Softmax):**
   $$(64 \times 47) + 47 = 3,008 + 47 = \mathbf{3,055}$$
5. **Total Trainable Parameters:** $80,384 + 8,256 + 3,055 = \mathbf{91,695}$.

---

## 7. Preprocessing Pipeline

The preprocessing pipeline in [src/preprocess.py](file:///c:/Users/Prathamesh/Desktop/dl%20hand/src/preprocess.py) enforces strict data isolation and mathematical integrity:

1. **Orientation Correction (Fortran Transposition Fix):**
   Raw EMNIST IDX binary data is stored in column-major order. Loading into NumPy yields transposed images. Applying batch transposition `np.transpose(images, (0, 2, 1))` restores upright characters.
2. **Pixel Intensity Normalization:**
   Raw integer pixels in $[0, 255]$ are converted to `float32` in $[0.0, 1.0]$ via $x_{\text{norm}} = \frac{x}{255.0}$.
3. **Stratified Split:**
   The 112,800 training images are divided into 101,520 training and 11,280 validation samples using `train_test_split(stratify=y, test_size=0.1, random_state=42)`.
4. **Sparse Integer Encoding:**
   Labels are formatted as integer class IDs $[0, 46]$ paired with `SparseCategoricalCrossentropy` for memory optimization.

---

## 8. Training Configuration & Callbacks

- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$)
- **Learning Rate:** $0.001$
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Batch Size:** $128$
- **Maximum Epochs:** $15$
- **Callbacks:**
  - `ModelCheckpoint`: Monitors `val_loss`, saves the best model checkpoint to [models/lstm_handwriting_model.keras](file:///c:/Users/Prathamesh/Desktop/dl%20hand/models/lstm_handwriting_model.keras).
  - `EarlyStopping`: Monitors `val_loss` with `patience=3` and `restore_best_weights=True`.

---

## 9. Actual Measured Results: Validation vs. Test

The results strictly distinguish validation metrics from held-out test metrics:

| Metric Category | Metric | Measured Value | Reference Artifact |
|:---|:---|:---:|:---|
| **Validation (Step 5 & 6)** | Best Epoch | **Epoch 12 / 15** | [training_history.json](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/training_history.json) |
| Validation | Minimum Validation Loss | **0.4359** | [training_history.json](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/training_history.json) |
| Validation | Best Validation Accuracy | **86.45%** | [training_history.json](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/training_history.json) |
| Validation | Final Epoch 15 Loss / Val Loss | 0.2878 / 0.4485 | [training_history.json](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/training_history.json) |
| Validation | Generalization Gap | **5.58%** (Controlled) | [training_history_curves.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/training_history_curves.png) |
| **Test (Step 7 Evaluation)** | Total Test Samples | **18,800** (400 per class) | [test_evaluation.json](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/test_evaluation.json) |
| Test | **Test Accuracy** | **86.53%** (16,268 / 18,800) | [test_evaluation.json](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/test_evaluation.json) |
| Test | **Test Loss** | **0.4402** | [test_evaluation.json](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/test_evaluation.json) |
| Test | **Macro Precision** | **86.72%** | [classification_report.csv](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/classification_report.csv) |
| Test | **Macro Recall** | **86.53%** | [classification_report.csv](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/classification_report.csv) |
| Test | **Macro F1-Score** | **86.44%** | [classification_report.csv](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/classification_report.csv) |
| Test | **Weighted F1-Score** | **86.44%** | [classification_report.csv](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/classification_report.csv) |

---

## 10. Confusion Matrix Findings & Error Analysis

The 47×47 confusion matrix in [outputs/figures/confusion_matrix.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/confusion_matrix.png) and error rankings in [outputs/metrics/top_confusions.json](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/top_confusions.json) reveal key classification trends:

### Top 5 Strongest Classes (Highest F1)
1. **`'W'` (Class 32):** F1 = **97.74%** (Precision = 98.23%, Recall = 97.25%) — Distinct 4-stroke diagonal alternation.
2. **`'3'` (Class 3):** F1 = **97.51%** (Precision = 97.03%, Recall = 98.00%) — Prominent twin rightward lobes.
3. **`'A'` (Class 10):** F1 = **96.71%** (Precision = 97.95%, Recall = 95.50%) — Strong triangular vertex and central bar.
4. **`'7'` (Class 7):** F1 = **96.40%** (Precision = 95.80%, Recall = 97.00%) — Unambiguous top horizontal and diagonal stroke.
5. **`'P'` (Class 25):** F1 = **96.21%** (Precision = 97.19%, Recall = 95.25%) — Left vertical stem with closed upper-right loop.

### Top 5 Weakest Classes (Lowest F1)
1. **`'L'` (Class 21):** F1 = **54.27%** (Precision = 67.79%, Recall = 45.25%) — 184 samples (46.0%) misclassified as `'1'`.
2. **`'f'` (Class 40):** F1 = **57.79%** (Precision = 66.67%, Recall = 51.00%) — 167 samples (41.8%) misclassified as `'F'`.
3. **`'q'` (Class 44):** F1 = **60.00%** (Precision = 60.77%, Recall = 59.25%) — 88 samples (22.0%) misclassified as `'9'`.
4. **`'1'` (Class 1):** F1 = **60.98%** (Precision = 49.09%, Recall = 80.50%) — Acts as a false-positive sink for `'L'` and `'I'`.
5. **`'I'` (Class 18):** F1 = **62.75%** (Precision = 73.49%, Recall = 54.75%) — 142 samples (35.5%) misclassified as `'1'`.

---

## 11. Environment Setup & Windows Instructions

This project runs on standard CPU hardware using Python 3.11.

### 1. Prerequisites
- Windows 10/11 64-bit
- Python 3.11.x installed

### 2. Virtual Environment Setup
Open PowerShell in the project directory:
```powershell
# Navigate to project root
cd "C:\Users\Prathamesh\Desktop\dl hand"

# Create virtual environment (if not already created)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

## 12. Reproduction & Execution Commands

All scripts can be executed safely using the activated virtual environment:

### Verify Dataset & Pipeline
```powershell
# Verify dataset cache and orientation exploration
python -m src.dataset
```

### Model Training Sanity Check (Zero Retraining)
```powershell
# Perform model architecture and path sanity check without training
python src/train.py --check
```

### Run Model Evaluation on Test Data
```powershell
# Evaluates checkpoint against all 18,800 test samples
python -m src.evaluate
```

### Run Prediction Demonstration
```powershell
# Generates a 4x4 visual gallery on 16 test samples
python -m src.demo_predictions
```

### Run Confusion Matrix & Error Analysis
```powershell
# Generates confusion matrix heatmap and misclassification analysis
python -m src.analyze_results
```

---

## 13. Project Directory Structure

```
c:\Users\Prathamesh\Desktop\dl hand\
├── data/                                      # Cached EMNIST Balanced binary files
│   ├── emnist-balanced-mapping.txt            # ASCII mapping (47 classes)
│   ├── emnist-balanced-train-images-idx3-ubyte.gz
│   ├── emnist-balanced-train-labels-idx1-ubyte.gz
│   ├── emnist-balanced-test-images-idx3-ubyte.gz
│   └── emnist-balanced-test-labels-idx1-ubyte.gz
├── models/                                    # Serialized model checkpoint
│   └── lstm_handwriting_model.keras          # Optimal Keras model (1.13 MB)
├── outputs/                                   # Evaluation metrics and visualizations
│   ├── figures/                               # Publication-quality visual plots
│   │   ├── emnist_samples_exploration.png     # Step 2: Raw vs. transposed orientation
│   │   ├── preprocessed_samples_visualization.png # Step 3: Verified preprocessed images
│   │   ├── training_history_curves.png        # Step 6: Loss and accuracy curves
│   │   ├── handwriting_prediction_demo.png    # Step 8: 16-sample prediction demo
│   │   ├── confusion_matrix.png               # Step 9: 47x47 normalized heatmap
│   │   └── misclassified_examples.png         # Step 9: 4x4 error gallery with top-3
│   └── metrics/                               # Numerical evaluation data
│       ├── training_history.json              # 15 epochs training & validation logs
│       ├── test_evaluation.json               # Test loss, accuracy, and macro metrics
│       ├── classification_report.csv          # Per-class precision, recall, F1, support
│       ├── confusion_matrix.csv               # 47x47 integer confusion matrix
│       ├── top_confusions.json                # Ranked pairwise error statistics
│       └── class_performance_summary.json     # Complete 47-class ranking
├── src/                                       # Modular Python source code
│   ├── __init__.py
│   ├── config.py                              # Central paths, constants & hyperparameters
│   ├── dataset.py                             # EMNIST downloading, parsing & caching
│   ├── preprocess.py                          # Transposition, normalization & splitting
│   ├── model.py                               # LSTM model definition & compilation
│   ├── train.py                               # Training loop, callbacks & checkpointing
│   ├── validation_analysis.py                 # Overfitting and convergence analysis
│   ├── evaluate.py                            # Official test set metric computation
│   ├── predict.py                             # Inference class with top-k candidates
│   ├── demo_predictions.py                   # Gallery generator for sample inference
│   └── analyze_results.py                     # Confusion matrix & error inspection
├── PROJECT_PROGRESS.md                        # Formal 10-step audit and progress tracker
├── PROJECT_REPORT.md                          # Comprehensive academic laboratory report
├── VIVA_QUESTIONS.md                          # Examination and viva preparation guide
├── requirements.txt                           # Pinned Python package dependencies
└── README.md                                  # Complete project documentation
```

---

## 14. Limitations & Future Improvements

1. **Scan Direction Sensitivity:** The single-directional LSTM reads row-by-row from top to bottom. It cannot observe vertical column transitions simultaneously.
   - *Improvement:* Implement a **Bidirectional LSTM (BiLSTM)** or a 2D-Grid LSTM scanning both horizontal and vertical directions.
2. **Scale & Size Invariance Ambiguity:** Because all characters are centered and scaled into 28×28 bounding boxes, uppercase `'F'` and lowercase `'f'`, or numeral `'0'` and letter `'O'`, lose aspect-ratio cues.
   - *Improvement:* Incorporate character aspect-ratio and bounding-box stroke thickness as auxiliary tabular input features.
3. **Single-Stroke Vertical Degeneracy:** Straight strokes (`'1'`, `'I'`, `'L'`) are geometrically degenerate without linguistic context.
   - *Improvement:* Pair the character recognizer with an N-gram language model or CTC beam search decoder for full word transcription.

---

## 15. Key Generated Artifacts Reference

- **Model Checkpoint:** [models/lstm_handwriting_model.keras](file:///c:/Users/Prathamesh/Desktop/dl%20hand/models/lstm_handwriting_model.keras)
- **Confusion Matrix Figure:** [outputs/figures/confusion_matrix.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/confusion_matrix.png)
- **Misclassified Error Gallery:** [outputs/figures/misclassified_examples.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/misclassified_examples.png)
- **Prediction Demo Gallery:** [outputs/figures/handwriting_prediction_demo.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/handwriting_prediction_demo.png)
- **Training Curves Plot:** [outputs/figures/training_history_curves.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/training_history_curves.png)
- **Test Metrics JSON:** [outputs/metrics/test_evaluation.json](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/test_evaluation.json)
- **Classification Report CSV:** [outputs/metrics/classification_report.csv](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/classification_report.csv)
- **Top Confusions JSON:** [outputs/metrics/top_confusions.json](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/metrics/top_confusions.json)
