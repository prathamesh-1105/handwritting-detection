# Academic Laboratory Practical Report

**Course / Subject:** Deep Learning Laboratory (B.Tech Computer Engineering)  
**Title of Experiment:** Design and Implementation of an LSTM Model for Handwriting Recognition  
**Student Name / Roll No:** Undergraduate Engineering Student  
**Academic Environment:** Native Windows CPU (`venv` environment; Python 3.11, TensorFlow 2.x)  
**Target Dataset:** NIST EMNIST Balanced Split (47 Classes)  

---

## 1. Aim

To design, implement, train, evaluate, and analyze a genuine Long Short-Term Memory (LSTM) Recurrent Neural Network for optical handwritten character recognition on the EMNIST Balanced dataset, processing $28 \times 28$ grayscale images as sequential temporal inputs without relying on Convolutional Neural Networks (CNNs).

---

## 2. Problem Statement

Handwritten character recognition is a core challenge in computer vision and optical character recognition (OCR) systems due to substantial inter-writer variance in slant, stroke pressure, curvature, and proportions. Traditional multi-layer perceptrons (MLPs) ignore the structural spatial correlation between adjacent pixels, while 2D Convolutional Neural Networks (CNNs) require spatial kernels. 

This experiment explores the sequence modeling paradigm: framing character recognition as an ordered sequence of horizontal scanlines. By passing horizontal rows successively into an LSTM cell, the network learns the sequential transitions of pen strokes from top to bottom. The practical task is to build a high-performing, CPU-compatible LSTM classifier for 47 distinct alphanumeric categories (digits, uppercase letters, and lowercase letters) and rigorously assess its performance, error modes, and orthographic confusion patterns.

---

## 3. Objectives

1. **Dataset Pipeline:** Download, cache, and parse the binary IDX files for the official EMNIST Balanced dataset (112,800 train and 18,800 test samples).
2. **Preprocessing & Sequence Formulation:** Diagnose and correct the Fortran column-major orientation quirk (batch transposition), normalize pixel values from $[0, 255]$ to $[0.0, 1.0]$, and formulate the 3D tensor shape $(N, 28, 28)$ representing 28 time steps with 28 features each.
3. **Architecture Design:** Construct a genuine Keras LSTM model featuring 128 LSTM units, Dropout regularization, a 64-unit Dense feature projection layer, and a 47-class Softmax classification head (91,695 total parameters).
4. **Model Training & Regularization:** Train the model using Adam optimization and Sparse Categorical Cross-Entropy across 15 epochs, logging loss and accuracy curves while checkpointing the best model based on validation loss.
5. **Comprehensive Evaluation:** Evaluate the saved best model exclusively on the held-out 18,800 test samples, reporting overall accuracy, loss, macro/weighted precision, recall, and F1-score.
6. **Error Analysis & Interpretation:** Construct a 47×47 confusion matrix, isolate the top pairwise confusions, identify the strongest and weakest classes, and visualize representative misclassifications.

---

## 4. Theory of LSTM, Gating Mechanisms & Sequential Vision

### 4.1 The Long Short-Term Memory (LSTM) Cell
Standard Recurrent Neural Networks (RNNs) suffer from the **vanishing gradient problem**: backpropagating gradients over repeated matrix multiplications causes gradients to exponentially decay, preventing the network from retaining information across long time horizons.

The LSTM architecture, introduced by Hochreiter and Schmidhuber (1997), overcomes this limitation through a dedicated linear memory highway called the **Cell State ($C_t$)**, regulated by three continuous, differentiable non-linear gating mechanisms:

1. **Forget Gate ($f_t$):** Determines what proportion of past memory to discard from the previous cell state $C_{t-1}$:
   $$f_t = \sigma\left(W_f \cdot [h_{t-1}, x_t] + b_f\right)$$
   where $\sigma(z) = \frac{1}{1 + e^{-z}}$ outputs values in $[0, 1]$.

2. **Input Gate ($i_t$) and Candidate Cell State ($\tilde{C}_t$):** Controls what new information from the current input $x_t$ and previous hidden state $h_{t-1}$ to store:
   $$i_t = \sigma\left(W_i \cdot [h_{t-1}, x_t] + b_i\right)$$
   $$\tilde{C}_t = \tanh\left(W_c \cdot [h_{t-1}, x_t] + b_c\right)$$

3. **Cell State Update ($C_t$):** The linear update equation that avoids gradient vanishing:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
   *(where $\odot$ denotes Hadamard element-wise product).*

4. **Output Gate ($o_t$) and Hidden State ($h_t$):** Determines the filtered output to pass to the next time step and higher network layers:
   $$o_t = \sigma\left(W_o \cdot [h_{t-1}, x_t] + b_o\right)$$
   $$h_t = o_t \odot \tanh(C_t)$$

```
                 +-------------------------------------------------+
                 |                    Cell State                   |
        C_{t-1} -+------------------[ * ]----------------(+)-> C_t |
                 |                    ^                   ^        |
                 |                    | f_t               |        |
                 |                    |         [ * ]-----+        |
                 |                    |          ^                 |
                 |                    |   i_t    | \tilde{C}_t     |
                 |                    |    +-----+-----+           |
                 |                   [s]  [s]        [tanh]        |
                 |                    ^    ^           ^           |
        h_{t-1} -+----+---------------+----+-----------+--->       |
                 |    |                                            |
                 |    +---------------------[s] (o_t)              |
                 |                           |                     |
                 |                           v                     |
                 |                   [tanh]-[ * ]----------> h_t   |
        x_t ----+----------------------------+                     |
                 +-------------------------------------------------+
```

### 4.2 Sequential Input Formulation from $28 \times 28$ Images
In this project, each 2D image $I \in \mathbb{R}^{28 \times 28}$ is interpreted as a temporal sequence:
- **Sequence Length ($T = 28$):** The number of rows, traversed sequentially from top ($t=1$) to bottom ($t=28$).
- **Feature Dimension ($D = 28$):** The 28 horizontal pixel values along row $t$.
- **Temporal Intuition:** As $t$ increments, the LSTM ingests progressive slices of the character. For example, when reading the letter `'A'`, the LSTM registers a single narrow stroke peak at $t=3$, split dual strokes at $t=10$, a solid horizontal crossbar bridging both strokes at $t=14$, and two widely spaced legs near $t=26$.

---

## 5. Dataset Description: EMNIST Balanced

The Extended MNIST (EMNIST) dataset is derived from NIST Special Database 19. The **Balanced split** is specifically curated to prevent class frequency bias across alphanumeric handwriting:

- **Total Classes ($K$):** 47 balanced classes.
- **Class Breakdown:**
  - 10 Digits: `0` to `9` (Classes 0–9).
  - 26 Uppercase Letters: `A` to `Z` (Classes 10–35).
  - 11 Distinct Lowercase Letters: `a, b, d, e, f, g, h, n, q, r, t` (Classes 36–46).
  - *Merging Justification:* Characters whose uppercase and lowercase forms have virtually identical geometry (e.g., `C/c`, `O/o`, `S/s`, `W/w`, `X/x`) are merged into the uppercase class to prevent ambiguous ground-truth labels.
- **Sample Distribution:**
  - Exactly **2,400 training samples** per class $\rightarrow$ **112,800 total training samples**.
  - Exactly **400 test samples** per class $\rightarrow$ **18,800 total held-out test samples**.
  - Grand total: **131,600 handwritten samples**.

---

## 6. Software and Hardware Requirements

### 6.1 Software Requirements
- **Operating System:** Windows 10/11 (64-bit)
- **Language:** Python 3.11.9
- **Core Libraries:**
  - `tensorflow==2.21.0` (Keras 3 deep learning framework)
  - `numpy==2.4.6` (Numerical linear algebra and array manipulation)
  - `scikit-learn==1.9.1` (Stratified splitting, classification report, confusion matrix)
  - `matplotlib==3.11.2` (Figure plotting and visual galleries)
  - `seaborn==0.13.2` (Heatmap visualization)
  - `requests==2.34.2` (HTTP archive retrieval)

### 6.2 Hardware Platform
- **Processor:** Standard Multi-core x86_64 CPU (Intel Core i5/i7/Ryzen)
- **RAM:** 8 GB or higher (peak memory usage during training was ~1.8 GB)
- **GPU Requirement:** **None required.** The architecture was explicitly engineered to train and evaluate within reasonable execution time on native CPU.

---

## 7. System Architecture and Pipeline Workflow

The experimental pipeline follows a strict modular workflow:

```
+-----------------------------------------------------------------------------------+
| 1. INGESTION (src/dataset.py)                                                    |
|    - Download official EMNIST gzip archives from NIST                             |
|    - Parse IDX3 (images) and IDX1 (labels) headers into NumPy uint8 arrays        |
|    - Validate class balance: 2,400 train / 400 test per class                     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 2. PREPROCESSING & SEQUENCING (src/preprocess.py)                                 |
|    - Batch Transposition: Correct Fortran column-major orientation                |
|    - Normalization: Scale uint8 [0, 255] to float32 [0.0, 1.0]                    |
|    - Stratified Partition: 101,520 train (90%) and 11,280 validation (10%)        |
|    - Sequential Reshape: Tensor shape (N, 28, 28) for LSTM ingestion              |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 3. MODEL COMPILATION & TRAINING (src/model.py & src/train.py)                     |
|    - Input: (None, 28, 28)                                                        |
|    - LSTM Layer: 128 units, unroll=False, tanh activation                         |
|    - Dropout: rate=0.2 (regularization)                                           |
|    - Dense Layer: 64 units, ReLU activation                                       |
|    - Output Layer: 47 units, Softmax activation                                   |
|    - Loss: Sparse Categorical Cross-Entropy, Optimizer: Adam (lr=0.001)           |
|    - Callbacks: ModelCheckpoint (val_loss min), EarlyStopping (patience=3)        |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 4. TEST EVALUATION & PREDICTION (src/evaluate.py & src/predict.py)                |
|    - Reload best checkpoint: models/lstm_handwriting_model.keras                  |
|    - Inference on 18,800 unseen test samples                                      |
|    - Compute Test Accuracy, Loss, Macro/Weighted Precision, Recall, F1            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 5. ERROR ANALYSIS & VISUALIZATION (src/analyze_results.py)                        |
|    - 47x47 Confusion Matrix Heatmap (outputs/figures/confusion_matrix.png)        |
|    - Top 10 Confusion Pairs extraction (outputs/metrics/top_confusions.json)     |
|    - Misclassified Test Samples Gallery (outputs/figures/misclassified_examples.png)
+-----------------------------------------------------------------------------------+
```

---

## 8. Algorithm and Pseudocode

```
ALGORITHM: LSTM_Handwriting_Recognition

INPUT:
    Raw training data: X_raw_train (112800, 28, 28), Y_raw_train (112800,)
    Raw test data    : X_raw_test  (18800, 28, 28),  Y_raw_test  (18800,)
    Class mapping    : M (47 classes)

OUTPUT:
    Trained Model Checkpoint: M_best
    Test Evaluation Metrics : Accuracy, Loss, Macro-F1
    Confusion Matrix Heatmap

PROCEDURE:
1. PREPROCESSING:
    FOR each dataset in [Train, Test]:
        X_transposed = Transpose(X_raw, axes=(0, 2, 1))    // Restore upright orientation
        X_norm = Float32(X_transposed) / 255.0             // Scale to [0.0, 1.0]
    
    // Stratified Train/Val split
    (X_train, Y_train), (X_val, Y_val) = StratifiedSplit(X_norm_train, Y_raw_train, val_ratio=0.10)

2. MODEL DEFINITION:
    Model = Sequential()
    Model.add(LSTM(units=128, input_shape=(28, 28), return_sequences=False))
    Model.add(Dropout(rate=0.2))
    Model.add(Dense(units=64, activation='relu'))
    Model.add(Dense(units=47, activation='softmax'))
    
    Model.compile(optimizer=Adam(lr=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

3. TRAINING:
    Initialize CheckpointCallback(monitor='val_loss', save_best_only=True)
    Initialize EarlyStoppingCallback(monitor='val_loss', patience=3, restore_best_weights=True)
    
    Fit Model on (X_train, Y_train) with validation_data=(X_val, Y_val),
        epochs=15, batch_size=128, callbacks=[CheckpointCallback, EarlyStoppingCallback]

4. EVALUATION:
    Load M_best from saved checkpoint
    Y_pred_probs = M_best.predict(X_norm_test)
    Y_pred = ArgMax(Y_pred_probs, axis=-1)
    
    Test_Accuracy = Mean(Y_pred == Y_raw_test)
    Compute Macro_Precision, Macro_Recall, Macro_F1
    Construct ConfusionMatrix(Y_raw_test, Y_pred)
    
    RETURN Results
```

---

## 9. Model Architecture and Parameter Derivation

| Layer # | Layer Name | Type | Output Dimension | Activation | Parameter Count |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | `lstm` | LSTM | `(None, 128)` | `tanh` (recurrent `sigmoid`) | 80,384 |
| 2 | `dropout` | Dropout ($p=0.2$) | `(None, 128)` | None | 0 |
| 3 | `dense` | Fully Connected | `(None, 64)` | `ReLU` | 8,256 |
| 4 | `dense_1` | Fully Connected (Head) | `(None, 47)` | `Softmax` | 3,055 |
| **Total** | | | | | **91,695** (358.18 KB) |

### Formal Mathematical Derivation:
1. **LSTM Parameters:**  
   An LSTM layer with input size $d=28$ and hidden size $h=128$ consists of 4 internal weight matrices for the gates ($f, i, c, o$). Each gate has input kernel weights $W_{x} \in \mathbb{R}^{28 \times 128}$, recurrent kernel weights $W_{h} \in \mathbb{R}^{128 \times 128}$, and bias $b \in \mathbb{R}^{128}$:
   $$\text{Params}_{\text{LSTM}} = 4 \times \left[(d \times h) + (h \times h) + h\right] = 4 \times \left[(28 \times 128) + (128 \times 128) + 128\right] = 4 \times [3,584 + 16,384 + 128] = \mathbf{80,384}$$
2. **Dense Hidden Parameters:**  
   $$\text{Params}_{\text{Dense}} = (128 \times 64) + 64 = 8,192 + 64 = \mathbf{8,256}$$
3. **Softmax Output Parameters:**  
   $$\text{Params}_{\text{Output}} = (64 \times 47) + 47 = 3,008 + 47 = \mathbf{3,055}$$
4. **Total Trainable Parameters:** $80,384 + 8,256 + 3,055 = \mathbf{91,695}$.

---

## 10. Training Methodology and Validation Dynamics

Training was executed for 15 epochs with mini-batch size 128 on the 101,520 training split, evaluated at each epoch against the 11,280 validation set:

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Learning Status |
|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | 1.3402 | 62.43% | 0.7182 | 78.47% | Rapid initial convergence |
| **2** | 0.6547 | 80.08% | 0.5701 | 82.26% | Steady progression |
| **3** | 0.5367 | 83.21% | 0.5135 | 83.74% | Cross-entropy reduction |
| **4** | 0.4727 | 84.86% | 0.4794 | 84.88% | Generalization stabilization |
| **5** | 0.4326 | 85.93% | 0.4746 | 85.34% | Gate weight tuning |
| **6** | 0.4042 | 86.66% | 0.4744 | 85.29% | Feature consolidation |
| **7** | 0.3807 | 87.35% | 0.4578 | 85.92% | Reaches 87% train accuracy |
| **8** | 0.3621 | 87.89% | 0.4563 | 85.90% | Loss flattening |
| **9** | 0.3475 | 88.22% | 0.4552 | 86.04% | Solidification |
| **10** | 0.3340 | 88.61% | 0.4468 | 86.13% | Val loss reaches 0.4468 |
| **11** | 0.3204 | 89.04% | 0.4566 | 85.93% | Slight fluctuation |
| **12** | **0.3093** | **89.37%** | **0.4359** | **86.45%** | **OPTIMAL CHECKPOINT (Min Val Loss)** |
| **13** | 0.3013 | 89.58% | 0.4444 | 86.29% | Overfitting threshold |
| **14** | 0.2929 | 89.84% | 0.4534 | 86.15% | Patience counter 2 |
| **15** | 0.2878 | 89.96% | 0.4485 | 86.13% | Final epoch completed |

- **Optimal Weights Restored:** `ModelCheckpoint` saved Epoch 12 weights (`val_loss = 0.4359`, `val_accuracy = 86.45%`).
- **Overfitting Diagnosis:** The generalization gap at Epoch 12 was $89.37\% - 86.45\% = \mathbf{2.92\%}$, and at Epoch 15 was $89.96\% - 86.13\% = \mathbf{3.83\%}$. This modest gap confirms that Dropout ($p=0.2$) effectively prevented severe memorization.

---

## 11. Final Test Evaluation Metrics

The optimal Epoch 12 model was evaluated on the completely unseen, official held-out test split of 18,800 images:

| Metric | Measured Value | Standard Interpretation |
|:---|:---:|:---|
| **Test Accuracy** | **86.53%** | 16,268 out of 18,800 test images correctly classified. |
| **Test Loss** | **0.4402** | Measured sparse categorical cross-entropy on unseen data. |
| **Macro-Averaged Precision** | **86.72%** | Average precision across all 47 classes with equal class weight. |
| **Macro-Averaged Recall** | **86.53%** | Average recall across all 47 classes with equal class weight. |
| **Macro-Averaged F1-Score** | **86.44%** | Harmonic mean of macro precision and macro recall. |
| **Weighted-Averaged F1-Score**| **86.44%** | Weighted mean by class support. Equal to Macro F1 because test set is balanced ($N_k = 400$). |

---

## 12. Confusion Matrix and Error Inspection

### 12.1 Class Extremes Analysis

```
TOP 5 STRONGEST CLASSES (Highest F1):
1. 'W' (Class 32) : F1 = 97.74% | Precision = 98.23% | Recall = 97.25%
2. '3' (Class 3)  : F1 = 97.51% | Precision = 97.03% | Recall = 98.00%
3. 'A' (Class 10) : F1 = 96.71% | Precision = 97.95% | Recall = 95.50%
4. '7' (Class 7)  : F1 = 96.40% | Precision = 95.80% | Recall = 97.00%
5. 'P' (Class 25) : F1 = 96.21% | Precision = 97.19% | Recall = 95.25%

TOP 5 WEAKEST CLASSES (Lowest F1):
1. 'L' (Class 21) : F1 = 54.27% | Precision = 67.79% | Recall = 45.25%
2. 'f' (Class 40) : F1 = 57.79% | Precision = 66.67% | Recall = 51.00%
3. 'q' (Class 44) : F1 = 60.00% | Precision = 60.77% | Recall = 59.25%
4. '1' (Class 1)  : F1 = 60.98% | Precision = 49.09% | Recall = 80.50%
5. 'I' (Class 18) : F1 = 62.75% | Precision = 73.49% | Recall = 54.75%
```

### 12.2 Primary Sources of Error
From the 47×47 confusion matrix, the top pairwise confusion counts reveal three major causes of misclassification:

1. **Vertical Stroke Degeneracy (`L` $\rightarrow$ `1` and `I` $\rightarrow$ `1`):**
   - 184 true `'L'` samples (46.0%) and 142 true `'I'` samples (35.5%) were classified as `'1'`.
   - *Cause:* When handwritten `'L'` lacks an elongated horizontal baseline, or `'I'` is written without top and bottom horizontal bars, all three reduce to a single vertical line. In a top-to-bottom row scan, all three generate identical temporal activations.
2. **Case Disambiguation Under Normalization (`f` $\rightarrow$ `F`):**
   - 167 true lowercase `'f'` samples (41.8%) were classified as uppercase `'F'`.
   - *Cause:* EMNIST normalizes all characters into a centered 28×28 bounding box. Without relative line height or ascender/descender reference baselines, lowercase `'f'` and uppercase `'F'` share the same crossbar topology.
3. **Continuous Loop Equivalence (`0` $\leftrightarrow$ `O`):**
   - 114 true `'0'` samples (28.5%) predicted `'O'`, and 111 true `'O'` samples (27.8%) predicted `'0'`.
   - *Cause:* Symmetrical bidirectional confusion due to topological equivalence between circular digit zero and letter O.

---

## 13. Referenced Figures and Visualizations

1. **Orientation Exploration:** [outputs/figures/emnist_samples_exploration.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/emnist_samples_exploration.png)  
   Demonstrates the raw column-major Fortran transposition quirk and verifies the upright visual restoration.
2. **Preprocessed Samples:** [outputs/figures/preprocessed_samples_visualization.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/preprocessed_samples_visualization.png)  
   Validates normalized $[0.0, 1.0]$ pixel intensity distributions.
3. **Learning Curves:** [outputs/figures/training_history_curves.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/training_history_curves.png)  
   Visualizes loss decay and accuracy progression across all 15 epochs.
4. **Prediction Demonstration:** [outputs/figures/handwriting_prediction_demo.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/handwriting_prediction_demo.png)  
   A 4×4 gallery showing predictions on 16 unseen test images with confidence percentages and top-3 rankings.
5. **Confusion Matrix Heatmap:** [outputs/figures/confusion_matrix.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/confusion_matrix.png)  
   High-resolution 47×47 percentage heatmap illustrating diagonal accuracy dominance and off-diagonal ambiguity clusters.
6. **Error Analysis Gallery:** [outputs/figures/misclassified_examples.png](file:///c:/Users/Prathamesh/Desktop/dl%20hand/outputs/figures/misclassified_examples.png)  
   Detailed inspection of 16 actual failure cases showcasing the model's top-3 competing hypotheses.

---

## 14. Limitations

1. **Unidirectional Row Scanning:** The LSTM processes the image from top to bottom. It has no mechanism to scan columns from left to right, making it less sensitive to vertical stroke continuity.
2. **Absence of 2D Local Receptive Fields:** Unlike CNNs, the LSTM does not learn small, translation-invariant 2D spatial kernels (e.g., $3 \times 3$ filters).
3. **Isolated Character Scope:** Real-world handwriting recognition typically involves connected cursive text. The model operates on segmented, isolated character boxes.

---

## 15. Conclusion

The laboratory experiment successfully designed and implemented an LSTM-based character classifier for the 47-class EMNIST Balanced dataset. Key conclusions include:
1. Treating a 2D image as a 1D sequence of horizontal row vectors ($28$ time steps $\times$ $28$ features) is effective, achieving **86.53% test accuracy** and **86.44% Macro F1-score**.
2. The recurrent cell state effectively carries stroke dependencies from the top of the character to the bottom across 28 steps without vanishing gradients.
3. With only **91,695 trainable parameters**, the model is compact, trains rapidly on CPU, and exhibits controlled generalization gap ($<4\%$).
4. Error analysis confirms that the remaining misclassifications stem from genuine orthographic ambiguities (`L/1`, `I/1`, `0/O`, `f/F`) that are difficult to resolve without linguistic sentence context.

---

## 16. Future Scope

1. **Bidirectional LSTM (BiLSTM):** Pass the sequence both top-to-bottom and bottom-to-top, or combine row-wise and column-wise LSTMs into a dual-stream architecture.
2. **Hybrid CNN-LSTM (CRNN):** Extract low-level spatial visual features with a lightweight CNN feature extractor before passing feature map slices into the LSTM.
3. **Connectionist Temporal Classification (CTC):** Extend the model to unsegmented whole-word or whole-sentence text lines using CTC loss.
4. **Language Model Integration:** Integrate an N-gram or Transformer language model to resolve single-character ambiguities using word-level context.

---

## 17. References

1. **Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017).** *EMNIST: an extension of MNIST to handwritten letters.* IEEE International Joint Conference on Neural Networks (IJCNN). [NIST EMNIST Repository](https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip).
2. **Hochreiter, S., & Schmidhuber, J. (1997).** *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
3. **Chollet, F., et al. (2024).** *Keras 3: Deep Learning for humans.* [https://keras.io](https://keras.io).
4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press.
