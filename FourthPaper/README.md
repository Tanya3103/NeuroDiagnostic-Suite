# CerebroAI: DEMNET Model on Parquet MRI Datasets

This folder contains the implementation of **Paper 4** of the CerebroAI project, focusing on training the DEMNET (Dementia Network) model on brain MRI images stored in a parquet format with byte-encoded image records.

---

## 1. Original Research Reference
* **Title:** DEMNET: A Deep Learning Model for Early Diagnosis of Alzheimer Diseases and Dementia From MR Images
* **Authors:** Suriya Murugan, Chandran Venkatesan, M. G. Sumithra, Xiao-Zhi Gao, et al.
* **Journal/Publisher:** IEEE Access, 2021
* **Key Concept:** Designing DEMNET, a custom CNN model built from scratch with reduced parameters and computational overhead to classify dementia stages. The original model uses the Synthetic Minority Over-sampling Technique (SMOTE) to counter dataset class imbalances and occlusion sensitivity maps to visualize brain features impacting classification.

---

## 2. Local Implementation Overview
Our codebase implements the DEMNET model architecture from scratch using PyTorch and trains it on a serialized Parquet dataset.

### Model Architecture (`DEMNET`)
The architecture alternates between convolutional feature extractors and high-dropout fully connected classifier blocks to mitigate overfitting:
* **Feature Extraction (Convolutional Blocks):**
  * **Block 1:** `Conv2d(3 -> 32, kernel=3, padding=1)` + `ReLU` + `BatchNorm2d(32)` + `MaxPool2d(2x2)`
  * **Block 2:** `Conv2d(32 -> 64, kernel=3, padding=1)` + `ReLU` + `BatchNorm2d(64)` + `MaxPool2d(2x2)`
  * **Block 3:** `Conv2d(64 -> 128, kernel=3, padding=1)` + `ReLU` + `BatchNorm2d(128)` + `MaxPool2d(2x2)`
  * **Block 4:** `Conv2d(128 -> 256, kernel=3, padding=1)` + `ReLU` + `BatchNorm2d(256)` + `MaxPool2d(2x2)`
* **Classifier (Fully Connected Layers):**
  * `Flatten` (Output size: $256 \times 11 \times 11 = 30,976$ features)
  * `Linear(30976 -> 512)` + `ReLU` + `Dropout(p=0.7)`
  * `Linear(512 -> 256)` + `ReLU` + `Dropout(p=0.5)`
  * `Linear(256 -> 4)` (Output classes) + `Dropout(p=0.2)`

---

## 3. Dataset & Custom Data Loader
* **Dataset Format:** Parquet file containing columns of binary bytes representing image data, alongside class target labels.
* **Custom Dataset Handler (`AlzheimerDataset`):**
  * Parses byte-encoded image data in each parquet row.
  * Dynamically decodes bytes to standard images using `io.BytesIO` and `PIL.Image.open()`.
  * Preprocesses raw scans to uniform dimensions.
* **Target Classes:** Mild Demented, Moderate Demented, Non-Demented, Very Mild Demented.
* **Preprocessing Pipeline:**
  * Resize to $176 \times 176$ pixels.
  * Convert to Tensor.
  * Normalize with `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`.
  * Train-validation split: 80% / 20%.

---

## 4. Model Training & Optimization
* **Loss Function:** Cross-Entropy Loss (Categorical Cross-Entropy)
* **Optimizer:** RMSprop ($\text{learning rate} = 0.001$), matching the optimizer used in the paper for stable RMS gradients.
* **Batch Size:** 32
* **Training Epochs:** 10
* **Device Fallback:** Automatically utilizes CUDA GPU acceleration if available; otherwise falls back to CPU.

### Performance Evaluation
The training script prints loss convergence over 10 training epochs and plots a side-by-side **Training vs. Validation Loss** curve to help verify that the model is fitting without overfitting.

The final trained model weights are saved at:
`E:/Intern/MEGA-MINDS/Trained Model/demnet.pth`

---

## 5. How to Run

### Requirements
Ensure you have the required dependencies installed:
```bash
pip install torch torchvision pandas pyarrow numpy matplotlib scikit-learn pillow
```

### Execution
Run the training script to load the parquet file, preprocess the dataset, train DEMNET, and display loss trends:
```bash
python Fourth.py
```
This script will:
1. Parse `Data/train-00000-of-00001-c08a401c53fe5312.parquet` using pandas and pyarrow.
2. Initialize the PyTorch DataLoader, decoding image binary bytes dynamically.
3. Train the DEMNET model for 10 epochs.
4. Export the trained network parameters to the Trained Model folder.
5. Display the Training vs. Validation Loss plot.
