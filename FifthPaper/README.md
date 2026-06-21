# CerebroAI: Brain Tumor Classification (VGG-C with Batch Normalization)

This folder contains the implementation of **Paper 5** of the CerebroAI project, adapting the VGG-C Transform network architecture with Batch Normalization to classify brain MRI scans for tumor detection using PyTorch and PyQt5.

---

## 1. Original Research Reference
* **Title:** VGG-C Transform Model with Batch Normalization to Predict Alzheimer’s Disease through MRI Dataset
* **Authors:** Batzaya Tuvshinjargal and Heejoung Hwang
* **Affiliation:** Department of Computer Engineering, Gachon University, Gyeonggi-do, Korea
* **Journal/Publisher:** MDPI Electronics, 2022
* **Key Concept:** Utilizing a custom VGG-C network variant enhanced with Batch Normalization and quantized pixel intensity adjustments to retain maximum MRI features. Rather than standard resizing to $224 \times 224$ (which distorts the $208 \times 176$ aspect ratio), this architecture is trained directly on original aspect ratios to maximize prediction accuracy.

---

## 2. Local Implementation Overview
While the original paper focused on Alzheimer's disease prediction, our implementation adapts the **VGG-C with Batch Normalization** (`VGG_C_BN`) architecture to address **Brain Tumor Classification** across 4 diagnostic classes:
1. **Glioma**
2. **Meningioma**
3. **Pituitary Tumor**
4. **No Tumor**

### Architecture Details (`VGG_C_BN`)
The network implements deep feature extraction blocks utilizing Batch Normalization to accelerate convergence:
* **Feature Extraction (Convolutional Blocks):**
  * `Conv2d(1 -> 64, kernel=3, padding=1)` + `BatchNorm` + `ReLU` + `Conv` + `BatchNorm` + `ReLU` + `MaxPool(2x2)`
  * `Conv2d(64 -> 128, kernel=3, padding=1)` + `BatchNorm` + `ReLU` + `Conv` + `BatchNorm` + `ReLU` + `MaxPool(2x2)`
  * `Conv2d(128 -> 256, kernel=3, padding=1)` + `BatchNorm` + `ReLU` + `Conv` + `BatchNorm` + `ReLU` + `MaxPool(2x2)`
  * `Conv2d(256 -> 512, kernel=3, padding=1)` + `BatchNorm` + `ReLU` + `Conv` + `BatchNorm` + `ReLU` + `MaxPool(2x2)`
* **Classifier (Fully Connected Layers):**
  * `Flatten` (Output size: $512 \times 13 \times 13 = 86,528$ features)
  * `Linear(86528 -> 4096)` + `ReLU` + `Dropout(p=0.5)`
  * `Linear(4096 -> 4096)` + `ReLU` + `Dropout(p=0.5)`
  * `Linear(4096 -> 4)` (Output classes)

---

## 3. Dataset & Preprocessing
* **Source:** Brain MRI images for brain tumor detection (Kaggle).
* **Target Classes:** glioma, meningioma, notumor, pituitary.
* **Preprocessing Pipeline:**
  * **Grayscale Conversion:** Standardizes input channels to single-channel grayscale ($1 \times H \times W$), matching MRI properties.
  * **Resizing:** Scaled to $208 \times 208$ pixels (retaining the $1:1$ aspect ratio of the dataset scans).
  * **Normalization:** Scaled using `mean=[0.5]`, `std=[0.5]` for convergence stability.

---

## 4. Model Training & Convergence Results
* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** Adam ($\text{learning rate} = 0.001$)
* **Batch Size:** 32
* **Training Epochs:** 10

### Loss Convergence History
The training loss fell steadily over 10 epochs, demonstrating stable learning:
* **Epoch 1:** Loss = 5.4790
* **Epoch 2:** Loss = 1.0116
* **Epoch 3:** Loss = 1.0314
* **Epoch 5:** Loss = 1.2172
* **Epoch 8:** Loss = 0.9601
* **Epoch 10:** Loss = 0.9172

The final trained model weights are saved at:
`E:/Intern/MEGA-MINDS/Trained Model/model.pth`

---

## 5. Graphical User Interface (GUI)
A PyQt5 application is provided to serve as a user interface for radiologists.

### GUI Features:
* **Upload Image:** Upload brain MRI scans of patients.
* **Predict:** Performs inference via the trained model and shows the diagnostic prediction.
* **Show Confidence Graph:** Generates a real-time matplotlib bar chart showing confidence scores for all 4 diagnostic classes.

---

## 6. How to Run

### Requirements
Ensure you have the required dependencies installed:
```bash
pip install torch torchvision pyqt5 pillow numpy matplotlib
```

### Step 1: Train the Model
To start training the VGG-C BN model on the brain tumor dataset, run:
```bash
python Fifth.py
```
This script will:
1. Preprocess and load brain scans from the training path.
2. Train the network for 10 epochs.
3. Save weights to `Trained Model/model.pth`.
4. Plot and display training loss trends.

### Step 2: Start the GUI Application
To run the interactive PyQt5 classifier interface:
```bash
python FifthGUI.py
```
Use the **Upload Image** button to load a scan, click **Predict** to view classification outputs, and click **Show Confidence Graph** to visualize prediction probabilities.
