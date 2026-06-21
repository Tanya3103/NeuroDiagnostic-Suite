# CerebroAI: Alzheimer's Disease Stage Classification (Custom CNN)

This folder contains the implementation of **Paper 1** of the CerebroAI project, focusing on the early detection and stage classification of Alzheimer's disease using a custom Convolutional Neural Network (CNN) built in PyTorch.

---

## 1. Original Research Reference
* **Title:** ADD-Net: An Effective Deep Learning Model for Early Detection of Alzheimer Disease in MRI Scans
* **Authors:** Mian Muhammad Sadiq Fareed, Shahid Zikria, Gulnaz Ahmed, et al.
* **Journal/Publisher:** IEEE Access, 2022
* **Key Concept:** Early detection of Alzheimer's disease stages from brain MRI scans using a lightweight convolutional neural network with reduced parameters and computational complexity. The original paper addresses class imbalances using synthetic oversampling (SMOTE-TOMEK) and interprets features using Grad-CAM heatmaps.

---

## 2. Local Implementation Overview
Our codebase adapts the custom CNN model for the classification of brain MRI scans into 4 distinct cognitive impairment stages:
1. **Mild Impairment** (Mild Cognitive Impairment / MCI)
2. **Moderate Impairment**
3. **No Impairment** (Healthy Controls)
4. **Very Mild Impairment**

### Architecture Details (`AlzheimerCNN`)
The custom network is designed to be lightweight, avoiding deep convolutional stacks to prevent overfitting on smaller clinical datasets:
* **Feature Extraction (Convolutional Layers):**
  * `Conv2d(3 -> 32, kernel=3, stride=1, padding=1)` + `ReLU` + `MaxPool2d(2x2)`
  * `Conv2d(32 -> 64, kernel=3, stride=1, padding=1)` + `ReLU` + `MaxPool2d(2x2)`
  * `Conv2d(64 -> 128, kernel=3, stride=1, padding=1)` + `ReLU` + `MaxPool2d(2x2)`
* **Classifier (Fully Connected Layers):**
  * `Flatten` (Output size: $128 \times 28 \times 28 = 100,352$ features)
  * `Linear(100352 -> 512)` + `ReLU` + `Dropout(p=0.5)`
  * `Linear(512 -> 4)` (Output classes)

---

## 3. Dataset & Data Preprocessing
* **Source:** Combined MRI Brain Scan Dataset (Kaggle)
* **Target Classes:** Mild Impairment, Moderate Impairment, No Impairment, Very Mild Impairment
* **Preprocessing Pipeline:**
  * Image resizing to $224 \times 224$ pixels.
  * Data Augmentation: `RandomHorizontalFlip()`, `RandomRotation(10)`.
  * Grayscale-to-RGB conversion (3 channels) for model input.
  * Normalization with `mean=[0.5]`, `std=[0.5]`.

---

## 4. Model Training & Evaluation
* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** Adam ($\text{learning rate} = 0.001$)
* **Batch Size:** 32
* **Training Epochs:** 10
* **Device Fallback:** Automatically utilizes CUDA GPU acceleration if available; otherwise falls back to CPU.

### Performance Results
* **Validation Loss:** 0.6099
* **Validation Accuracy:** 71.46%

The model is saved after training at:
`E:\Intern\MEGA-MINDS\Trained Model\alzheimers_model.pth`

---

## 5. Graphical User Interface (GUI)
A user-friendly PyQt5 desktop application is provided to enable real-time diagnostic predictions. 

### Features:
* **Upload Image:** Allows clinicians to select and upload patient brain MRI scans (PNG, JPG, JPEG).
* **Interactive Display:** Visualizes the uploaded image in a designated panel.
* **Real-time Prediction:** Runs model inference instantly and outputs the predicted impairment stage.

---

## 6. How to Run

### Requirements
Ensure you have the required dependencies installed:
```bash
pip install torch torchvision pyqt5 pillow matplotlib
```

### Step 1: Train the Model
To start training the CNN model on the dataset, run:
```bash
python First.py
```
This script will:
1. Load and preprocess the training and validation images.
2. Train the model for 10 epochs.
3. Save the weights to `Trained Model/alzheimers_model.pth`.
4. Plot and display the training loss curve over epochs.
5. Print final validation loss and classification accuracy.

### Step 2: Start the GUI Application
To run the interactive PyQt5 classifier interface:
```bash
python FirstGUI.py
```
Use the **Upload Image** button to test the model's prediction capabilities interactively.
