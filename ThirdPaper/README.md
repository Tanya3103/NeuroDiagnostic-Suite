# CerebroAI: Alzheimer's Classification using ResNet-18 Transfer Learning

This folder contains the implementation of **Paper 3** of the CerebroAI project, focusing on Alzheimer's disease stage classification from brain MRI scans using a fine-tuned ResNet-18 architecture via deep transfer learning.

---

## 1. Original Research Reference
* **Title:** Accurate Detection of Alzheimer’s Disease Using Lightweight Deep Learning Model on MRI Data
* **Authors:** Ahmed A. Abd El-Latif, Samia Allaoua Chelloug, Maali Alabdulhafith, Mohamed Hammad
* **Journal/Publisher:** Diagnostics 2023, 13, 1216
* **Key Concept:** Designing and evaluating lightweight end-to-end deep learning architectures (consisting of only 7 layers) to achieve high-accuracy binary and multi-class classification on MRI datasets with minimal computational latency and parameter complexity.

---

## 2. Local Implementation Overview
Our codebase utilizes a pre-trained **ResNet-18** CNN architecture from `torchvision.models` for transfer learning, replacing its final fully connected classification head to categorize input brain scans into 4 cognitive impairment stages.

### Architecture Details
* **Backbone:** ResNet-18 (pre-trained on ImageNet-1K weights, yielding robust spatial feature extractors).
* **Classification Head:** Linear projection layers mapping 512 input features to 4 outputs (`model.fc = nn.Linear(512, 4)`).
* **Hardware Acceleration:** Automatically targets GPU execution via PyTorch CUDA interface if available, optimizing batch-processing throughput.

---

## 3. Dataset & Data Preprocessing
* **Source:** Combined Dataset containing MRI scans.
* **Target Classes:** Non-Demented, Very Mild Demented, Mild Demented, Moderate Demented.
* **Preprocessing Pipeline:**
  * Resizing MRI inputs to $150 \times 150$ pixels.
  * Image tensor conversion.
  * Normalizing across three channels with `mean=[0.5, 0.5, 0.5]`, `std=[0.5, 0.5, 0.5]`.

---

## 4. Model Training & Optimization
* **Loss Function:** Cross-Entropy Loss (handles multi-class outputs).
* **Optimizer:** Adam ($\text{learning rate} = 0.001$).
* **Batch Size:** 32.
* **Training Epochs:** 10.

### Performance Results
* **Training Accuracy:** Reached a final training accuracy of **99.20%**.
* **Validation Performance:** Attained a weighted average **F1-score of 98%**, showcasing strong generalization capabilities on unseen MRI test scans.
* **Loss Convergence:** Steady decrease in loss values over 10 epochs, verifying stable gradient descent updates.

### Output Visualizations
After evaluation, the script generates:
1. **Classification Report:** Detailed precision, recall, and F1-score breakdown for each impairment stage.
2. **Confusion Matrix Heatmap:** Displays correct vs. misclassified count distributions.
3. **Multi-class ROC & AUC Curves:** Visualizes true positive vs. false positive rates for each of the 4 categories.
4. **Accuracy & Loss Curves:** Plotted over epochs to visualize training progress.

The final fine-tuned model is saved at:
`E:\Intern\MEGA-MINDS\Trained Model\alzheimers_cnn.pth`

---

## 5. How to Run

### Requirements
Ensure you have the required dependencies installed:
```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn
```

### Execution
Run the training script to train the model, evaluate it, and plot results:
```bash
python Third.py
```
This script will:
1. Load the train and test subsets of the Combined Dataset.
2. Load ResNet-18 with pre-trained ImageNet weights and modify the final fully connected layer.
3. Train the model for 10 epochs, printing epoch progress to the console.
4. Save the trained weights to the `Trained Model` folder.
5. Print a comprehensive classification report on the validation dataset.
6. Display Confusion Matrix and class-specific ROC/AUC curves.
7. Output training accuracy and loss curves.
