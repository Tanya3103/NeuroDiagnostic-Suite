# NeuroDiagnostic-Suite

An advanced artificial intelligence and machine learning repository containing a suite of deep learning models, classic classifiers, and graphical user interfaces (GUIs) optimized for the detection, classification, and analysis of **Alzheimer's Disease** and **Brain Tumors** from neuroimaging scans (MRI) and structured patient records.

---

## 💻 Repository Structure & Directory Layout
```text
Task 1/
│
├── CODE/                                 # Central codebase folder
│   ├── First.py                          # Custom Alzheimer's CNN training
│   ├── FirstGUI.py                       # PyQt5 GUI for Custom Alzheimer's Classifier
│   ├── Second.py                         # SVM & Decision Tree (clinical tabular data)
│   ├── Third.py                          # ResNet-18 Transfer Learning training
│   ├── Fourth.py                         # DEMNET MRI classification (from Parquet data)
│   ├── Fifth.py                          # VGG-C BN Brain Tumor classification training
│   └── FifthGUI.py                       # PyQt5 GUI + Live confidence graph for Brain Tumor Classifier
│
├── DataSet/                              # Dataset folders
│   ├── 1,2,3/                            # Combined image dataset (Models 1, 3)
│   ├── Another/                          # Parquet dataset (Model 4)
│   ├── DataSet_5/                        # Brain tumor images (Model 5)
│   └── alzheimers_disease_data.csv       # Clinical tabular data (Model 2)
│
├── Trained Model/                        # Saved checkpoints (.pth & .pkl)
│
├── Research Paper/                       # Original PDF Research Papers 1-5
│
└── Final Report/                         # Reports and individual section documentations
    ├── First Paper/                      # [Custom CNN] References, code copy, and reports
    │   └── README.md                     # Model 1 Detailed Documentation
    ├── Second Paper/                     # [SVM & DT] References, code copy, and reports
    │   └── README.md                     # Model 2 Detailed Documentation
    ├── Third Paper/                      # [ResNet-18] References, code copy, and reports
    │   └── README.md                     # Model 3 Detailed Documentation
    ├── Fourth Paper/                     # [DEMNET] References, code copy, and reports
    │   └── README.md                     # Model 4 Detailed Documentation
    └── Fifth Paper/                      # [VGG-C BN] References, code copy, and reports
        └── README.md                     # Model 5 Detailed Documentation
```

---

## 🧠 Model Portfolio Overview
The suite integrates 5 separate diagnostic frameworks. Below is a comparison of their objectives, modalities, and performance:

| Section | Model / Architecture | Objective | Data Modality | Key Metrics | Link |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Model 1** | **Custom CNN** | Alzheimer's Stage Classification (4 Classes) | MRI Images (224x224) | Accuracy: **71.46%** | [First Paper README](file:///d:/Project/Task%201/Final%20Report/First%20Paper/README.md) |
| **Model 2** | **PCA + SVM / DT** | Alzheimer's Diagnosis Prediction | Clinical Tabular (33 features) | SVM Acc: **72.56%** / DT: **63.95%** | [Second Paper README](file:///d:/Project/Task%201/Final%20Report/Second%20Paper/README.md) |
| **Model 3** | **ResNet-18 (Transfer Learning)** | Alzheimer's Stage Classification (4 Classes) | MRI Images (150x150) | Accuracy: **99.20%** | [Third Paper README](file:///d:/Project/Task%201/Final%20Report/Third%20Paper/README.md) |
| **Model 4** | **DEMNET** | Alzheimer's Stage Classification (4 Classes) | Parquet-Encoded MRI Bytes | RMSprop Convergence | [Fourth Paper README](file:///d:/Project/Task%201/Final%20Report/Fourth%20Paper/README.md) |
| **Model 5** | **VGG-C with Batch Normalization** | Brain Tumor Classification (4 Classes) | Grayscale MRI (208x208) | Train Loss: **0.9172** | [Fifth Paper README](file:///d:/Project/Task%201/Final%20Report/Fifth%20Paper/README.md) |

---

## 🛠️ Installation & Setup
To set up the `NeuroDiagnostic-Suite` on your local environment, install the required packages using the terminal:

```bash
pip install torch torchvision numpy pandas pyarrow scikit-learn pyqt5 pillow matplotlib seaborn
```

*Note: For GPU acceleration during training (highly recommended for Models 1, 3, 4, 5), ensure you have an active NVIDIA GPU with matching CUDA drivers installed.*

---

## 🖥️ Running Code & Executing Models

All scripts are located in the [CODE](file:///d:/Project/Task%201/CODE) directory, but can also be found in their respective paper folders in [Final Report](file:///d:/Project/Task%201/Final%20Report).

### 1. Custom CNN for Alzheimer's Classification
* **Train model:**
  ```bash
  python CODE/First.py
  ```
* **Run GUI application:**
  ```bash
  python CODE/FirstGUI.py
  ```

### 2. SVM & Decision Tree (Tabular Clinical Data)
* **Train & evaluate classifiers:**
  ```bash
  python CODE/Second.py
  ```

### 3. ResNet-18 Transfer Learning
* **Train model:**
  ```bash
  python CODE/Third.py
  ```

### 4. DEMNET Model (Parquet MRI Bytes)
* **Train model:**
  ```bash
  python CODE/Fourth.py
  ```

### 5. VGG-C BN for Brain Tumor Classification
* **Train model:**
  ```bash
  python CODE/Fifth.py
  ```
* **Run GUI application (with live confidence bar charts):**
  ```bash
  python CODE/FifthGUI.py
  ```

---

## 📊 Summary of Section Documentation
For detailed insights into each research paper's context, original author details, specific model parameters, datasets, and convergence curves, navigate to the individual documentation links:
* 📄 [Model 1: Custom CNN Details](file:///d:/Project/Task%201/Final%20Report/First%20Paper/README.md)
* 📄 [Model 2: SVM/DT Details](file:///d:/Project/Task%201/Final%20Report/Second%20Paper/README.md)
* 📄 [Model 3: ResNet-18 Transfer Learning Details](file:///d:/Project/Task%201/Final%20Report/Third%20Paper/README.md)
* 📄 [Model 4: DEMNET Details](file:///d:/Project/Task%201/Final%20Report/Fourth%20Paper/README.md)
* 📄 [Model 5: VGG-C BN Tumor Classification Details](file:///d:/Project/Task%201/Final%20Report/Fifth%20Paper/README.md)
