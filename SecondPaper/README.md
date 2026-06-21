# CerebroAI: Alzheimer's Prediction from Structured Patient Data (SVM & Decision Tree)

This folder contains the implementation of **Paper 2** of the CerebroAI project, focusing on classifying and predicting Alzheimer's disease diagnosis from structured patient clinical data using classic Machine Learning algorithms (Support Vector Machines and Decision Trees) combined with Principal Component Analysis (PCA).

---

## 1. Original Research Reference
* **Title:** An Efficient Machine Learning Technique for Early Detection of Alzheimer’s Disease
* **Authors:** Karn Singh, Prof. Aditi Purohit, Dr. Sanjay Sharma
* **Affiliation:** Department of AIML, Technocrats Institute of Technology Excellence, Bhopal, India
* **Journal:** International Journal for Research in Applied Science & Engineering Technology (IJRASET), April 2024
* **Key Concept:** Applying feature extraction, dimensionality reduction (PCA), and classification algorithms (SVM, Decision Trees, Random Forests) to optimize patient dataset features, enhancing diagnostic sensitivity and specificity for early-stage Alzheimer's disease.

---

## 2. Local Implementation Overview
Our codebase implements a complete data-science pipeline that loads structured clinical patient profiles and predicts the likelihood of Alzheimer's disease diagnosis ($\text{Diagnosis} \in \{0, 1\}$).

### Algorithm Pipeline
1. **Feature Preprocessing:**
   * Drop clinical identifiers like `DoctorInCharge`.
   * Impute missing values with column means.
   * Standardize numerical values using `StandardScaler` to bring features onto a common scale.
2. **Dimensionality Reduction (PCA):**
   * Projects the 33-dimensional clinical feature space onto a reduced **10-dimensional space** ($n_{\text{components}} = 10$) using Principal Component Analysis to capture maximum variance and prevent overfitting.
3. **Model Selection:**
   * **Support Vector Classifier (SVC):** Trained using a `linear` kernel with probability estimates enabled.
   * **Decision Tree Classifier (DT):** Trained as a baseline model for comparison.

---

## 3. Dataset & Preprocessing
* **Source:** Kaggle's Alzheimer's Disease Dataset (containing 2,149 patient records).
* **Features Used (33 total):**
  * *Demographic/Clinical:* `Age`, `Gender`, `Ethnicity`, `EducationLevel`, `BMI`, `Smoking`, `AlcoholConsumption`, `PhysicalActivity`, `DietQuality`, `SleepQuality`.
  * *Medical History:* `FamilyHistoryAlzheimers`, `CardiovascularDisease`, `Diabetes`, `Depression`, `HeadInjury`, `Hypertension`.
  * *Vital Signs & Blood Chemistry:* `SystolicBP`, `DiastolicBP`, `CholesterolTotal`, `CholesterolLDL`, `CholesterolHDL`, `CholesterolTriglycerides`.
  * *Cognitive & Functional Assessments:* `MMSE` (Mini-Mental State Examination), `FunctionalAssessment`, `MemoryComplaints`, `BehavioralProblems`, `ADL` (Activities of Daily Living), `Confusion`, `Disorientation`, `PersonalityChanges`, `DifficultyCompletingTasks`, `Forgetfulness`.
* **Data Split:** 80% Training, 20% Testing.

---

## 4. Model Performance & Evaluation
Models are evaluated on the 20% test subset. The results are summarized below:

| Model | Accuracy | F1-Score (Class 0 - No AD) | F1-Score (Class 1 - AD) | Weighted Avg F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **72.56%** | **80.0%** | **55.0%** | **71.0%** |
| **Decision Tree (DT)** | 63.95% | 72.0% | 51.0% | 64.0% |

### Diagnostics Generated:
* **Confusion Matrix:** Shows the misclassification rate (particularly highlighting class-wise recall).
* **ROC Curve & AUC Score:** Demonstrates the classifier’s true positive vs. false positive trade-off (plotted for the SVM model).
* **Precision-Recall Curve:** Visualizes the model precision-recall trade-off under different decision thresholds.

Trained models are saved at:
* **SVM:** `E:/Intern/MEGA-MINDS/Trained Model/alzheimers_svm.pkl`
* **Decision Tree:** `E:/Intern/MEGA-MINDS/Trained Model/alzheimers_dt.pkl`

---

## 5. How to Run

### Requirements
Ensure you have the required dependencies installed:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

### Execution
Run the main script to preprocess the data, train both models, print evaluation reports, and display performance curves:
```bash
python Second.py
```
This script will:
1. Load `alzheimers_disease_data.csv`.
2. Standardize features and reduce dimensionality using PCA ($n=10$).
3. Train the SVM and Decision Tree classifiers.
4. Print classification reports to the console.
5. Display the SVM Confusion Matrix, ROC Curve, and Precision-Recall Curve.
6. Export the models as `.pkl` files to the Trained Model folder.
