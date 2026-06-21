import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
data_path = "E://Intern//MEGA-MINDS//DataSet//alzheimers_disease_data.csv"
df = pd.read_csv(data_path)

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns.tolist())
print("Missing Values:", df.isnull().sum())

# Drop irrelevant column
df.drop(columns=['DoctorInCharge'], inplace=True)

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Select relevant features and target variable
features = [
    'PatientID', 'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking', 'AlcoholConsumption',
    'PhysicalActivity', 'DietQuality', 'SleepQuality', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
    'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
    'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
    'MemoryComplaints', 'BehavioralProblems', 'ADL', 'Confusion', 'Disorientation', 'PersonalityChanges',
    'DifficultyCompletingTasks', 'Forgetfulness'
]
target = "Diagnosis"

# Encode categorical features (if needed)
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])

# Split features and target
X = df[features]
y = df[target]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)

# Train Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Evaluate models
y_pred_svm = svm_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print("SVM Accuracy:", accuracy_svm)
print("Decision Tree Accuracy:", accuracy_dt)

# Generate classification reports
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM Confusion Matrix")
plt.show()

# ROC Curve for SVM
y_score_svm = svm_model.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_score_svm, pos_label=1)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure()
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label='SVM ROC curve (area = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - SVM')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
y_precision, y_recall, _ = precision_recall_curve(y_test, y_score_svm)
plt.figure()
plt.plot(y_recall, y_precision, color='red', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - SVM')
plt.show()

# Save models
joblib.dump(svm_model, "E:/Intern/MEGA-MINDS/Trained Model/alzheimers_svm.pkl")
joblib.dump(dt_model, "E:/Intern/MEGA-MINDS/Trained Model/alzheimers_dt.pkl")
print("Models saved successfully!")
