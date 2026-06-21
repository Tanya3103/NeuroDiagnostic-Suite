import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define the model class (matching training model architecture)
class VGG_C_BN(nn.Module):
    def __init__(self, num_classes=4):
        super(VGG_C_BN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 13 * 13, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define image preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((208, 208)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Load trained model
def load_model(model_path):
    model = VGG_C_BN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Define the GUI class
class BrainTumorClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.model = load_model("E:/Intern/MEGA-MINDS/Trained Model/model.pth")
        self.class_names = ["glioma", "meningioma", "notumor", "pituitary"]
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Brain Tumor Classification")
        self.setGeometry(500, 200, 400, 500)
        layout = QVBoxLayout()
        
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("border: 2px solid black;")
        layout.addWidget(self.image_label)
        
        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.load_image)
        layout.addWidget(self.upload_button)
        
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.clicked.connect(self.predict_image)
        layout.addWidget(self.predict_button)
        
        self.result_label = QLabel("Prediction: ", self)
        layout.addWidget(self.result_label)
        
        self.graph_button = QPushButton("Show Confidence Graph", self)
        self.graph_button.clicked.connect(self.show_graph)
        layout.addWidget(self.graph_button)
        
        self.setLayout(layout)
    
    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(300, 300))
            self.image_path = file_path
    
    def predict_image(self):
        if not hasattr(self, 'image_path'):
            QMessageBox.warning(self, "Warning", "Please upload an image first.")
            return
        
        image = preprocess_image(self.image_path)
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.softmax(output, dim=1).squeeze().numpy()
            predicted_class = np.argmax(probabilities)
            prediction = self.class_names[predicted_class]
        
        self.result_label.setText(f"Prediction: {prediction}")
        self.probabilities = probabilities
    
    def show_graph(self):
        if not hasattr(self, 'probabilities'):
            QMessageBox.warning(self, "Warning", "Please make a prediction first.")
            return
        
        plt.figure(figsize=(8,5))
        plt.bar(self.class_names, self.probabilities, color=['blue', 'green', 'red', 'purple'])
        plt.xlabel('Tumor Type')
        plt.ylabel('Confidence Score')
        plt.title('Classification Confidence Scores')
        plt.ylim([0, 1])
        plt.show()

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BrainTumorClassifier()
    window.show()
    sys.exit(app.exec_())
