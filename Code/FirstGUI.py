import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QFont, QColor
from PyQt5.QtCore import Qt
from PIL import Image

# Define the CNN model
class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

# Load the trained model
def load_model(model_path, device):
    model = AlzheimerCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class AlzheimerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Alzheimer's Disease Classifier")
        self.setGeometry(100, 100, 500, 500)
        self.setStyleSheet("background-color: #f0f0f0;")
        
        title_label = QLabel("Alzheimer's Disease Classifier", self)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50;")
        
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("border: 2px dashed #7f8c8d; background-color: white;")
        self.image_label.setAlignment(Qt.AlignCenter)
        
        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.setFont(QFont("Arial", 12))
        self.upload_button.setStyleSheet("background-color: #3498db; color: white; padding: 10px; border-radius: 5px;")
        self.upload_button.clicked.connect(self.load_image)
        
        self.result_label = QLabel("Prediction: ", self)
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #e74c3c;")
        
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.upload_button, alignment=Qt.AlignCenter)
        layout.addWidget(self.result_label)
        
        self.setLayout(layout)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model("E:/Intern/MEGA-MINDS/Trained Model/alzheimers_model.pth", self.device)
        self.class_labels = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]
    
    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose an Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
        
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Predict class
            image = Image.open(file_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                prediction = self.class_labels[predicted_class]
                
                self.result_label.setText(f"Prediction: {prediction}")
                self.result_label.setStyleSheet("color: #27ae60;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AlzheimerApp()
    window.show()
    sys.exit(app.exec_())
