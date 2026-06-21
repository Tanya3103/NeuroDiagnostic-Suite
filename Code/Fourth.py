import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import io
from sklearn.model_selection import train_test_split

# Dataset Path (Parquet File)
data_path = "E:\\Intern\\MEGA-MINDS\\DataSet\\Another\\Alzheimer MRI Disease Classification Dataset\\Data\\train-00000-of-00001-c08a401c53fe5312.parquet"

# Load dataset
df = pd.read_parquet(data_path)
print("Dataset Loaded:", df.shape)

# Define Image Dataset Class
class AlzheimerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_data = self.data.iloc[idx, 0]  # First column contains image data
        label = self.data.iloc[idx, -1]  # Last column contains labels
        
        # Extract byte data from dictionary
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image_data = image_data['bytes']
        else:
            raise TypeError("Image format not recognized. Ensure dataset contains valid byte encodings.")
        
        # Convert byte data to image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image, label

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((176, 176)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Train-Test Split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = AlzheimerDataset(train_df, transform=transform)
val_dataset = AlzheimerDataset(val_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define DEMNET Model
class DEMNET(nn.Module):
    def __init__(self):
        super(DEMNET, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.7)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256 * 11 * 11, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)  # 4 Classes

    def forward(self, x):
        x = self.pool(self.batchnorm1(self.relu(self.conv1(x))))
        x = self.pool(self.batchnorm2(self.relu(self.conv2(x))))
        x = self.pool(self.batchnorm3(self.relu(self.conv3(x))))
        x = self.pool(self.batchnorm4(self.relu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.dropout3(self.fc3(x))
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, optimizer
model = DEMNET().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save model
torch.save(model.state_dict(), "E:/Intern/MEGA-MINDS/Trained Model/demnet.pth")

# Plot Training and Validation Loss
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()
