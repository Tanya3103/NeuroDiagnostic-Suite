import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# Use installed GPU if available, otherwise fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Data paths
data_dir = "E:\\Intern\\MEGA-MINDS\\DataSet\\1,2,3\\archive (1)\\Combined Dataset"  # Update with your local dataset path
model_save_path = "E:\\Intern\\MEGA-MINDS\\Trained Model\\alzheimers_model.pth"

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define CNN model
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

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    
    # Load dataset
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize model
    model = AlzheimerCNN(num_classes=4).to(DEVICE)
    
    # Load model if it exists
    if os.path.exists(model_save_path):
        print("\n✅ Model already trained. Loading saved model...")
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    else:
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training function
        def train(model, train_loader, criterion, optimizer, device, epochs=10):
            model.train()
            train_losses = []
            for epoch in range(epochs):
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                avg_loss = running_loss / len(train_loader)
                train_losses.append(avg_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            print("Training complete!")
            return train_losses
        
        # Train the model
        train_losses = train(model, train_loader, criterion, optimizer, DEVICE, epochs=10)
        
        # Save the model
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved as {model_save_path}")
        
        # Plot training loss
        plt.plot(range(1, 11), train_losses, marker='o', linestyle='-', color='b', label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid()
        plt.show()
    
    # Model Evaluation function
    def evaluate(model, val_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_val_loss, accuracy
    
    # Evaluate the model
    val_loss, val_accuracy = evaluate(model, val_loader, nn.CrossEntropyLoss(), DEVICE)
    print("Model evaluation complete!")