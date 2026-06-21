import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Dataset path from Kaggle
data_dir = "E://Intern//MEGA-MINDS//DataSet//1,2,3//archive (1)//Combined Dataset"

# Image preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load training and validation data
train_data = datasets.ImageFolder(root=data_dir+'/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = datasets.ImageFolder(root=data_dir+'/test', transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Define CNN model
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 classes
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Compile model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
num_epochs = 10
train_acc = []
train_loss = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_acc = 100 * correct / total
    train_acc.append(epoch_acc)
    train_loss.append(running_loss / len(train_loader))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {epoch_acc:.2f}%")

# Save trained model
torch.save(model.state_dict(), "E://Intern/MEGA-MINDS//Trained Model//alzheimers_cnn.pth")

# Evaluate model
val_labels_list = []
val_preds = []
val_pred_probs = []
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to('cuda' if torch.cuda.is_available() else 'cpu')
        labels_list = labels.numpy()
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, preds = torch.max(outputs, 1)
        val_preds.extend(preds.cpu().numpy())
        val_pred_probs.extend(probs)
        val_labels_list.extend(labels_list)
val_preds = np.array(val_preds)
val_pred_probs = np.array(val_pred_probs)

# Classification report
print("Classification Report:\n", classification_report(val_labels_list, val_preds))

# Confusion matrix
conf_matrix = confusion_matrix(val_labels_list, val_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_data.class_to_idx.keys(), yticklabels=train_data.class_to_idx.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
plt.figure()
for i in range(4):  # 4 classes
    fpr, tpr, _ = roc_curve(np.array(val_labels_list) == i, val_pred_probs[:, i])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Plot training accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_acc, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_loss, label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()
