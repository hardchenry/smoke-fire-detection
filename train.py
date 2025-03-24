import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class SmokeFireDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'labels')
        self.images = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Read YOLO format labels (class_id, x_center, y_center, width, height)
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            with open(label_path, 'r') as f:
                label_line = f.readline().strip().split()
                class_id = int(label_line[0])  # We only need the class ID for classification
        else:
            class_id = 0  # Default to 'no smoke/fire' if no label file or empty file
            
        return image, class_id

def get_model(model_name, num_classes=2):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'efficientnet':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{model.__class__.__name__}_best.pth')
        
    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_results(results_dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for model_name, metrics in results_dict.items():
        ax1.plot(metrics['train_acc'], label=f'{model_name} (train)')
        ax1.plot(metrics['val_acc'], label=f'{model_name} (val)')
    ax1.set_title('Accuracy vs. Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    
    for model_name, metrics in results_dict.items():
        ax2.plot(metrics['train_loss'], label=f'{model_name} (train)')
        ax2.plot(metrics['val_loss'], label=f'{model_name} (val)')
    ax2.set_title('Loss vs. Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = SmokeFireDataset('data', split='train', transform=transform)
    val_dataset = SmokeFireDataset('data', split='val', transform=transform)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Models to compare
    models_to_train = {
        'ResNet50': 'resnet50',
        'VGG16': 'vgg16',
        'EfficientNet': 'efficientnet'
    }
    
    results = {}
    
    for model_name, model_type in models_to_train.items():
        print(f'\nTraining {model_name}...')
        model = get_model(model_type).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses, val_losses, train_accuracies, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs, device
        )
        
        results[model_name] = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accuracies,
            'val_acc': val_accuracies
        }
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
        }, f'{model_name.lower()}_final.pth')
    
    # Plot and save comparison results
    plot_results(results)

if __name__ == '__main__':
    main()
