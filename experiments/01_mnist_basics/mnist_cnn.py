"""
MNIST Digit Classification - Convolutional Neural Network

This script demonstrates:
- Convolutional Neural Networks (CNNs)
- How CNNs extract spatial features
- Comparison with simple feedforward networks
- Advanced training techniques

CNNs are the foundation of computer vision!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path


class ConvNet(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.

    Architecture:
    - Conv Layer 1: 1 → 32 channels (3x3 kernel)
    - Conv Layer 2: 32 → 64 channels (3x3 kernel)
    - Max Pooling: 2x2
    - Fully Connected: 64*7*7 → 128 → 10

    Why CNNs?
    - Preserve spatial structure (unlike flattening)
    - Learn local patterns (edges, corners)
    - Translation invariant (detect features anywhere)
    - Fewer parameters than fully connected
    """

    def __init__(self):
        super(ConvNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 → 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 → 28x28
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 → 14x14 → 7x7

        # Batch normalization (stabilizes training)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv Block 1: Conv → BN → ReLU → Pool
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  # 28x28 → 14x14

        # Conv Block 2: Conv → BN → ReLU → Pool
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  # 14x14 → 7x7

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ImprovedConvNet(nn.Module):
    """
    Improved CNN with modern techniques.

    Improvements:
    - More conv layers (deeper network)
    - Residual connections (inspired by ResNet)
    - Adaptive pooling
    - Better regularization
    """

    def __init__(self):
        super(ImprovedConvNet, self).__init__()

        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Conv block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Conv block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # Feature extraction
        x = self.conv1(x)  # 28x28
        x = self.pool(x)    # 14x14

        x = self.conv2(x)  # 14x14
        x = self.pool(x)    # 7x7

        x = self.conv3(x)  # 7x7

        # Global pooling
        x = self.adaptive_pool(x)  # 1x1
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x


def train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch and return metrics."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    # Step scheduler if provided
    if scheduler is not None:
        scheduler.step()

    return total_loss / len(train_loader), 100 * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(test_loader), 100 * correct / total


def visualize_feature_maps(model, test_loader, device, save_path='feature_maps.png'):
    """Visualize convolutional feature maps."""
    model.eval()
    images, labels = next(iter(test_loader))
    sample_image = images[0:1].to(device)  # Take first image

    # Get feature maps from first conv layer
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Register hook
    if hasattr(model, 'conv1'):
        if isinstance(model.conv1, nn.Sequential):
            model.conv1[0].register_forward_hook(get_activation('conv1'))
        else:
            model.conv1.register_forward_hook(get_activation('conv1'))

    # Forward pass
    with torch.no_grad():
        _ = model(sample_image)

    # Plot feature maps
    if 'conv1' in activation:
        feature_maps = activation['conv1'].squeeze(0).cpu()  # Remove batch dim

        # Plot first 16 feature maps
        num_maps = min(16, feature_maps.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.ravel()

        for i in range(num_maps):
            axes[i].imshow(feature_maps[i], cmap='viridis')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')

        plt.suptitle('Feature Maps from First Convolutional Layer', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Feature maps saved to '{save_path}'")
        plt.show()


def visualize_filters(model, save_path='conv_filters.png'):
    """Visualize learned convolutional filters."""
    # Get first conv layer weights
    if hasattr(model, 'conv1'):
        if isinstance(model.conv1, nn.Sequential):
            weights = model.conv1[0].weight.data.cpu()
        else:
            weights = model.conv1.weight.data.cpu()

        # Normalize for visualization
        weights = (weights - weights.min()) / (weights.max() - weights.min())

        # Plot first 16 filters
        num_filters = min(16, weights.shape[0])
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        axes = axes.ravel()

        for i in range(num_filters):
            # Take first channel (grayscale input has 1 channel)
            filter_img = weights[i, 0]
            axes[i].imshow(filter_img, cmap='gray')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')

        plt.suptitle('Learned Convolutional Filters', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Filters saved to '{save_path}'")
        plt.show()


def compare_models(simple_acc, cnn_acc, improved_cnn_acc):
    """Compare model accuracies."""
    models = ['Simple NN', 'CNN', 'Improved CNN']
    accuracies = [simple_acc, cnn_acc, improved_cnn_acc]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12)

    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Model Comparison on MNIST', fontsize=14, fontweight='bold')
    plt.ylim([90, 100])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiments/01_mnist_basics/model_comparison.png')
    print("Model comparison saved to 'model_comparison.png'")
    plt.show()


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 15

    # Data loading with augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Data augmentation
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root='../../data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.MNIST(
        root='../../data',
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Choose model
    print("\nSelect model:")
    print("1. Basic CNN")
    print("2. Improved CNN")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '2':
        model = ImprovedConvNet().to(device)
        model_name = 'Improved CNN'
    else:
        model = ConvNet().to(device)
        model_name = 'Basic CNN'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"\nModel: {model_name}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nArchitecture:\n{model}\n")

    # Training loop
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    best_acc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 50)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), '../../models/mnist_cnn_best.pth')
            print(f"✓ Best model saved! (Accuracy: {best_acc:.2f}%)")

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(test_losses, label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(test_accs, label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiments/01_mnist_basics/cnn_training_history.png')
    print("\nTraining history saved to 'cnn_training_history.png'")
    plt.show()

    # Visualize feature maps and filters
    print("\nVisualizing learned features...")
    visualize_feature_maps(model, test_loader, device,
                          save_path='experiments/01_mnist_basics/feature_maps.png')
    visualize_filters(model, save_path='experiments/01_mnist_basics/conv_filters.png')

    # Save final model
    torch.save(model.state_dict(), '../../models/mnist_cnn.pth')
    print(f"\nFinal model saved to '../../models/mnist_cnn.pth'")

    print(f"\n{'='*50}")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
