"""
MNIST Experiment Builder - Interactive Hyperparameter Tuning

This script lets you:
- Build custom network architectures
- Experiment with different hyperparameters
- Compare results across experiments
- Learn what each parameter does

Perfect for understanding how neural networks work!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime


class FlexibleNN(nn.Module):
    """
    Flexible neural network with customizable architecture.
    """

    def __init__(self, layer_sizes, dropout_rate=0.2, activation='relu'):
        """
        Args:
            layer_sizes: List of layer sizes, e.g., [784, 128, 64, 10]
            dropout_rate: Dropout probability
            activation: 'relu', 'tanh', or 'sigmoid'
        """
        super(FlexibleNN, self).__init__()

        self.flatten = nn.Flatten()
        self.layer_sizes = layer_sizes
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            # Add activation and dropout for all but last layer
            if i < len(layer_sizes) - 2:
                layers.append(self.activation)
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


class ExperimentConfig:
    """Configuration for an experiment."""

    def __init__(self, name='experiment', **kwargs):
        self.name = name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Model architecture
        self.layer_sizes = kwargs.get('layer_sizes', [784, 128, 64, 10])
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.activation = kwargs.get('activation', 'relu')

        # Training hyperparameters
        self.batch_size = kwargs.get('batch_size', 64)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.epochs = kwargs.get('epochs', 10)
        self.optimizer_type = kwargs.get('optimizer', 'adam')

        # Data augmentation
        self.use_augmentation = kwargs.get('use_augmentation', False)

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'layer_sizes': self.layer_sizes,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'optimizer': self.optimizer_type,
            'use_augmentation': self.use_augmentation
        }


class Experiment:
    """Run and track a single experiment."""

    def __init__(self, config: ExperimentConfig, device='cpu'):
        self.config = config
        self.device = device

        # Results
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.best_test_acc = 0

        # Model
        self.model = None
        self.optimizer = None
        self.criterion = None

    def prepare_data(self):
        """Prepare data loaders."""
        if self.config.use_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(
            root='../../data', train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            root='../../data', train=False, download=True, transform=test_transform
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )

    def build_model(self):
        """Build model based on config."""
        self.model = FlexibleNN(
            layer_sizes=self.config.layer_sizes,
            dropout_rate=self.config.dropout_rate,
            activation=self.config.activation
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        # Choose optimizer
        if self.config.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate
            )
        elif self.config.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.config.learning_rate, momentum=0.9
            )
        elif self.config.optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=self.config.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader, desc="Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return total_loss / len(self.train_loader), 100 * correct / total

    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(self.test_loader), 100 * correct / total

    def run(self):
        """Run the full experiment."""
        print(f"\n{'='*60}")
        print(f"Running: {self.config.name}")
        print(f"{'='*60}")
        print(f"Architecture: {' → '.join(map(str, self.config.layer_sizes))}")
        print(f"Activation: {self.config.activation}")
        print(f"Dropout: {self.config.dropout_rate}")
        print(f"Optimizer: {self.config.optimizer_type.upper()}")
        print(f"Learning Rate: {self.config.learning_rate}")
        print(f"Batch Size: {self.config.batch_size}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Data Augmentation: {self.config.use_augmentation}")
        print(f"{'='*60}\n")

        self.prepare_data()
        self.build_model()

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}\n")

        # Training loop
        for epoch in range(self.config.epochs):
            print(f"Epoch {epoch + 1}/{self.config.epochs}")

            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)

            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc

            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        print(f"\n{'='*60}")
        print(f"Best Test Accuracy: {self.best_test_acc:.2f}%")
        print(f"Final Test Accuracy: {self.test_accs[-1]:.2f}%")
        print(f"{'='*60}\n")

    def save_results(self, save_dir='experiments/01_mnist_basics/results'):
        """Save experiment results."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        results = {
            'config': self.config.to_dict(),
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_losses': self.test_losses,
            'test_accs': self.test_accs,
            'best_test_acc': self.best_test_acc,
            'final_test_acc': self.test_accs[-1] if self.test_accs else 0
        }

        filename = f"{self.config.name}_{self.config.timestamp}.json"
        filepath = Path(save_dir) / filename

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filepath}")


def interactive_mode():
    """Interactive experiment builder."""
    print("\n" + "="*60)
    print("MNIST Experiment Builder")
    print("="*60)

    # Get experiment name
    name = input("\nExperiment name (default: experiment): ").strip() or "experiment"

    # Architecture
    print("\n--- Architecture ---")
    print("Input layer is fixed at 784 (28x28 image)")
    print("Output layer is fixed at 10 (digits 0-9)")

    hidden_layers_str = input("Hidden layer sizes (comma-separated, e.g., 128,64): ").strip()
    if hidden_layers_str:
        hidden_layers = [int(x.strip()) for x in hidden_layers_str.split(',')]
        layer_sizes = [784] + hidden_layers + [10]
    else:
        layer_sizes = [784, 128, 64, 10]

    # Activation
    print("\n--- Activation Function ---")
    print("1. ReLU (default, most common)")
    print("2. Tanh")
    print("3. Sigmoid")
    activation_choice = input("Choose activation (1-3): ").strip() or "1"
    activation_map = {'1': 'relu', '2': 'tanh', '3': 'sigmoid'}
    activation = activation_map.get(activation_choice, 'relu')

    # Dropout
    dropout_str = input("\nDropout rate (0.0-0.5, default: 0.2): ").strip()
    dropout_rate = float(dropout_str) if dropout_str else 0.2

    # Training hyperparameters
    print("\n--- Training Hyperparameters ---")

    lr_str = input("Learning rate (default: 0.001): ").strip()
    learning_rate = float(lr_str) if lr_str else 0.001

    batch_str = input("Batch size (default: 64): ").strip()
    batch_size = int(batch_str) if batch_str else 64

    epochs_str = input("Number of epochs (default: 10): ").strip()
    epochs = int(epochs_str) if epochs_str else 10

    # Optimizer
    print("\n--- Optimizer ---")
    print("1. Adam (default, adaptive learning rate)")
    print("2. SGD (classic, with momentum)")
    print("3. RMSprop")
    opt_choice = input("Choose optimizer (1-3): ").strip() or "1"
    opt_map = {'1': 'adam', '2': 'sgd', '3': 'rmsprop'}
    optimizer = opt_map.get(opt_choice, 'adam')

    # Data augmentation
    aug_choice = input("\nUse data augmentation? (y/N): ").strip().lower()
    use_augmentation = aug_choice == 'y'

    # Create config
    config = ExperimentConfig(
        name=name,
        layer_sizes=layer_sizes,
        dropout_rate=dropout_rate,
        activation=activation,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        optimizer=optimizer,
        use_augmentation=use_augmentation
    )

    # Confirm
    print("\n" + "="*60)
    print("Experiment Configuration:")
    print("="*60)
    for key, value in config.to_dict().items():
        if key != 'timestamp':
            print(f"  {key}: {value}")
    print("="*60)

    confirm = input("\nRun this experiment? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("Experiment cancelled.")
        return

    # Run experiment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    experiment = Experiment(config, device)
    experiment.run()
    experiment.save_results()


def predefined_experiments():
    """Run a series of predefined comparison experiments."""
    print("\n" + "="*60)
    print("Predefined Experiments - Comparing Different Configurations")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    configs = [
        # Baseline
        ExperimentConfig(
            name='baseline',
            layer_sizes=[784, 128, 64, 10],
            dropout_rate=0.2,
            activation='relu',
            learning_rate=0.001,
            epochs=10
        ),

        # Deep network
        ExperimentConfig(
            name='deep_network',
            layer_sizes=[784, 256, 128, 64, 32, 10],
            dropout_rate=0.2,
            activation='relu',
            learning_rate=0.001,
            epochs=10
        ),

        # High learning rate
        ExperimentConfig(
            name='high_lr',
            layer_sizes=[784, 128, 64, 10],
            dropout_rate=0.2,
            activation='relu',
            learning_rate=0.01,
            epochs=10
        ),

        # With augmentation
        ExperimentConfig(
            name='with_augmentation',
            layer_sizes=[784, 128, 64, 10],
            dropout_rate=0.2,
            activation='relu',
            learning_rate=0.001,
            epochs=10,
            use_augmentation=True
        ),

        # Different activation
        ExperimentConfig(
            name='tanh_activation',
            layer_sizes=[784, 128, 64, 10],
            dropout_rate=0.2,
            activation='tanh',
            learning_rate=0.001,
            epochs=10
        )
    ]

    results = []

    for config in configs:
        experiment = Experiment(config, device)
        experiment.run()
        experiment.save_results()

        results.append({
            'name': config.name,
            'best_acc': experiment.best_test_acc,
            'final_acc': experiment.test_accs[-1]
        })

    # Print summary
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    print(f"{'Experiment':<25} {'Best Acc':<12} {'Final Acc':<12}")
    print("-"*60)
    for result in results:
        print(f"{result['name']:<25} {result['best_acc']:>10.2f}% {result['final_acc']:>10.2f}%")
    print("="*60)


def main():
    print("\n🔬 MNIST Experiment Builder")
    print("\nWhat would you like to do?")
    print("1. Interactive mode (build custom experiment)")
    print("2. Run predefined comparison experiments")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        interactive_mode()
    elif choice == '2':
        predefined_experiments()
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
