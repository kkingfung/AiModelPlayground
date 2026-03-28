# Create New Experiment Command

Scaffold a new ML experiment following project conventions.

## Usage

When the user types `/experiment-create [experiment-name]`:

1. **Determine experiment number**:
   - Check existing experiments (01-05)
   - Assign next available number

2. **Create directory structure**:
   ```
   experiments/0X_experiment-name/
     ├── train.py
     ├── model.py
     ├── data.py
     ├── evaluate.py
     ├── utils.py
     ├── README.md
     └── requirements.txt (if needed)
   ```

3. **Generate boilerplate files**:

### train.py Template
```python
"""
Training script for [experiment-name]

This script demonstrates:
- [Key concept 1]
- [Key concept 2]
- [Key concept 3]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import Model
from data import get_dataloaders


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training"):
        # Training logic here
        pass

    return total_loss / len(train_loader), 100 * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Evaluation logic here
            pass

    return total_loss / len(test_loader), 100 * correct / total


def main():
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 10

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # Model
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), '../../models/[experiment-name].pth')
    print("\nModel saved!")


if __name__ == "__main__":
    main()
```

### model.py Template
```python
"""
Model architecture for [experiment-name]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    [Brief description of model architecture]

    Architecture:
        - Layer 1: ...
        - Layer 2: ...
        - Output: ...
    """

    def __init__(self):
        super(Model, self).__init__()
        # Define layers here

    def forward(self, x):
        """Forward pass."""
        # Implement forward pass
        return x
```

### data.py Template
```python
"""
Data loading and preprocessing for [experiment-name]
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_dataloaders(batch_size=64):
    """
    Get train and test dataloaders.

    Args:
        batch_size: Batch size for dataloaders

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transforms
    transform = transforms.Compose([
        # Add transforms here
    ])

    # Load datasets
    # train_dataset = ...
    # test_dataset = ...

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
```

### README.md Template
```markdown
# Experiment: [Experiment Name]

## What You'll Learn

This experiment teaches:
1. [Concept 1]
2. [Concept 2]
3. [Concept 3]

## The Problem

[Description of the problem this experiment solves]

## Running the Experiment

\```bash
cd experiments/0X_experiment-name
python train.py
\```

## Expected Results

After training, you should see:
- **Training Accuracy**: [expected range]
- **Test Accuracy**: [expected range]

## Key Concepts

### [Concept 1]
[Explanation]

### [Concept 2]
[Explanation]

## Experiments to Try

1. [Modification 1]
2. [Modification 2]
3. [Modification 3]

## Next Steps

Once you've completed this experiment:
- Move on to Experiment [next number]
- Try [advanced variation]
```

4. **Create Jupyter notebook**:
   - Create `notebooks/0X_experiment-name.ipynb`
   - Add interactive exploration cells

5. **Update main README**:
   - Add experiment to project structure
   - Update learning path

## Conventions

- **Naming**: `0X_lowercase-with-hyphens`
- **Numbering**: Sequential (01, 02, 03...)
- **Style**: Follow PEP 8
- **Comments**: Clear and educational
- **Type hints**: Use where helpful
