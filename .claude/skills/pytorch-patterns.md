# PyTorch Patterns and Best Practices

Common patterns and best practices for PyTorch development.

## Model Definition

### Basic Pattern

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    """
    Brief description of model.

    Architecture:
        Input → Layer1 → Layer2 → Output
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network."""
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
```

### Using nn.Sequential

```python
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)
```

### CNN Pattern

```python
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

## Training Loop Pattern

### Basic Training Loop

```python
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()  # Set to training mode
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Move to device
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy
```

### Evaluation Pattern

```python
def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy
```

## Device Management

### Best Practice

```python
# Set device once at the start
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to device
model = MyModel().to(device)

# In training loop
for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    # ... training code
```

### Multi-GPU

```python
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(device)
```

## Saving and Loading

### Save Model

```python
# Save just the state dict (recommended)
torch.save(model.state_dict(), 'model.pth')

# Save entire model (not recommended)
torch.save(model, 'model_full.pth')

# Save checkpoint with metadata
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

### Load Model

```python
# Load state dict
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

## Memory Management

### Clear Cache

```python
# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Gradient Accumulation

```python
# For larger effective batch size without more memory
accumulation_steps = 4

optimizer.zero_grad()
for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target)

    # Normalize loss
    loss = loss / accumulation_steps

    loss.backward()

    # Update every N steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Common Patterns

### Custom Loss Function

```python
class CustomLoss(nn.Module):
    def __init__(self, weight=None):
        super(CustomLoss, self).__init__()
        self.weight = weight

    def forward(self, predictions, targets):
        # Implement custom loss logic
        loss = ...
        return loss

# Usage
criterion = CustomLoss(weight=class_weights)
```

### Learning Rate Scheduler

```python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Step decay
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

# In training loop
for epoch in range(epochs):
    train(...)
    scheduler.step()
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Usage
early_stopping = EarlyStopping(patience=5)

for epoch in range(epochs):
    train_loss = train(...)
    val_loss = evaluate(...)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

## Debugging Tips

### Check Gradients

```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
    else:
        print(f"{name}: No gradient")
```

### Gradient Clipping

```python
# Prevent gradient explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Check for NaN

```python
def check_nan(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN!")

# In training loop
output = model(data)
check_nan(output, "model output")
```
