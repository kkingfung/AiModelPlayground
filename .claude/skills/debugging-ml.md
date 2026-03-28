# Debugging ML Models

Common issues when training ML models and how to fix them.

## Loss Not Decreasing

### Symptoms
- Loss stays constant or decreases very slowly
- Accuracy stuck at random guessing
- Model predictions are all the same

### Common Causes & Solutions

#### 1. Learning Rate Too Low

```python
# ❌ Too slow
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

# ✅ Try higher learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ✅ Use learning rate finder
from torch_lr_finder import LRFinder
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()
```

#### 2. Data Not Normalized

```python
# ❌ Raw pixel values [0, 255]
transform = transforms.ToTensor()

# ✅ Normalize to standard range
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

#### 3. Wrong Loss Function

```python
# ❌ Binary classification with multi-class loss
criterion = nn.CrossEntropyLoss()  # For multi-class
model_output = torch.sigmoid(logits)  # Binary output

# ✅ Use correct loss
criterion = nn.BCEWithLogitsLoss()  # For binary classification
```

#### 4. Dying ReLU Problem

```python
# ❌ Too many dead neurons
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),  # Might die if weights initialized poorly
    ...
)

# ✅ Use LeakyReLU or initialize properly
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.LeakyReLU(0.01),  # Allows small gradient for negative values
    ...
)

# Or initialize weights properly
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
```

## Loss Becomes NaN

### Symptoms
- Loss suddenly becomes NaN during training
- Model outputs contain NaN values

### Common Causes & Solutions

#### 1. Learning Rate Too High

```python
# ❌ Exploding gradients
optimizer = torch.optim.SGD(model.parameters(), lr=10.0)

# ✅ Lower learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# ✅ Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### 2. Numerical Instability

```python
# ❌ Can cause log(0) = -inf
loss = -torch.log(predictions)

# ✅ Add small epsilon
loss = -torch.log(predictions + 1e-8)

# ✅ Use stable loss functions
criterion = nn.CrossEntropyLoss()  # Numerically stable
```

#### 3. Bad Initialization

```python
# ❌ Poor initialization
# Weights might be too large

# ✅ Use proper initialization
model.apply(lambda m: nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) else None)
```

#### 4. Check for NaN in Data

```python
# Add data validation
def check_data(batch):
    data, labels = batch
    assert not torch.isnan(data).any(), "NaN in input data!"
    assert not torch.isinf(data).any(), "Inf in input data!"
    return batch

# In training loop
for batch in train_loader:
    batch = check_data(batch)
    # ... continue training
```

## Overfitting

### Symptoms
- Training accuracy much higher than validation
- Validation loss increases while training loss decreases
- Poor generalization to new data

### Solutions

#### 1. Add Regularization

```python
# L2 Regularization (weight decay)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4  # L2 penalty
)

# Dropout
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout = nn.Dropout(0.5)  # Drop 50% of neurons
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Only active during training
        x = self.fc2(x)
        return x
```

#### 2. Data Augmentation

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

#### 3. Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
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
    val_loss = validate(...)

    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break
```

#### 4. Reduce Model Complexity

```python
# ❌ Too complex for dataset
model = nn.Sequential(
    nn.Linear(784, 2048),
    nn.ReLU(),
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

# ✅ Simpler model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10)
)
```

#### 5. Get More Data

```python
# Use data augmentation to artificially increase dataset size
# Or collect more real data if possible
```

## Underfitting

### Symptoms
- Both training and validation accuracy are low
- Loss plateaus at a high value
- Model too simple for the task

### Solutions

#### 1. Increase Model Capacity

```python
# ❌ Too simple
model = nn.Linear(784, 10)  # Just one layer

# ✅ Add more layers/neurons
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
```

#### 2. Train Longer

```python
# Increase number of epochs
EPOCHS = 50  # Instead of 10
```

#### 3. Reduce Regularization

```python
# ❌ Too much regularization
model.dropout = nn.Dropout(0.8)  # Dropping too many neurons
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-2)  # Too high

# ✅ Reduce regularization
model.dropout = nn.Dropout(0.2)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
```

#### 4. Better Features

```python
# Engineer better features
# Use pre-trained embeddings
# Add domain-specific preprocessing
```

## Slow Training

### Symptoms
- Training takes very long
- GPU utilization is low
- Bottleneck in data loading

### Solutions

#### 1. Optimize DataLoader

```python
# ❌ Slow data loading
train_loader = DataLoader(dataset, batch_size=32)

# ✅ Parallel loading
train_loader = DataLoader(
    dataset,
    batch_size=64,  # Larger batch (if memory allows)
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

#### 2. Use GPU Efficiently

```python
# Check GPU usage
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

# Monitor GPU memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

#### 3. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()

    # Mixed precision forward pass
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Scaled backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 4. Profile Your Code

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    for batch in train_loader:
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Memory Issues

### Symptoms
- CUDA out of memory error
- Training crashes
- Slow performance due to swapping

### Solutions

#### 1. Reduce Batch Size

```python
# ❌ Too large
train_loader = DataLoader(dataset, batch_size=512)

# ✅ Smaller batch
train_loader = DataLoader(dataset, batch_size=32)
```

#### 2. Gradient Accumulation

```python
# Simulate larger batch size without using more memory
accumulation_steps = 4

optimizer.zero_grad()
for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 3. Clear Cache

```python
# Clear unused cache
torch.cuda.empty_cache()

# Use gradient checkpointing (for very deep models)
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(model, x):
    return checkpoint(model, x)
```

#### 4. Use Smaller Model

```python
# Reduce number of parameters
# Use depthwise separable convolutions
# Reduce feature dimensions
```

## Debugging Checklist

```python
def debug_model(model, train_loader, device):
    """
    Comprehensive debugging checks.
    """
    print("="*50)
    print("Model Debugging Report")
    print("="*50)

    # 1. Check model is on correct device
    print(f"\n1. Device: {next(model.parameters()).device}")

    # 2. Check model is in correct mode
    print(f"2. Training mode: {model.training}")

    # 3. Check data shape
    data, labels = next(iter(train_loader))
    print(f"3. Input shape: {data.shape}")
    print(f"   Label shape: {labels.shape}")

    # 4. Check forward pass
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        output = model(data)
        print(f"4. Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"   Contains NaN: {torch.isnan(output).any()}")
        print(f"   Contains Inf: {torch.isinf(output).any()}")

    # 5. Check gradients
    model.train()
    data, labels = data.to(device), labels.to(device)
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, labels)
    loss.backward()

    print(f"5. Loss: {loss.item():.4f}")
    print("   Gradient statistics:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"   {name}: mean={param.grad.abs().mean():.6f}, "
                  f"max={param.grad.abs().max():.6f}")
        else:
            print(f"   {name}: No gradient!")

    # 6. Check parameter statistics
    print("6. Parameter statistics:")
    for name, param in model.named_parameters():
        print(f"   {name}: mean={param.data.abs().mean():.6f}, "
              f"std={param.data.std():.6f}")

    print("\n" + "="*50)
    print("Debugging complete!")
    print("="*50)

# Usage
debug_model(model, train_loader, device)
```

## Quick Fixes Summary

| Problem | Quick Fix |
|---------|-----------|
| Loss not decreasing | Increase learning rate, check data normalization |
| Loss becomes NaN | Decrease learning rate, add gradient clipping |
| Overfitting | Add dropout, data augmentation, early stopping |
| Underfitting | Increase model size, train longer |
| Slow training | Increase batch size, use num_workers, mixed precision |
| Out of memory | Decrease batch size, gradient accumulation |
| Dying ReLU | Use LeakyReLU, check initialization |
| Exploding gradients | Gradient clipping, lower learning rate |
