# AiModelPlayground Quick Reference

## Getting Started (5 minutes)

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run first experiment
cd experiments/01_mnist_basics
python mnist_simple.py
```

## Common Commands

### Training
```bash
# Train a model
python experiments/01_mnist_basics/mnist_simple.py

# With custom parameters
python train.py --epochs 20 --batch-size 128 --lr 0.001
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open specific notebook
jupyter notebook notebooks/01_mnist_exploration.ipynb
```

### Claude Code Commands
```bash
# Create new experiment
/experiment-create sentiment-analysis

# Train model
/train mnist

# Evaluate model
/evaluate mnist_simple

# Create notebook
/notebook experiment-name

# Validate all code
/validate-all
```

## Quick PyTorch Patterns

### Model Definition
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)
```

### Training Loop
```python
model.train()
for data, target in train_loader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

### Evaluation
```python
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        # Compute metrics
```

## Common Issues

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce batch_size |
| Loss is NaN | Lower learning rate, add gradient clipping |
| Training too slow | Use GPU, increase num_workers |
| Import errors | Check venv is activated |

## Useful Agents

```bash
# Research architectures
"ml-researcher agent, find best architecture for text classification"

# Optimize training
"model-trainer agent, help optimize my training loop"

# Review code
"code-reviewer agent, review my model implementation"

# Data pipeline help
"data-engineer agent, optimize my DataLoader"
```

## GPU Usage

```python
# Check GPU availability
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

## Saving/Loading Models

```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

## Hyperparameters

### Good Starting Points
- Learning rate: 0.001 (Adam), 0.01 (SGD)
- Batch size: 32-128
- Epochs: 10-50
- Dropout: 0.2-0.5

### Tuning Tips
- If loss not decreasing: increase LR
- If loss becomes NaN: decrease LR
- If overfitting: add dropout, data augmentation
- If underfitting: increase model size, train longer

## Data Transforms

```python
# Training (with augmentation)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Validation (no augmentation)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

## Project Structure

```
experiments/
  01_mnist_basics/      ← Start here
    train.py           ← Main training script
    model.py           ← Model definition
    README.md          ← Instructions

notebooks/             ← Interactive exploration
models/                ← Saved models (.pth)
data/                  ← Datasets (auto-downloaded)
utils/                 ← Shared utilities
```

## Next Steps

1. **Complete MNIST basics** - experiments/01_mnist_basics/
2. **Explore interactively** - notebooks/01_mnist_exploration.ipynb
3. **Move to NLP** - Create experiment 02 with `/experiment-create`
4. **Read skills** - .claude/skills/ for advanced patterns

## Resources

- PyTorch Docs: https://pytorch.org/docs/
- PyTorch Tutorials: https://pytorch.org/tutorials/
- Papers with Code: https://paperswithcode.com/
- Hugging Face: https://huggingface.co/

## Help

```bash
# Ask Claude Code agents
"I need help with [problem]"

# Or check documentation
.claude/skills/       ← Code patterns
CLAUDE.md            ← Agent guide
README.md            ← Full documentation
```
