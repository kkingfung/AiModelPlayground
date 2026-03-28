# Create Jupyter Notebook Command

Create an interactive Jupyter notebook for exploration or experimentation.

## Usage

When the user types `/notebook [experiment-name]`:

1. **Determine notebook type**:
   - **Exploration**: For understanding data/models
   - **Tutorial**: For learning specific concepts
   - **Experiment**: For testing ideas interactively

2. **Create notebook structure**:
   - Setup and imports
   - Data loading and exploration
   - Model definition (if applicable)
   - Training/evaluation (if applicable)
   - Visualization
   - Conclusions and next steps

3. **Save location**:
   - `notebooks/0X_experiment-name.ipynb`

## Notebook Template Structure

### Cell 1: Title and Description
```markdown
# [Experiment Name]

**Purpose**: [What this notebook demonstrates]

**Learning Goals**:
- [Goal 1]
- [Goal 2]
- [Goal 3]

**Prerequisites**:
- Understanding of [concept]
- Completed [previous experiment]
```

### Cell 2: Imports and Setup
```python
# Core imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
%matplotlib inline
%load_ext autoreload
%autoreload 2

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### Cell 3: Data Loading
```python
# Load and explore data
# [Add specific data loading code]

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

### Cell 4: Data Visualization
```python
# Visualize sample data
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    # Visualization code
    pass

plt.tight_layout()
plt.show()
```

### Cell 5: Data Analysis
```markdown
## Data Analysis

Observations:
- [Key observation 1]
- [Key observation 2]

Statistics:
- [Relevant statistics]
```

### Cell 6: Model Definition
```python
class Model(nn.Module):
    """[Model description]"""

    def __init__(self):
        super(Model, self).__init__()
        # Architecture

    def forward(self, x):
        # Forward pass
        return x

model = Model().to(device)
print(model)
```

### Cell 7: Quick Training Test
```python
# Train for 1 epoch to verify everything works

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

print("✅ Training test successful!")
```

### Cell 8: Experiments Section
```markdown
## Experiments to Try

Run each cell below to try different variations:
```

### Cell 9-N: Interactive Experiments
```python
# Experiment 1: [Description]
# Try changing [parameter] and observe the effect
```

### Final Cell: Conclusions
```markdown
## Conclusions

What we learned:
- [Learning 1]
- [Learning 2]

Next steps:
- [What to explore next]
- [Further reading]
```

## Notebook Types

### 1. Exploration Notebook
Focus: Understanding data and existing models
- Data statistics
- Visualization
- Model inspection
- Performance analysis

### 2. Tutorial Notebook
Focus: Teaching specific concepts
- Step-by-step explanations
- Interactive examples
- Exercises with solutions
- Visual demonstrations

### 3. Experiment Notebook
Focus: Testing new ideas
- Quick prototyping
- A/B comparisons
- Hyperparameter sweeps
- Ablation studies

## Output Format

```
=== Creating Jupyter Notebook ===

Notebook type: [exploration/tutorial/experiment]
Location: notebooks/0X_experiment-name.ipynb

Creating structure:
✅ Title and description
✅ Imports and setup
✅ Data loading section
✅ Visualization section
✅ Model definition
✅ Training/evaluation
✅ Experiments section
✅ Conclusions

Notebook created successfully!

To open:
  jupyter notebook notebooks/0X_experiment-name.ipynb

Or use VS Code:
  code notebooks/0X_experiment-name.ipynb
```

## Best Practices

1. **Clear Structure**: Use markdown cells to organize
2. **Run Order**: Cells should work sequentially
3. **Self-Contained**: Include all necessary imports
4. **Reproducible**: Set random seeds
5. **Well-Documented**: Explain each step
6. **Interactive**: Encourage experimentation
7. **Visual**: Include plots and visualizations
8. **Checkpoints**: Save progress periodically
