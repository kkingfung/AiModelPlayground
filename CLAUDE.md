# AiModelPlayground - AI/ML Learning Platform

## Project Overview

**AI Model Playground** is an educational machine learning platform designed for hands-on experimentation and progressive learning from fundamentals to advanced AI development.

- **Purpose**: Learning AI/ML through practical experiments
- **Framework**: PyTorch + Transformers + Jupyter
- **Structure**: Progressive experiments (beginner → advanced)
- **Focus**: Education, experimentation, best practices

**Tech Stack**: PyTorch, TensorFlow (optional), Transformers, Jupyter, Python 3.8+

## Project Structure

```
experiments/
  01_mnist_basics/       - Neural network fundamentals (COMPLETE)
  02_sentiment_analysis/ - NLP and text classification (TODO)
  03_transfer_learning/  - Fine-tuning pre-trained models (TODO)
  04_text_generation/    - Generative language models (TODO)
  05_custom_architecture/- Advanced custom experiments (TODO)

notebooks/               - Interactive Jupyter notebooks
utils/                   - Shared utilities
models/                  - Saved trained models (gitignored)
data/                    - Datasets (gitignored)
```

## Coding Conventions

### Python Style

```python
# Class names: PascalCase
class SimpleNN(nn.Module):
    """
    A simple neural network for MNIST classification.

    Architecture:
        Input → 128 → 64 → Output
    """

    def __init__(self):
        super(SimpleNN, self).__init__()
        # Fields: _camel_case (private) or snake_case
        self._hidden_size = 128
        self.network = nn.Sequential(...)

    # Methods: snake_case
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

# Functions: snake_case
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch and return metrics."""
    pass

# Constants: UPPER_CASE
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
```

### Documentation Style

- **Docstrings**: Google-style or NumPy-style
- **Comments**: English (code) / Japanese (when needed for clarity)
- **Type hints**: Use Python 3.8+ type annotations

```python
def evaluate(model: nn.Module,
            test_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> tuple[float, float]:
    """
    Evaluate model on test set.

    Args:
        model: Neural network model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run on (cpu/cuda)

    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    pass
```

## Available Agents

### Project Management
```
ml-planner           - ML project planning, experiment design, roadmap
```

### Development & Training
```
ml-researcher        - Research ML architectures, papers, state-of-the-art
model-trainer        - Training loop implementation, optimization, hyperparameters
data-engineer        - Data loading, preprocessing, augmentation
notebook-developer   - Jupyter notebook creation, interactive experimentation
```

### Quality & Optimization
```
code-reviewer        - Python code quality, ML best practices
model-evaluator      - Model evaluation, metrics analysis, performance testing
performance-optimizer- Training speed, memory optimization, GPU utilization
```

### Documentation
```
doc-writer           - Technical documentation, experiment reports
tutorial-creator     - Create learning materials, step-by-step guides
```

## How to Use Agents

**List available agents**:
```
/agents
```

**Invoke an agent** (natural language):
```
"Use ml-researcher to find the best architecture for sentiment analysis"
"Call model-trainer to help optimize my training loop"
"Ask code-reviewer to review my custom CNN implementation"
```

Claude will automatically invoke the appropriate agent using the Task tool.

## Available Skills

Skills in `.claude/skills/` provide reusable patterns:

### ML/DL Fundamentals
- `pytorch-patterns.md` - PyTorch model patterns, training loops
- `data-loading.md` - Dataset, DataLoader, transforms
- `model-evaluation.md` - Metrics, validation, testing strategies

### Architecture & Design
- `neural-architectures.md` - Common architectures (CNN, RNN, Transformer)
- `loss-functions.md` - Choosing and implementing loss functions
- `optimization.md` - Optimizers, learning rate schedules

### Advanced Topics
- `transfer-learning.md` - Fine-tuning pre-trained models
- `generative-models.md` - GANs, VAEs, diffusion models
- `nlp-patterns.md` - Tokenization, embeddings, transformers
- `computer-vision.md` - Image classification, detection, segmentation

### Tools & Workflow
- `jupyter-workflow.md` - Effective notebook usage
- `model-serialization.md` - Saving/loading models, checkpoints
- `tensorboard-logging.md` - Training visualization
- `debugging-ml.md` - Common issues and debugging strategies

## Available Commands

Quick actions via slash commands:

```
/experiment-create [name]  - Scaffold new experiment
/train [experiment]        - Run training script
/evaluate [model]          - Evaluate saved model
/visualize [results]       - Create visualizations
/notebook [experiment]     - Create Jupyter notebook
/validate-all              - Run all validations
```

## Common Workflows

### Starting a New Experiment

1. **Plan the experiment**:
   ```
   /experiment-create sentiment-analysis
   ```

2. **Implement**:
   - Create model class
   - Define training loop
   - Add evaluation metrics

3. **Train**:
   ```python
   python experiments/02_sentiment_analysis/train.py
   ```

4. **Evaluate**:
   ```python
   python experiments/02_sentiment_analysis/evaluate.py
   ```

5. **Document**:
   - Update experiment README
   - Add results to main documentation

### Code Review Workflow

```
"code-reviewer agent, please review experiments/01_mnist_basics/mnist_simple.py"
```

The agent will:
- Check PyTorch best practices
- Verify training loop correctness
- Suggest optimizations
- Identify potential bugs

## Best Practices

### Model Development

1. **Start Simple**: Begin with simple baseline models
2. **Validate Data**: Always visualize and understand your data first
3. **Monitor Training**: Use TensorBoard or logging
4. **Test Incrementally**: Test each component separately
5. **Save Checkpoints**: Save models at regular intervals

### Code Organization

```python
# experiments/XX_name/
#   train.py          - Training script
#   model.py          - Model definitions
#   data.py           - Data loading/processing
#   evaluate.py       - Evaluation script
#   utils.py          - Helper functions
#   README.md         - Experiment documentation
#   requirements.txt  - Experiment-specific dependencies
```

### Experiment Tracking

Track these for each experiment:
- Model architecture
- Hyperparameters
- Training/validation losses
- Final metrics
- Training time
- Hardware used (CPU/GPU)

## Resources

- **PyTorch Docs**: https://pytorch.org/docs/
- **Hugging Face**: https://huggingface.co/docs
- **Papers with Code**: https://paperswithcode.com/
- **Deep Learning Book**: https://www.deeplearningbook.org/
- **FastAI**: https://course.fast.ai/

---
Last Updated: 2026-03-25
