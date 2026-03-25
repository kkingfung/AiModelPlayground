# AI Model Playground

A learning repository for AI/ML model development experiments.

## Getting Started

1. Install Python 3.8+ if you haven't already
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── experiments/
│   ├── 01_mnist_basics/          # Image classification fundamentals
│   ├── 02_sentiment_analysis/    # NLP and text processing
│   ├── 03_transfer_learning/     # Using pre-trained models
│   ├── 04_text_generation/       # Generative models
│   └── 05_custom_architecture/   # Building from scratch
├── utils/                        # Shared utilities
├── models/                       # Saved models
├── data/                         # Datasets (gitignored)
└── notebooks/                    # Jupyter notebooks for exploration
```

## Learning Path

### 1. MNIST Basics (Start Here!)
Learn the fundamentals:
- Neural network architecture
- Training and validation loops
- Loss functions and optimizers
- Model evaluation

### 2. Sentiment Analysis
Dive into NLP:
- Text preprocessing and tokenization
- Word embeddings
- RNN/LSTM architectures
- Real-world text classification

### 3. Transfer Learning
Leverage pre-trained models:
- Fine-tuning for custom tasks
- Feature extraction
- Image classification with ResNet/VGG

### 4. Text Generation
Create your own text:
- Language models
- GPT-2 fine-tuning
- Creative applications

### 5. Custom Architecture
Build something unique:
- Design your own model
- Experiment with novel architectures
- Combine different techniques

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [FastAI Course](https://course.fast.ai/)
- [Papers with Code](https://paperswithcode.com/)

## Tips

- Start simple and build complexity gradually
- Always visualize your data first
- Monitor training with TensorBoard
- Experiment with hyperparameters
- Read the errors carefully - they're teaching you!
