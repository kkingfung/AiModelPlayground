# Quick Start Guide

## Installation (5 minutes)

1. **Create a virtual environment**:
   ```bash
   python -m venv venv

   # On Windows:
   venv\Scripts\activate

   # On Mac/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will download PyTorch, TensorFlow, and other ML libraries. It may take a few minutes.

## Your First Experiment (10 minutes)

Let's train your first neural network!

```bash
cd experiments/01_mnist_basics
python mnist_simple.py
```

**What you'll see:**
- Dataset downloading (first time only)
- Training progress bars
- Accuracy improving each epoch
- Beautiful visualizations of predictions

**Expected output:**
```
Using device: cuda  # or cpu if no GPU
Loading MNIST dataset...
Training samples: 60000
Test samples: 10000

Epoch 1/10
Training: 100%|████████| 938/938 [00:15<00:00, loss=0.2341, acc=93.12%]
Evaluating: 100%|████████| 157/157 [00:02<00:00]
Train Loss: 0.2518, Train Acc: 92.58%
Test Loss: 0.1234, Test Acc: 96.32%
...
```

## Understanding the Results

After training, you'll see:

1. **Training History Plot**: Shows how loss decreases and accuracy increases
2. **Predictions Visualization**: 10 sample images with true vs predicted labels
3. **Saved Model**: `models/mnist_simple.pth` - your trained model!

**What does 97% accuracy mean?**
- Out of 10,000 test images, your model correctly identified 9,700 digits!
- That's better than many humans can do!

## Next Steps

### Option 1: Experiment with MNIST
Try modifying the code:
- Change learning rate (line 131): `LEARNING_RATE = 0.01`
- Add more layers to the network (class SimpleNN)
- Increase epochs: `EPOCHS = 20`

### Option 2: Explore Interactively
```bash
jupyter notebook notebooks/01_mnist_exploration.ipynb
```

This lets you experiment with the data visually!

### Option 3: Move to Next Experiment
Once you're comfortable with MNIST, try:
```bash
cd experiments/02_sentiment_analysis
```

## Troubleshooting

**Import error for torch:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory:**
- Reduce batch size: `BATCH_SIZE = 32`

**Slow training:**
- Normal! First epoch is slowest
- GPU can be 10-100x faster than CPU

## Key Concepts Checklist

After this experiment, you should understand:
- ✅ Neural networks have layers that process data
- ✅ Training is an iterative process (epochs)
- ✅ Loss measures how wrong predictions are
- ✅ Backpropagation adjusts weights to improve accuracy
- ✅ Test accuracy tells us how well the model generalizes

## Resources

- **Stuck?** Check `experiments/01_mnist_basics/README.md` for detailed explanations
- **Visual learner?** 3Blue1Brown's Neural Network series on YouTube
- **Want theory?** Read Chapter 1 of Deep Learning Book (free online)

## Remember

**You don't need to understand everything immediately!**

AI/ML is learned by doing. Run the code, see what happens, change things, break things, fix things. That's how you learn!

Happy experimenting! 🚀
