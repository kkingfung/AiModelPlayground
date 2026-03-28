# Getting Started - AI Tools for Game Development

**Get up and running with AI tools in 15 minutes**

---

## 🎯 What You'll Learn

By the end of this guide, you'll be able to:
- ✅ Classify game assets automatically
- ✅ Generate game text (item descriptions, dialogue)
- ✅ Analyze player reviews
- ✅ Optimize models for deployment

**Time Required**: 15-20 minutes

---

## 📋 Prerequisites

### Required
- **Python 3.8+** installed
- **8GB RAM** minimum (16GB recommended)
- **10GB free disk space**

### Optional (for better performance)
- **NVIDIA GPU** with CUDA support
- **16GB+ RAM** for larger models

### Check Your Setup
```bash
python --version  # Should show 3.8 or higher
pip --version     # Should be installed
```

---

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/AiModelPlayground.git
cd AiModelPlayground
```

### Step 2: Install Dependencies

Choose your experiment:

#### For Computer Vision + NLP
```bash
cd experiments/07_practical_game_ai
pip install -r requirements.txt
```

#### For Reinforcement Learning
```bash
cd experiments/08_reinforcement_learning
pip install -r requirements.txt
```

#### For Everything
```bash
# Install all dependencies
pip install -r experiments/07_practical_game_ai/requirements.txt
pip install -r experiments/08_reinforcement_learning/requirements.txt
```

**Estimated time**: 5-10 minutes (depending on internet speed)

### Step 3: Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import gymnasium; print('Gymnasium: OK')"
```

Expected output:
```
PyTorch: 2.x.x
Transformers: 4.x.x
Gymnasium: OK
```

---

## 🎮 Quick Start Examples

### Example 1: Image Classification (5 minutes)

Automatically categorize game assets into types (character, UI, item, etc.)

```bash
cd experiments/07_practical_game_ai/1_computer_vision

# Prepare sample data
mkdir -p data/sample/{character,ui,item}
# (Add a few images to each folder)

# Train classifier
python image_classifier.py \
    --train data/sample \
    --epochs 5 \
    --model-name mobilenet_v3_small

# Predict
python image_classifier.py \
    --predict path/to/new_image.png \
    --model checkpoints/best.pth
```

**What happens?**
- Model trains on your images (5 epochs = ~2 minutes)
- Saves best model to `checkpoints/best.pth`
- Predicts category with confidence score

**Output example**:
```
Top class: character
Confidence: 94.2%

Top 3 predictions:
  1. character: 94.20%
  2. item: 4.30%
  3. ui: 1.50%
```

### Example 2: Text Generation (2 minutes)

Generate item descriptions, quests, dialogue

```bash
cd experiments/07_practical_game_ai/2_nlp

# Generate item description
python text_generator.py \
    --prompt "The Sword of Flames is a legendary weapon that" \
    --max-length 100 \
    --model gpt2
```

**Output example**:
```
The Sword of Flames is a legendary weapon that was forged in the heart of a
dying star. Its blade burns with eternal fire, dealing massive damage to
enemies while protecting its wielder from frost and ice. Only those pure of
heart can wield its awesome power.
```

**Python API**:
```python
from text_generator import GameTextGenerator

gen = GameTextGenerator()

# Item description
desc = gen.generate_item_description(
    item_name="Flameblade",
    item_type="weapon",
    rarity="legendary"
)
print(desc)

# Quest text
quest = gen.generate_quest_text(
    quest_type="rescue",
    location="dark forest"
)
print(quest["title"])
print(quest["description"])
```

### Example 3: Sentiment Analysis (3 minutes)

Analyze player reviews to understand feedback

```bash
cd experiments/07_practical_game_ai/2_nlp

# Single review
python sentiment_analyzer.py \
    --text "This game is amazing! Best RPG ever!"
```

**Output**:
```
Sentiment: POSITIVE
Confidence: 99.8%
Scores: {'POSITIVE': 0.998, 'NEGATIVE': 0.002}
```

**Batch analysis**:
```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

reviews = [
    {"text": "Great game! Love it.", "rating": 5},
    {"text": "Too many bugs, unplayable", "rating": 1},
    {"text": "Good but needs improvement", "rating": 3}
]

analysis = analyzer.analyze_reviews(reviews)
print(f"Positive: {analysis['statistics']['positive_count']}")
print(f"Negative: {analysis['statistics']['negative_count']}")
```

### Example 4: Model Optimization (5 minutes)

Convert models to ONNX for 3-10x speedup

```bash
cd experiments/07_practical_game_ai/3_optimization

# Export to ONNX
python onnx_export.py \
    --model checkpoints/classifier.pth \
    --output classifier.onnx \
    --input-shape 1 3 224 224

# Benchmark
python onnx_export.py \
    --benchmark \
    --model checkpoints/classifier.pth \
    --onnx classifier.onnx
```

**Output**:
```
PyTorch (FP32):  100ms → 10 FPS
ONNX (CPU):      30ms  → 33 FPS  (3.3x faster!)
ONNX (GPU):      10ms  → 100 FPS (10x faster!)

Model size:
  PyTorch: 25.4 MB
  ONNX:    25.2 MB
```

---

## 🎯 Next Steps

### For Developers

**1. Learn the Fundamentals**
- Read [Computer Vision Guide](use-cases/computer-vision.md)
- Read [NLP Guide](use-cases/nlp.md)
- Read [RL Guide](use-cases/reinforcement-learning.md)

**2. Integrate with Your Game**
- [Unity Integration](integration/unity.md)
- [Unreal Integration](integration/unreal.md)
- [API Reference](api-reference.md)

**3. Optimize for Production**
- [Performance Optimization](best-practices/performance.md)
- [Model Compression](advanced/model-compression.md)
- [Deployment Guide](advanced/edge-deployment.md)

### For Designers/Artists

**1. Understand Capabilities**
- [Non-Technical Guide](non-technical-guide.md)
- [Asset Pipeline](asset-pipeline.md)

**2. Explore Use Cases**
- [Auto-tagging Assets](use-cases/computer-vision.md#auto-tagging)
- [UI Testing](use-cases/computer-vision.md#ui-testing)
- [Text Generation](use-cases/nlp.md#text-generation)

### For Managers

**1. Plan Integration**
- [Project Planning](project-planning.md)
- [Cost Analysis](cost-analysis.md)
- [Team Integration](team-integration.md)

**2. Review Case Studies**
- [Success Stories](case-studies/)
- [Performance Benchmarks](reference/benchmarks.md)

---

## 🔧 Common Issues

### Issue: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
pip install torch torchvision
```

### Issue: `CUDA out of memory`

**Solution**:
```bash
# Use smaller batch size
python script.py --batch-size 16  # Instead of 64

# Or use CPU
python script.py --device cpu
```

### Issue: Models run slow

**Solutions**:
1. Convert to ONNX (3-10x speedup)
2. Use quantization (FP16/INT8)
3. Use smaller model variants (MobileNet instead of ResNet)

See [Troubleshooting](reference/troubleshooting.md) for more solutions.

---

## 📚 Learning Resources

### Tutorials
- [Computer Vision Tutorial](../experiments/07_practical_game_ai/1_computer_vision/CV_GUIDE.md)
- [NLP Tutorial](../experiments/07_practical_game_ai/2_nlp/NLP_GUIDE.md)
- [RL Basics](../experiments/08_reinforcement_learning/1_basics/BASICS_GUIDE.md)

### Code Examples
- [examples/](examples/) - Ready-to-use code snippets
- [integration/](integration/) - Unity/Unreal examples

### Community
- GitHub Issues - Bug reports
- GitHub Discussions - Questions & ideas
- Discord - Real-time chat (coming soon)

---

## ✅ Checklist

Mark your progress:

- [ ] Python 3.8+ installed
- [ ] Dependencies installed
- [ ] Ran image classification example
- [ ] Ran text generation example
- [ ] Ran sentiment analysis example
- [ ] Converted model to ONNX
- [ ] Read relevant use case guides
- [ ] Ready to integrate with game project

---

## 🎉 What's Next?

**You're ready to use AI in your game!**

Choose your path:
1. **Asset Automation** → [Asset Pipeline Guide](asset-pipeline.md)
2. **Player Insights** → [Sentiment Analysis Guide](use-cases/nlp.md#sentiment-analysis)
3. **Intelligent NPCs** → [RL Guide](use-cases/reinforcement-learning.md)
4. **Production Deploy** → [Unity Integration](integration/unity.md)

**Need help?** Check [FAQ](reference/faq.md) or [Troubleshooting](reference/troubleshooting.md)

---

**Welcome to AI-powered game development! 🎮🤖🚀**
