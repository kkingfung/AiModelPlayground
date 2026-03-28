# Experiment 06: Domain-Specific AI Assistant

## Overview

This experiment demonstrates how to build a domain-specific AI assistant for game development teams using RAG (Retrieval-Augmented Generation) and fine-tuning techniques.

## Goal

Create an AI assistant that can:
- Ingest game documentation, design documents, and manuals
- Answer questions about game rules and mechanics
- Provide code examples specific to your game project
- Maintain consistent style and terminology
- Be updated without retraining (RAG) while having domain understanding (fine-tuning)

## Approach

### Hybrid System: RAG + Fine-tuning

**RAG (Retrieval-Augmented Generation)**:
- For knowledge that changes frequently (documentation, rules)
- Allows updating without retraining
- Provides source citations

**Fine-tuning**:
- For consistent output style and format
- Domain-specific reasoning patterns
- Game development terminology

## Files

### Core Implementation
- `rag_pipeline.py` - Document ingestion and retrieval system
- `fine_tuning.py` - LoRA fine-tuning for domain adaptation
- `game_dev_assistant.py` - Complete assistant combining RAG + fine-tuned model
- `data_preparation.py` - Process game docs into training/retrieval format

### Example Data
- `sample_docs/` - Example game documentation
- `training_data/` - Generated training examples for fine-tuning

### Notebooks
- `01_rag_exploration.ipynb` - Explore RAG retrieval
- `02_fine_tuning_exploration.ipynb` - Experiment with fine-tuning
- `03_complete_assistant.ipynb` - Full system demo

## Quick Start

### 1. Install Dependencies

```bash
pip install langchain sentence-transformers faiss-cpu transformers peft datasets accelerate bitsandbytes
```

### 2. Prepare Your Documentation

```python
# Place your game docs in sample_docs/
python data_preparation.py --docs-dir sample_docs/
```

### 3. Build RAG Knowledge Base

```python
python rag_pipeline.py --build --docs-dir sample_docs/
```

### 4. (Optional) Fine-tune Model

```python
python fine_tuning.py --data training_data/game_dev_qa.json --epochs 3
```

### 5. Run Assistant

```python
python game_dev_assistant.py --question "How does the combat system work?"
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Question                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              RAG System (Retrieval)                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Vector DB (FAISS)                                │   │
│  │  - Game documentation chunks                      │   │
│  │  - Embeddings: sentence-transformers              │   │
│  └──────────────────┬───────────────────────────────┘   │
│                     │                                    │
│                     ▼                                    │
│           Top-K Relevant Docs                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Fine-tuned LLM (Generation)                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Base Model: Llama-2-7b / Mistral-7B              │   │
│  │  + LoRA adapters (game dev domain)                │   │
│  │                                                    │   │
│  │  Context: Retrieved docs                          │   │
│  │  Question: User query                             │   │
│  └──────────────────┬───────────────────────────────┘   │
│                     │                                    │
│                     ▼                                    │
│             Generated Answer                             │
│             + Source Citations                           │
└─────────────────────────────────────────────────────────┘
```

## Use Cases

### 1. Game Documentation Q&A
```python
assistant.ask("How do I implement the inventory system?")
# Returns: Detailed answer with code examples + source docs
```

### 2. Rule Clarification
```python
assistant.ask("What happens when a player's health reaches 0?")
# Returns: Explanation from game design docs
```

### 3. Code Generation
```python
assistant.ask("Generate code for player movement in Unity")
# Returns: Code following your project's patterns
```

### 4. Design Pattern Lookup
```python
assistant.ask("What's our approach to state management?")
# Returns: Answer from architecture docs
```

## Training Data Format

For fine-tuning, we use instruction-following format:

```json
[
  {
    "instruction": "Explain the combat system in our game",
    "input": "",
    "output": "The combat system uses a turn-based approach where each player selects an action (Attack, Defend, or Special). Damage is calculated using the formula: damage = attack_power * (1 - defense_modifier) * critical_multiplier..."
  },
  {
    "instruction": "How do I implement inventory sorting?",
    "input": "",
    "output": "To implement inventory sorting, use the InventoryManager.SortItems() method. This supports sorting by: ItemType, Rarity, Name, or Quantity. Example code:\n\n```csharp\nInventoryManager.Instance.SortItems(SortType.Rarity);\n```"
  }
]
```

## Performance Optimization

### RAG Optimization
- **Chunking**: 500-1000 tokens with 100-200 token overlap
- **Embedding Model**: `all-MiniLM-L6-v2` (fast) or `all-mpnet-base-v2` (accurate)
- **Retrieval**: MMR (Maximum Marginal Relevance) for diversity
- **Top-K**: Retrieve 5-10 documents, rerank to top 3

### Fine-tuning Optimization
- **LoRA rank**: 16 (balance between capacity and efficiency)
- **Quantization**: 8-bit for 16GB GPU, 4-bit (QLoRA) for smaller GPUs
- **Training**: 3-5 epochs, learning rate 2e-4
- **Dataset**: 100-500 high-quality examples minimum

## Evaluation

### RAG Evaluation
```python
# Retrieval quality
precision = relevant_retrieved / total_retrieved
recall = relevant_retrieved / total_relevant

# Response quality
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
score = scorer.score(reference, generated)
```

### Fine-tuning Evaluation
```python
# Perplexity on validation set
# Human evaluation of response quality
# Task-specific metrics (accuracy for classification, BLEU for generation)
```

## Deployment

### Option 1: Local Server (FastAPI)
```python
# See game_dev_assistant.py for API server implementation
uvicorn game_dev_assistant:app --host 0.0.0.0 --port 8000
```

### Option 2: Integrate with IDE
```python
# VS Code extension or Unity Editor plugin
# Query assistant from your development environment
```

### Option 3: CLI Tool
```bash
game-assistant "How does crafting work?"
```

## Next Steps

1. **Add More Documents**: Continuously update your knowledge base
2. **Improve Fine-tuning Data**: Generate more Q&A pairs from docs
3. **Add Code Search**: Integrate codebase search alongside docs
4. **Multi-modal**: Add image understanding for design diagrams
5. **Agent System**: Add tool use (code execution, file editing)

## Resources

- **LangChain Docs**: https://python.langchain.com/docs/get_started/introduction
- **PEFT/LoRA**: https://huggingface.co/docs/peft/index
- **Sentence Transformers**: https://www.sbert.net/
- **FAISS**: https://github.com/facebookresearch/faiss

## Common Issues

### Issue: Poor Retrieval Quality
**Solution**: Improve chunking strategy, try different embedding models, add metadata filtering

### Issue: Model Hallucinates
**Solution**: Use stricter prompts ("answer ONLY from context"), increase retrieval quality

### Issue: Out of Memory
**Solution**: Use QLoRA (4-bit), reduce batch size, use gradient accumulation

### Issue: Slow Inference
**Solution**: Quantize model, use smaller base model, cache common queries

## Learning Objectives

After completing this experiment, you'll understand:

✅ How to build RAG systems with vector databases
✅ How to fine-tune LLMs efficiently with LoRA
✅ When to use RAG vs fine-tuning vs both
✅ How to evaluate retrieval and generation quality
✅ How to deploy domain-specific AI assistants
✅ How to create training data from documentation
✅ Best practices for production RAG systems

---

**Estimated Time**: 2-3 days (RAG: 1 day, Fine-tuning: 1-2 days)
**Difficulty**: High (combines multiple advanced techniques)
**Prerequisites**: Understanding of transformers, embeddings, neural network training
