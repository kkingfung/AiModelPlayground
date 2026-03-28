# Quick Start Guide - Domain-Specific AI

Get started building your game development AI assistant in under 10 minutes!

## Option 1: RAG Only (No GPU Required)

Perfect for testing, fast to set up, doesn't require expensive hardware.

### Step 1: Install Dependencies

```bash
pip install langchain sentence-transformers faiss-cpu
```

### Step 2: Create Sample Documentation

```bash
python rag_pipeline.py --create-samples
```

This creates sample game docs in `sample_docs/`:
- `combat_system.md`
- `inventory_system.md`
- `movement_system.md`

### Step 3: Build Knowledge Base

```bash
python rag_pipeline.py --docs-dir sample_docs --build
```

This creates a `vector_store/` directory with your indexed documents.

### Step 4: Query Your AI

```bash
# Search for specific information
python rag_pipeline.py --load --query "How does combat work?" --k 3

# Try more queries
python rag_pipeline.py --load --query "How do I sort inventory?"
python rag_pipeline.py --load --query "What affects player speed?"
```

### Step 5: Use with Your Own Docs

```bash
# 1. Put your game docs in a folder (markdown, txt files)
mkdir my_game_docs/
cp /path/to/your/docs/*.md my_game_docs/

# 2. Build vector store
python rag_pipeline.py --docs-dir my_game_docs --build --vector-store my_game_vector

# 3. Query
python rag_pipeline.py --load --vector-store my_game_vector --query "Your question here"
```

---

## Option 2: RAG + Simple Generation (Use OpenAI/Claude API)

Use RAG for retrieval + API for generation. Good balance.

### Step 1: Set up RAG (as above)

Follow Option 1 steps 1-3.

### Step 2: Use with OpenAI

```python
# example_with_openai.py
import openai
from rag_pipeline import GameDocRAG

# Initialize RAG
rag = GameDocRAG(docs_dir="sample_docs", vector_store_path="vector_store")
rag.load_vector_store()

# Retrieve relevant docs
query = "How does combat work?"
docs = rag.retrieve(query, k=3)

# Format context
context = "\n\n".join([doc.page_content for doc in docs])

# Generate with OpenAI
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a game development expert. Answer based on the provided documentation."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)

print(response.choices[0].message.content)
```

---

## Option 3: Full System (RAG + Fine-tuned LLM)

Requires GPU, takes longer, but most powerful and fully local.

### Prerequisites

- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 3090, 4090, A100, etc.)
- **CUDA**: Installed and working
- **Time**: 2-3 hours for first-time setup + training

### Step 1: Install Full Dependencies

```bash
pip install torch transformers peft datasets accelerate bitsandbytes
pip install langchain sentence-transformers faiss-cpu
```

### Step 2: Build RAG Knowledge Base

```bash
# Create sample docs
python rag_pipeline.py --create-samples

# Build vector store
python rag_pipeline.py --docs-dir sample_docs --build
```

### Step 3: Create Training Data

```bash
# Generate sample training data
python fine_tuning.py --create-sample training_data.json
```

This creates `training_data.json` with example Q&A pairs.

### Step 4: Fine-tune Model (Optional but Recommended)

```bash
# Using QLoRA (4-bit) for 16GB GPU
python fine_tuning.py \
    --data training_data.json \
    --output game-dev-lora \
    --epochs 3 \
    --qlora

# This takes 1-2 hours on a consumer GPU
```

**Note**: You can skip this step and use just the base model, but fine-tuning improves domain-specific responses.

### Step 5: Run Complete Assistant

```bash
# With fine-tuned model
python game_dev_assistant.py \
    --vector-store vector_store \
    --lora game-dev-lora \
    --question "How does the combat system work?"

# Interactive mode
python game_dev_assistant.py \
    --vector-store vector_store \
    --lora game-dev-lora \
    --interactive
```

---

## Quick Test Queries

Try these questions with your assistant:

### Combat System
```bash
python game_dev_assistant.py --question "Explain the combat system"
python game_dev_assistant.py --question "How do critical hits work?"
python game_dev_assistant.py --question "What status effects are available?"
```

### Inventory
```bash
python game_dev_assistant.py --question "How do I sort inventory items?"
python game_dev_assistant.py --question "Explain the weight system"
python game_dev_assistant.py --question "How does item stacking work?"
```

### Movement
```bash
python game_dev_assistant.py --question "What's the player speed formula?"
python game_dev_assistant.py --question "What terrain modifiers exist?"
python game_dev_assistant.py --question "How does sprinting work?"
```

---

## Using with Your Own Game Project

### 1. Gather Documentation

Collect all your game documentation:
- Design documents (`.md`, `.txt`)
- API documentation
- Code comments (extract to markdown)
- Wiki pages
- Meeting notes

Put them all in one directory:
```
my_game_docs/
├── design/
│   ├── combat.md
│   ├── progression.md
│   └── economy.md
├── technical/
│   ├── architecture.md
│   └── api_reference.md
└── gameplay/
    ├── tutorial.md
    └── mechanics.md
```

### 2. Build Knowledge Base

```bash
python rag_pipeline.py --docs-dir my_game_docs --build --vector-store my_game_kb
```

### 3. Generate Training Data

Create Q&A pairs from your docs (manually or using GPT-4):

```json
[
  {
    "instruction": "How does our skill tree work?",
    "input": "",
    "output": "The skill tree uses a branching system where players unlock abilities by spending skill points. Each branch represents a different playstyle..."
  },
  {
    "instruction": "Explain our matchmaking algorithm",
    "input": "",
    "output": "Matchmaking uses a modified Elo system with additional factors for player level, region, and latency..."
  }
]
```

Save as `my_training_data.json`.

### 4. Fine-tune (Optional)

```bash
python fine_tuning.py \
    --data my_training_data.json \
    --output my-game-lora \
    --epochs 3 \
    --qlora
```

### 5. Deploy

```bash
# Run assistant
python game_dev_assistant.py \
    --vector-store my_game_kb \
    --lora my-game-lora \
    --interactive
```

---

## Jupyter Notebooks

For interactive exploration:

```bash
jupyter notebook
# Open: 01_rag_exploration.ipynb
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'langchain'"
```bash
pip install langchain sentence-transformers faiss-cpu
```

### "CUDA out of memory"
Solutions:
1. Use QLoRA (add `--qlora` flag)
2. Reduce batch size: `--batch-size 1`
3. Use smaller model: `--model microsoft/phi-2`
4. Use CPU-only RAG (Option 1)

### "Vector store not found"
Make sure you ran the build step:
```bash
python rag_pipeline.py --docs-dir sample_docs --build
```

### Slow retrieval
- Use GPU for embeddings: Change `device: 'cpu'` to `device: 'cuda'` in `rag_pipeline.py`
- Reduce number of documents retrieved: `--k 3` instead of `--k 10`

### Poor quality answers
- Add more/better documentation
- Increase retrieval: `--k 5` or `--k 10`
- Fine-tune with more training data
- Use larger base model

---

## Next Steps

1. **Add Your Docs**: Replace sample docs with your game documentation
2. **Expand Training Data**: Create 50-100 Q&A pairs from your docs
3. **Experiment**: Try different models, chunk sizes, retrieval strategies
4. **Deploy**: Set up API server for team access
5. **Iterate**: Collect feedback and improve

---

## Performance Benchmarks

### RAG Only
- **Retrieval**: ~50-200ms per query
- **Total**: ~50-200ms (no generation)
- **Memory**: ~500MB (embeddings + vector store)

### RAG + API (OpenAI)
- **Retrieval**: ~100ms
- **Generation**: ~1-3s (OpenAI API)
- **Total**: ~1-3s
- **Cost**: ~$0.001-0.01 per query

### RAG + Local LLM (7B)
- **Retrieval**: ~100ms
- **Generation**: ~2-5s (GPU), ~30-60s (CPU)
- **Total**: ~2-5s
- **Memory**: ~8-12GB VRAM (4-bit), ~14-16GB (8-bit)

---

## Resources

- **Full Guide**: See `README.md`
- **RAG Patterns**: `.claude/skills/rag-patterns.md`
- **Fine-tuning Guide**: `.claude/skills/fine-tuning-patterns.md`
- **Agent Help**: Ask `rag-specialist` or `fine-tuning-specialist` agents

---

**Happy building! 🚀**
