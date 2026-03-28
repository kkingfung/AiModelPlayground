# Multimodal AI for Game Development - Complete Guide

## Overview

This guide shows how to build a multimodal AI assistant that understands **both text and images** for game development documentation.

## Why Multimodal?

Traditional RAG (text-only) misses important visual information:

### ❌ Text-Only Limitations
- Can't search by screenshot
- Can't analyze UI mockups
- Can't understand diagrams
- Can't compare visual designs
- Bug reports with images are difficult

### ✅ Multimodal Advantages
- **Search by screenshot**: "Find similar UI designs"
- **Visual bug analysis**: Upload screenshot + description
- **Diagram understanding**: State machines, architecture diagrams
- **Design references**: Search character art by example
- **UI documentation**: Screenshot + description retrieval

## Architecture

```
┌─────────────────────────────────────────────────┐
│         User Input                               │
│  Text: "How does inventory UI work?"            │
│  Image: inventory_screenshot.png (optional)     │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         CLIP Encoder (Unified Embedding)         │
│  ┌──────────────┐         ┌─────────────────┐  │
│  │ Text Encoder │         │ Vision Encoder  │  │
│  │ (Transformer)│         │ (ViT)           │  │
│  └──────┬───────┘         └────────┬────────┘  │
│         └──────────┬────────────────┘           │
│                    ▼                             │
│         512-dim Embedding Vector                 │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│      Multimodal Vector Store (FAISS)             │
│  ┌───────────────────────────────────────────┐  │
│  │ Text Documents (embedded)                 │  │
│  │ - design_doc.md → [0.12, 0.45, ...]      │  │
│  │                                            │  │
│  │ Images (embedded)                         │  │
│  │ - inventory_ui.png → [0.23, 0.67, ...]   │  │
│  │ - combat_diagram.png → [0.89, 0.12, ...] │  │
│  └───────────────────────────────────────────┘  │
│                    │                             │
│                    ▼                             │
│        Top-K Results (Text + Images)             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│    Vision-Language Model (Optional)              │
│    GPT-4V / LLaVA / Gemini                      │
│                                                  │
│    Generates answer grounded in                  │
│    retrieved text + images                       │
└─────────────────────────────────────────────────┘
```

## Quick Start (3 Options)

### Option 1: CLIP-based Search Only (Fastest, No GPU Required)

Perfect for: Image search, finding similar UI designs

```bash
# 1. Install
pip install transformers pillow torch faiss-cpu

# 2. Create sample docs with images
python multimodal_assistant.py --create-sample sample_mm_docs

# 3. Add your screenshots to sample_mm_docs/images/

# 4. Build multimodal vector store
python multimodal_assistant.py \
    --docs-dir sample_mm_docs \
    --build \
    --save mm_vector_store

# 5. Search by text (returns both text and images)
python multimodal_assistant.py \
    --load mm_vector_store \
    --query "combat UI health bars" \
    --k 5

# 6. Search by image (find similar)
python multimodal_assistant.py \
    --load mm_vector_store \
    --query-image my_screenshot.png \
    --k 5
```

**What this gives you:**
- ✅ Text-to-image search ("find UI screenshots")
- ✅ Image-to-image search (find similar designs)
- ✅ Unified search across docs and images
- ✅ Fast (CPU-friendly)
- ❌ No natural language answers (just retrieval)

---

### Option 2: CLIP + GPT-4V (Best Quality, Cloud API)

Perfect for: Production quality, visual understanding

```bash
# 1. Install CLIP for retrieval
pip install transformers pillow torch faiss-cpu openai

# 2. Build vector store (same as Option 1)
python multimodal_assistant.py --docs-dir sample_mm_docs --build --save mm_store

# 3. Use with GPT-4V for generation
```

```python
# multimodal_gpt4v_example.py
from multimodal_assistant import MultimodalGameDevAssistant
import openai
import base64

# Load assistant
assistant = MultimodalGameDevAssistant(vector_store_path="mm_store")

# Query
query = "How does the inventory UI work?"
response = assistant.ask(query, k=3)

# Get top image result
image_results = [r for r in response['results'] if r['item']['type'] == 'image']

if image_results:
    image_path = image_results[0]['item']['path']

    # Encode for GPT-4V
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()

    # Ask GPT-4V
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=500
    )

    print(gpt_response.choices[0].message.content)
```

**What this gives you:**
- ✅ Best visual understanding
- ✅ Natural language answers
- ✅ Grounded in retrieved images
- ✅ No GPU needed locally
- ❌ API costs (~$0.01-0.05 per query with images)
- ❌ Requires internet

---

### Option 3: CLIP + LLaVA (Fully Local, GPU Required)

Perfect for: Privacy, full control, no API costs

```bash
# 1. Install (requires GPU with 12GB+ VRAM)
pip install transformers pillow torch faiss-gpu accelerate

# 2. Build vector store
python multimodal_assistant.py --docs-dir sample_mm_docs --build --save mm_store

# 3. Use with LLaVA (local vision-language model)
```

```python
# multimodal_llava_example.py
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor
from PIL import Image
import torch

# Load LLaVA (first time will download ~13GB)
model_name = "llava-hf/llava-1.5-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
image_processor = CLIPImageProcessor.from_pretrained(model_name)

# Load assistant
from multimodal_assistant import MultimodalGameDevAssistant
assistant = MultimodalGameDevAssistant(vector_store_path="mm_store")

# Query and retrieve
response = assistant.ask("How does inventory work?", k=3)

# Get top image
image_results = [r for r in response['results'] if r['item']['type'] == 'image']

if image_results:
    image_path = image_results[0]['item']['path']
    image = Image.open(image_path)

    # Ask LLaVA
    prompt = "USER: <image>\nDescribe this UI and explain how it works.\nASSISTANT:"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    image_tensor = image_processor(images=image, return_tensors="pt")["pixel_values"].to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            pixel_values=image_tensor,
            max_new_tokens=200
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer.split("ASSISTANT:")[-1])
```

**What this gives you:**
- ✅ Fully local (no API)
- ✅ Privacy (no data sent to cloud)
- ✅ Natural language answers
- ✅ No ongoing costs
- ❌ Requires GPU (12GB+ VRAM)
- ❌ Slower than GPT-4V
- ❌ Lower quality than GPT-4V

---

## Use Cases

### 1. UI/UX Documentation

```bash
# Add UI screenshots with captions
python multimodal_assistant.py --build --docs-dir ui_screenshots

# Search for specific UI patterns
python multimodal_assistant.py \
    --load mm_store \
    --query "inventory grid layout with sort buttons" \
    --modality image
```

### 2. Bug Report Analysis

```python
# User reports bug with screenshot
bug_screenshot = "bug_item_overlap.png"
bug_description = "Items overlapping in inventory at high resolutions"

# Find similar UI states
response = assistant.ask(
    query="inventory UI layout",
    k=5,
    modality="image"
)

# Compare with expected state using VLM
# (GPT-4V or LLaVA code here)
```

### 3. Design Reference Search

```bash
# Find similar character designs
python multimodal_assistant.py \
    --load mm_store \
    --query-image concept_art/warrior_draft.png \
    --modality image \
    --k 10
```

### 4. Technical Diagram Understanding

```python
# Add architecture diagrams
assistant.vector_store.add_image(
    "diagrams/combat_state_machine.png",
    caption="Combat state machine: Idle -> PlayerTurn -> EnemyTurn -> Result",
    metadata={"type": "diagram", "system": "combat"}
)

# Search for state machine diagrams
results = assistant.ask("state machine transitions", modality="image")
```

## Organizing Your Multimodal Documentation

### Directory Structure

```
game_docs/
├── text/
│   ├── design/
│   │   ├── combat_system.md
│   │   ├── inventory_system.md
│   │   └── progression.md
│   └── technical/
│       ├── architecture.md
│       └── api_reference.md
│
├── images/
│   ├── ui_screenshots/
│   │   ├── combat_ui.png
│   │   ├── combat_ui.txt          # Caption file
│   │   ├── inventory_ui.png
│   │   └── inventory_ui.txt
│   │
│   ├── diagrams/
│   │   ├── combat_state_machine.png
│   │   ├── combat_state_machine.txt
│   │   ├── data_flow.png
│   │   └── data_flow.txt
│   │
│   └── concept_art/
│       ├── character_warrior.png
│       ├── character_warrior.txt
│       ├── environment_forest.png
│       └── environment_forest.txt
```

### Caption File Format

```
# combat_ui.txt
Combat user interface showing health bars at top, action buttons (Attack, Defend, Special) at bottom, turn indicator on right, and character portraits on left. Resolution: 1920x1080, Version: v1.2
```

### Best Practices

1. **Always add captions**: Either `.txt` files or via metadata
2. **Use consistent naming**: `feature_type_variant.png`
3. **Tag liberally**: Add rich metadata
4. **Organize by type**: screenshots, diagrams, concept art
5. **Version control images**: Track changes to UI over time

## Performance Benchmarks

### CLIP Embedding Speed
- **CPU**: ~50-100 images/sec (small batch)
- **GPU**: ~500-1000 images/sec

### Search Speed (1000 images + 5000 text chunks)
- **Query**: ~10-50ms
- **Return top-5**: ~10-50ms

### End-to-End (CLIP retrieval + GPT-4V generation)
- **Retrieval**: ~50ms
- **GPT-4V**: ~1-3s
- **Total**: ~1-3s

### Memory Usage
- **CLIP model**: ~600MB
- **Vector index (1000 items)**: ~50MB
- **LLaVA-7B**: ~14GB VRAM

## Tips & Tricks

### 1. Improve Search Quality

```python
# Fine-tune CLIP on your specific game assets
# This dramatically improves domain-specific retrieval

from transformers import CLIPModel, CLIPProcessor, Trainer

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Prepare pairs: (image, caption) from your game
# Train for 10-20 epochs

# Result: Much better at finding YOUR game's UI patterns
```

### 2. Hybrid Search (Text + Visual)

```python
# Search by both text and reference image
text_embedding = clip_embed_text("inventory grid")
image_embedding = clip_embed_image("reference_ui.png")

# Combine (average or weighted sum)
combined_embedding = 0.5 * text_embedding + 0.5 * image_embedding

# Search with combined query
results = search_with_embedding(combined_embedding)
```

### 3. Metadata Filtering

```python
# Search only within specific screen
results = assistant.ask(
    "health bar",
    k=5,
    metadata_filter={"screen": "combat", "version": "v1.2"}
)
```

### 4. Batch Processing

```python
# Process entire screenshot folder at once
screenshots_dir = Path("screenshots")

for img_path in screenshots_dir.glob("*.png"):
    # Auto-generate caption with BLIP
    caption = generate_caption(str(img_path))

    # Add to store
    assistant.vector_store.add_image(
        str(img_path),
        caption=caption,
        metadata={"auto_captioned": True}
    )
```

## Troubleshooting

### "CUDA out of memory"
**Solutions**:
- Use CLIP-base instead of CLIP-large
- Use CPU for CLIP (slower but works)
- Batch process images in smaller groups
- Use faiss-cpu instead of faiss-gpu

### "Poor retrieval quality"
**Solutions**:
- Add better captions to images
- Fine-tune CLIP on your domain
- Increase k (retrieve more results)
- Use hybrid search (text + image)

### "Images not loading"
**Solutions**:
- Check file paths are absolute or relative to correct directory
- Ensure images are RGB (not RGBA or grayscale)
- Check image file permissions

## Next Steps

1. **Add your screenshots**: Organize UI screenshots with captions
2. **Add diagrams**: State machines, architecture diagrams
3. **Add concept art**: Character designs, environments
4. **Build vector store**: Process all assets
5. **Test queries**: Try text and image searches
6. **Integrate with VLM**: Add GPT-4V or LLaVA for generation
7. **Deploy**: Set up API server for team access

## Complete Example

See `multimodal_assistant.py` for full implementation with:
- CLIP-based multimodal embeddings
- Unified text+image vector store
- Text-to-image and image-to-image search
- Metadata filtering and organization
- Save/load functionality

## Resources

- **Agent**: `.claude/agents/multimodal-specialist.md`
- **Patterns**: `.claude/skills/multimodal-patterns.md`
- **CLIP Paper**: https://arxiv.org/abs/2103.00020
- **LLaVA**: https://llava-vl.github.io/
- **GPT-4V**: https://platform.openai.com/docs/guides/vision

---

**Ready to build multimodal AI for your game dev team! 🎨🤖**
