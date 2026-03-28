# Multimodal AI Patterns - Quick Reference

Patterns for building AI systems that understand both text and images.

## Core Concept: Unified Embedding Space

```
Text: "Combat UI with health bars"  ─┐
                                      ├─→ CLIP ─→ [Embedding Vector]
Image: combat_ui.png                ─┘            (512 dimensions)

Both mapped to same space → Can compare text-to-image similarity!
```

## CLIP Quick Start

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Embed text
text_inputs = processor(text=["A screenshot of a game UI"], return_tensors="pt")
text_features = model.get_text_features(**text_inputs)

# Embed image
image = Image.open("screenshot.png")
image_inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**image_inputs)

# Compute similarity
similarity = torch.cosine_similarity(text_features, image_features)
print(f"Similarity: {similarity.item():.3f}")
```

## Multimodal Vector Store Pattern

```python
import faiss
import numpy as np

class MultimodalVectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatIP(512)  # Inner product for cosine sim
        self.items = []  # Metadata

    def add_text(self, text: str, metadata: dict):
        embedding = clip_embed_text(text)
        self.index.add(np.array([embedding]))
        self.items.append({"type": "text", "content": text, **metadata})

    def add_image(self, image_path: str, caption: str, metadata: dict):
        embedding = clip_embed_image(image_path)
        self.index.add(np.array([embedding]))
        self.items.append({"type": "image", "path": image_path, **metadata})

    def search(self, query_text: str, k: int = 5):
        query_embedding = clip_embed_text(query_text)
        distances, indices = self.index.search(np.array([query_embedding]), k)

        return [self.items[idx] for idx in indices[0]]

# Usage
store = MultimodalVectorStore()
store.add_text("Combat UI with health bars", {"source": "docs.md"})
store.add_image("ui_screenshot.png", "Combat screen", {"type": "ui"})

# Search returns both text and images!
results = store.search("health bar UI")
```

## Vision-Language Models

### LLaVA (Local)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load LLaVA
model = AutoModelForCausalLM.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

# Ask about image
image = Image.open("screenshot.png")
prompt = "USER: <image>\nDescribe this UI and its elements.\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### GPT-4 Vision (API)

```python
import openai
import base64

def ask_gpt4v(image_path: str, question: str) -> str:
    # Encode image
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode()

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content

# Usage
answer = ask_gpt4v("bug_screenshot.png", "What UI elements are visible?")
```

## Image Captioning (BLIP)

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path: str) -> str:
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")

    out = model.generate(**inputs, max_length=50)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

# Auto-caption all images
for img in Path("screenshots").glob("*.png"):
    caption = generate_caption(str(img))
    print(f"{img.name}: {caption}")
```

## Use Case Patterns

### 1. UI Documentation Search

```python
# Index UI screenshots with descriptions
store.add_image(
    "inventory_ui.png",
    caption="Inventory screen with 6x4 grid, sort buttons, and weight display",
    metadata={"screen": "inventory", "type": "ui"}
)

# Search by text
results = store.search("inventory grid layout", k=3, modality="image")

# Returns relevant screenshots
for result in results:
    print(f"Found: {result['path']} - {result['caption']}")
```

### 2. Bug Report Analysis

```python
# User uploads screenshot + description
bug_image = "bug_overlap.png"
bug_text = "Items overlapping in inventory"

# Find similar UI states
similar = store.search_by_image(bug_image, k=5)

# Analyze with VLM
analysis = ask_gpt4v(
    bug_image,
    f"Analyze this bug: {bug_text}. What's wrong with the UI?"
)

print(f"Analysis: {analysis}")
```

### 3. Design Reference Search

```python
# Search by example image
query_image = "character_concept_art.png"

# Find similar designs (image-to-image search)
similar_designs = store.search_by_image(query_image, k=5, modality="image")

for design in similar_designs:
    print(f"Similar: {design['path']}")
    print(f"  {design['caption']}")
    print(f"  Score: {design['score']:.3f}")
```

### 4. Multimodal RAG

```python
# Complete workflow
def answer_with_visual_context(question: str):
    # 1. Retrieve relevant text + images
    results = store.search(question, k=5)

    # 2. Separate text and images
    text_context = [r for r in results if r['type'] == 'text']
    image_results = [r for r in results if r['type'] == 'image']

    # 3. Format context
    context = "\n".join([r['content'] for r in text_context])

    # 4. Generate answer with VLM (if images found)
    if image_results:
        answer = ask_gpt4v(
            image_results[0]['path'],
            f"Context: {context}\n\nQuestion: {question}"
        )
    else:
        answer = ask_llm(f"Context: {context}\n\nQuestion: {question}")

    return answer, image_results

# Usage
answer, images = answer_with_visual_context("How does the inventory UI work?")
print(answer)
print(f"Referenced images: {[img['path'] for img in images]}")
```

## Image Preprocessing

```python
def preprocess_for_clip(image_path: str, max_size: int = 512) -> Image.Image:
    """画像を前処理."""
    img = Image.open(image_path).convert("RGB")

    # Resize if too large
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    return img

def preprocess_screenshot(image_path: str) -> Image.Image:
    """スクリーンショット用前処理."""
    img = Image.open(image_path).convert("RGB")

    # Remove black bars if present
    # Crop to content area
    # Enhance contrast if needed

    return img
```

## Metadata Best Practices

```python
# Rich metadata for images
image_metadata = {
    "type": "ui_screenshot",
    "screen": "inventory",
    "resolution": "1920x1080",
    "date": "2024-03-26",
    "tags": ["inventory", "UI", "grid", "items"],
    "related_docs": ["inventory_system.md"],
    "ui_elements": ["grid", "sort_buttons", "weight_display", "close_button"],
    "game_state": "in_game",
    "version": "v1.2.0"
}

store.add_image(
    "inventory_ui_v1.2.png",
    caption="Inventory UI with grid layout and weight system",
    metadata=image_metadata
)

# Now can filter by metadata
results = store.search("inventory", filter={"screen": "inventory", "version": "v1.2.0"})
```

## CLIP Model Variants

```python
# Base (fastest, smallest)
model = "openai/clip-vit-base-patch32"  # 151M params, 512 dim

# Large (better quality)
model = "openai/clip-vit-large-patch14"  # 428M params, 768 dim

# Japanese-specific
model = "rinna/japanese-clip-vit-b-16"  # For Japanese text

# Domain-specific (fine-tuned)
model = "fine-tuned-game-ui-clip"  # Your custom model
```

## Fine-tuning CLIP

```python
# Fine-tune CLIP on your game assets
from transformers import CLIPModel, CLIPProcessor, Trainer, TrainingArguments

# Load base model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare dataset
# Pairs of (image, caption) from your game
dataset = [
    ("inventory_ui.png", "Inventory screen with grid layout"),
    ("combat_ui.png", "Combat UI with health bars and action buttons"),
    # ...
]

# Training
training_args = TrainingArguments(
    output_dir="./game-ui-clip",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    learning_rate=1e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_dataset
)

trainer.train()
```

## Evaluation

### Image-Text Retrieval

```python
def evaluate_retrieval(test_queries, ground_truth_images):
    """Evaluate retrieval accuracy."""
    correct = 0

    for query, expected_image in zip(test_queries, ground_truth_images):
        results = store.search(query, k=5, modality="image")
        retrieved_paths = [r['item']['path'] for r in results]

        if expected_image in retrieved_paths:
            correct += 1

    accuracy = correct / len(test_queries)
    return accuracy

# Test
test_queries = [
    "inventory grid",
    "combat health bars",
    "character selection screen"
]
ground_truth = [
    "inventory_ui.png",
    "combat_ui.png",
    "char_select.png"
]

accuracy = evaluate_retrieval(test_queries, ground_truth)
print(f"Retrieval accuracy: {accuracy:.2%}")
```

## Common Patterns Summary

### Text → Image Search
```python
results = store.search("combat UI", modality="image")
# Returns: Screenshots of combat UI
```

### Image → Similar Images
```python
results = store.search_by_image("example_ui.png", modality="image")
# Returns: Similar UI screenshots
```

### Multimodal Search (Both)
```python
results = store.search("inventory system")
# Returns: Both text docs and screenshots about inventory
```

### Image Understanding
```python
answer = ask_gpt4v("screenshot.png", "What UI elements are present?")
# Returns: Detailed description of UI elements
```

## Performance Tips

1. **Batch Processing**: Embed multiple images at once
```python
embeddings = clip_embed_images_batch(image_paths, batch_size=32)
```

2. **Cache Embeddings**: Don't recompute
```python
# Save embeddings
np.save("embeddings.npy", embeddings)

# Load later
embeddings = np.load("embeddings.npy")
```

3. **Use GPU**: 10-100x faster
```python
model.to("cuda")
```

4. **Quantization**: Smaller, faster
```python
model = model.half()  # FP16
```

## Resources

- **CLIP**: https://github.com/openai/CLIP
- **LLaVA**: https://llava-vl.github.io/
- **BLIP**: https://github.com/salesforce/BLIP
- **GPT-4V**: https://platform.openai.com/docs/guides/vision
- **Gemini**: https://ai.google.dev/gemini-api/docs/vision
