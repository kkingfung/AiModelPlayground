# Fine-tuning Patterns - Quick Reference

Efficient fine-tuning patterns using LoRA, QLoRA, and PEFT for domain adaptation.

## LoRA Fine-tuning (Standard)

### Basic Setup
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,                              # Rank
    lora_alpha=32,                     # Scaling
    target_modules=["q_proj", "v_proj"], # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4M || all params: 7B || trainable%: 0.057%
```

### Training
```python
training_args = TrainingArguments(
    output_dir="./game-dev-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
    warmup_ratio=0.05
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save (only adapters, ~few MB)
model.save_pretrained("./game-dev-lora")
tokenizer.save_pretrained("./game-dev-lora")
```

### Inference
```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./game-dev-lora")

# Generate
inputs = tokenizer("Explain combat system", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## QLoRA (4-bit Quantization)

### For Consumer GPUs (16GB VRAM)
```python
from transformers import BitsAndBytesConfig

# 4-bit config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True       # Nested quantization
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Rest is same as LoRA
# Can train 7B models on 16GB GPU!
```

## Dataset Preparation

### Instruction-Following Format
```python
# Standard format
data = [
    {
        "instruction": "Explain the combat system",
        "input": "",
        "output": "The combat system uses turn-based mechanics..."
    },
    {
        "instruction": "How do I sort inventory?",
        "input": "",
        "output": "Use InventoryManager.SortItems()..."
    }
]

# Format function
def format_instruction(example):
    prompt = f"""Below is an instruction for game development. Write a response.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

    return {"text": prompt}

# Apply
from datasets import Dataset
dataset = Dataset.from_list(data)
dataset = dataset.map(format_instruction)
```

### Tokenization
```python
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

tokenized = dataset.map(
    tokenize_function,
    remove_columns=dataset.column_names,
    batched=True
)
```

## LoRA Hyperparameters

### Rank (r)
```python
# Low rank: Faster, less capacity
lora_config = LoraConfig(r=8, ...)   # Small tasks

# Medium rank: Balanced
lora_config = LoraConfig(r=16, ...)  # Most common

# High rank: More capacity, slower
lora_config = LoraConfig(r=64, ...)  # Complex tasks
```

### Alpha (Scaling)
```python
# Usually alpha = 2 * rank
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,  # 2x rank
    ...
)
```

### Target Modules
```python
# For Llama-2
target_modules = ["q_proj", "v_proj"]  # Basic (attention)

# For better results
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Full coverage
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

## Dataset Creation Strategies

### From Documentation
```python
import openai

def generate_qa_from_docs(doc_content):
    """Generate Q&A pairs from documentation."""
    prompt = f"""Based on this documentation, generate 5 question-answer pairs.

Documentation:
{doc_content}

Format as JSON:
[
  {{"question": "...", "answer": "..."}},
  ...
]
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.choices[0].message.content)
```

### From Code Examples
```python
def create_code_explanations(code_files):
    """Generate code explanation pairs."""
    dataset = []

    for file_path in code_files:
        with open(file_path) as f:
            code = f.read()

        # Extract functions
        functions = extract_functions(code)

        for func in functions:
            dataset.append({
                "instruction": f"Explain the {func.name} function",
                "input": func.code,
                "output": func.docstring or generate_doc(func.code)
            })

    return dataset
```

### Synthetic Data
```python
def augment_examples(base_examples, num_variations=3):
    """Create variations of examples."""
    augmented = []

    for example in base_examples:
        # Original
        augmented.append(example)

        # Variations (paraphrase with LLM)
        for _ in range(num_variations):
            variation = paraphrase_with_llm(example)
            augmented.append(variation)

    return augmented
```

## Training Optimization

### Memory Optimization
```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# Gradient accumulation
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Small batch
    gradient_accumulation_steps=16, # Accumulate to 16
    ...
)

# Mixed precision
training_args = TrainingArguments(
    fp16=True,  # or bf16=True
    ...
)
```

### Learning Rate Scheduling
```python
training_args = TrainingArguments(
    learning_rate=2e-4,
    lr_scheduler_type="cosine",  # Cosine decay
    warmup_ratio=0.05,           # 5% warmup
    ...
)
```

### Early Stopping
```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

## Evaluation

### Perplexity
```python
import torch
from torch.nn import CrossEntropyLoss

def calculate_perplexity(model, eval_dataset):
    """Calculate perplexity on eval set."""
    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_dataset:
            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item() * batch['input_ids'].numel()
            total_tokens += batch['input_ids'].numel()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()
```

### Human Evaluation
```python
def create_eval_set(model, prompts):
    """Generate responses for human eval."""
    results = []

    for prompt in prompts:
        base_response = generate(base_model, prompt)
        tuned_response = generate(finetuned_model, prompt)

        results.append({
            "prompt": prompt,
            "base": base_response,
            "finetuned": tuned_response,
            "rating": None  # Fill manually
        })

    return results
```

## Model Selection

### Small Models (CPU-friendly)
```python
# Phi-2 (2.7B) - Good for testing
model_name = "microsoft/phi-2"

# GPT-2 (1.5B) - Very fast
model_name = "gpt2-large"
```

### Medium Models (Consumer GPU)
```python
# Mistral 7B - Excellent performance
model_name = "mistralai/Mistral-7B-v0.1"

# Llama 2 7B - Good balance
model_name = "meta-llama/Llama-2-7b-chat-hf"
```

### Large Models (High-end GPU)
```python
# Llama 2 13B - Better quality
model_name = "meta-llama/Llama-2-13b-chat-hf"

# CodeLlama 13B - Code-focused
model_name = "codellama/CodeLlama-13b-hf"
```

## Preventing Catastrophic Forgetting

### Mix General + Domain Data
```python
# 80% domain-specific, 20% general
domain_data = load_domain_data()     # 800 examples
general_data = load_general_data()   # 200 examples

combined = domain_data + general_data
dataset = Dataset.from_list(combined)
```

### Lower Learning Rate
```python
training_args = TrainingArguments(
    learning_rate=1e-4,  # Lower than default (2e-4)
    ...
)
```

### Fewer Epochs
```python
training_args = TrainingArguments(
    num_train_epochs=3,  # Don't overtrain (3-5 is good)
    ...
)
```

## Multi-GPU Training

### DataParallel
```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    ...
)

# Automatically uses all available GPUs
trainer = Trainer(...)
trainer.train()
```

### DeepSpeed
```python
# deepspeed_config.json
{
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}

# Run with
# deepspeed train.py --deepspeed deepspeed_config.json
```

## Common Issues

### Issue: Out of Memory
**Solutions**:
- Use QLoRA (4-bit)
- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation

### Issue: Poor Quality
**Solutions**:
- Increase dataset quality
- More diverse examples
- Try different learning rates
- Increase LoRA rank

### Issue: Slow Training
**Solutions**:
- Use smaller model for testing
- Reduce max_length
- Use fp16/bf16
- Optimize data loading

### Issue: Model Forgets
**Solutions**:
- Mix general data
- Lower learning rate
- Fewer epochs
- Increase dataset size

## Complete Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import torch

# 1. Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    load_in_8bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 2. Prepare for LoRA
model = prepare_model_for_kbit_training(model)

# 3. LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. Prepare dataset
data = [{"instruction": "...", "output": "..."}]
dataset = Dataset.from_list(data)
# ... tokenize ...

# 5. Train
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# 6. Save
model.save_pretrained("./my-lora")
```

## Resources

- **PEFT Docs**: https://huggingface.co/docs/peft/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **TRL**: https://github.com/huggingface/trl
