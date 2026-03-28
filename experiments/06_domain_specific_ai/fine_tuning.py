"""
Fine-tuning Script for Domain-Specific AI

This module fine-tunes a language model on game development data using LoRA (Low-Rank Adaptation).
Supports both full training and QLoRA (quantized) for consumer GPUs.
"""

import os
import argparse
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel
    )
    from datasets import Dataset, load_dataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers/PEFT not installed. Install with: pip install transformers peft")


class GameDevFineTuner:
    """ゲーム開発ドメイン向けLLMファインチューニング."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        output_dir: str = "game-dev-lora",
        use_qlora: bool = False
    ):
        """
        初期化.

        Args:
            model_name: ベースモデル名
            output_dir: 出力ディレクトリ
            use_qlora: QLoRA（4bit量子化）を使用するか
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers/PEFT required. Install with: pip install transformers peft")

        self.model_name = model_name
        self.output_dir = output_dir
        self.use_qlora = use_qlora

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[Any] = None

    def load_model(self):
        """ベースモデルとトークナイザーを読み込む."""
        print(f"Loading model: {self.model_name}")

        # トークナイザー
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # モデル読み込み設定
        if self.use_qlora:
            # QLoRA: 4bit量子化
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            print("Loaded model with QLoRA (4-bit)")
        else:
            # 標準: 8bit量子化
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            print("Loaded model with 8-bit quantization")

        # 勾配チェックポイント有効化（メモリ節約）
        self.model.config.use_cache = False
        self.model = prepare_model_for_kbit_training(self.model)

    def apply_lora(self, rank: int = 16, alpha: int = 32):
        """
        LoRAを適用.

        Args:
            rank: LoRAランク（大きいほど容量大、メモリ使用多）
            alpha: スケーリング係数
        """
        print(f"Applying LoRA (rank={rank}, alpha={alpha})")

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],  # Llama-2用
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(self, data_path: str) -> Dataset:
        """
        トレーニングデータセットを準備.

        Args:
            data_path: データファイルパス（JSON）

        Returns:
            Hugging Face Dataset
        """
        print(f"Loading dataset from {data_path}")

        # JSONファイル読み込み
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # データセット作成
        dataset = Dataset.from_dict({
            "instruction": [item["instruction"] for item in data],
            "input": [item.get("input", "") for item in data],
            "output": [item["output"] for item in data]
        })

        # フォーマット関数
        def format_instruction(example):
            """命令形式でフォーマット."""
            prompt = f"""Below is an instruction for game development. Write a response that appropriately answers the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

            return {"text": prompt}

        # フォーマット適用
        dataset = dataset.map(format_instruction)

        # トークン化
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset.column_names,
            batched=True
        )

        print(f"Prepared {len(tokenized_dataset)} examples")
        return tokenized_dataset

    def train(
        self,
        train_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4
    ):
        """
        トレーニング実行.

        Args:
            train_dataset: トレーニングデータセット
            num_epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            gradient_accumulation_steps: 勾配累積ステップ
        """
        print(f"Starting training for {num_epochs} epochs")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=True,  # Mixed precision
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=3,
            optim="adamw_torch",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            report_to="none"  # Disable wandb/tensorboard
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )

        # トレーニング開始
        trainer.train()

        print("Training complete!")

    def save_model(self):
        """LoRAアダプターを保存."""
        print(f"Saving LoRA adapters to {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("Model saved!")

    def generate_text(self, prompt: str, max_length: int = 200) -> str:
        """
        テキスト生成.

        Args:
            prompt: 入力プロンプト
            max_length: 最大長

        Returns:
            生成されたテキスト
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated


def create_sample_training_data(output_path: str):
    """
    サンプルトレーニングデータを作成.

    Args:
        output_path: 出力ファイルパス
    """
    training_data = [
        {
            "instruction": "Explain the combat system in our game",
            "input": "",
            "output": "The combat system uses a turn-based approach where each player selects an action (Attack, Defend, or Special). Damage is calculated using the formula: damage = attack_power * (1 - defense_modifier) * critical_multiplier. There's a 10% chance of a critical hit that deals 2x damage."
        },
        {
            "instruction": "How do I implement inventory sorting?",
            "input": "",
            "output": "To implement inventory sorting, use the InventoryManager.SortItems() method. This supports sorting by: ItemType, Rarity, Name, or Quantity. Example code:\n\n```csharp\nInventoryManager.Instance.SortItems(SortType.Rarity);\n```\n\nItems are sorted in descending order by default."
        },
        {
            "instruction": "What's the player movement speed formula?",
            "input": "",
            "output": "Player speed is calculated as: final_speed = base_speed * (1 + agility/100) * terrain_modifier * status_modifier\n\nWhere:\n- base_speed: Default 5.0 units/second\n- agility: Each point increases speed by 1%\n- terrain_modifier: Grass=1.0x, Road=1.2x, Water=0.5x\n- status_modifier: Haste=1.5x, Slow=0.5x"
        },
        {
            "instruction": "How do status effects work?",
            "input": "",
            "output": "Status effects are temporary modifiers applied during combat:\n\n- **Poison**: Deals 5% max HP damage per turn for 3 turns\n- **Stun**: Skips the next turn\n- **Buff**: Increases attack by 25% for 2 turns\n\nStatus effects are checked at the start of each turn and decremented after applying their effect."
        },
        {
            "instruction": "Explain the inventory weight system",
            "input": "",
            "output": "Each item has a weight value. Players have a maximum carry capacity. If total inventory weight exceeds this capacity, the player becomes encumbered and movement speed is reduced by 25%. You can check current weight with:\n\n```csharp\nfloat currentWeight = InventoryManager.Instance.GetTotalWeight();\nfloat maxCapacity = player.MaxCarryCapacity;\nbool isEncumbered = currentWeight > maxCapacity;\n```"
        },
        {
            "instruction": "How do I add a new item to inventory?",
            "input": "",
            "output": "Use the InventoryManager.AddItem() method:\n\n```csharp\nItem newItem = ItemDatabase.GetItem(itemID);\nInventoryManager.Instance.AddItem(newItem);\n```\n\nThe system automatically handles stacking for stackable items (up to MaxStackSize, default 99). If inventory is full, the method returns false."
        },
        {
            "instruction": "What are the different terrain modifiers?",
            "input": "",
            "output": "Terrain modifiers affect player movement speed:\n\n- **Grass**: 1.0x (normal speed)\n- **Road**: 1.2x (20% faster)\n- **Water**: 0.5x (50% slower)\n- **Mountain**: 0.7x (30% slower)\n\nThe current terrain is detected using raycasting from the player's position."
        },
        {
            "instruction": "How does the critical hit system work?",
            "input": "",
            "output": "Critical hits have a 10% base chance to occur during an attack. When a critical hit happens, damage is multiplied by 2x. Implementation:\n\n```csharp\nfloat critChance = 0.1f; // 10%\nbool isCrit = Random.value < critChance;\nfloat critMultiplier = isCrit ? 2f : 1f;\nfloat finalDamage = baseDamage * critMultiplier;\n```\n\nCertain items and abilities can increase critical hit chance."
        }
    ]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    print(f"Created sample training data at {output_path}")


def load_and_generate(lora_path: str, prompt: str):
    """
    保存されたLoRAモデルを読み込んで生成.

    Args:
        lora_path: LoRAアダプターのパス
        prompt: プロンプト
    """
    print(f"Loading LoRA model from {lora_path}")

    # ベースモデル読み込み
    base_model_name = "meta-llama/Llama-2-7b-chat-hf"  # 元のモデル名
    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRAアダプター読み込み
    model = PeftModel.from_pretrained(base_model, lora_path)

    # 生成
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{result}")


def main():
    """メイン関数."""
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Game Development")
    parser.add_argument("--data", type=str, help="Training data JSON file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Base model")
    parser.add_argument("--output", type=str, default="game-dev-lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--qlora", action="store_true", help="Use QLoRA (4-bit)")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--create-sample", type=str, help="Create sample training data")
    parser.add_argument("--generate", type=str, help="Generate from trained model")
    parser.add_argument("--prompt", type=str, help="Prompt for generation")

    args = parser.parse_args()

    # サンプルデータ作成
    if args.create_sample:
        create_sample_training_data(args.create_sample)
        return

    # 生成モード
    if args.generate:
        if not args.prompt:
            print("Error: --prompt required for generation")
            return
        load_and_generate(args.generate, args.prompt)
        return

    # トレーニングモード
    if not args.data:
        print("Error: --data required for training")
        return

    # ファインチューナー初期化
    tuner = GameDevFineTuner(
        model_name=args.model,
        output_dir=args.output,
        use_qlora=args.qlora
    )

    # モデル読み込み
    tuner.load_model()

    # LoRA適用
    tuner.apply_lora(rank=args.rank)

    # データセット準備
    dataset = tuner.prepare_dataset(args.data)

    # トレーニング
    tuner.train(
        train_dataset=dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # 保存
    tuner.save_model()

    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()
