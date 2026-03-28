"""
Game Text Generator

ゲーム用テキスト（アイテム説明、クエスト、会話など）を自動生成.
GPT-2/LLaMAなどのLLMをファインチューニングして、ゲーム固有のスタイルで生成.

使い方:
    # 推論（事前学習済みモデル）
    python text_generator.py \
        --prompt "The ancient sword glows with" \
        --model gpt2 \
        --max-length 100

    # ファインチューニング
    python text_generator.py \
        --train game_texts.json \
        --model gpt2 \
        --epochs 3

    # バッチ生成
    python text_generator.py \
        --batch-generate prompts.txt \
        --model fine_tuned/checkpoint-1000
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from tqdm import tqdm


class GameTextGenerator:
    """
    ゲームテキスト生成器.

    GPT-2やLLaMAなどのLLMを使用して、
    ゲーム用のテキストを生成します.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: ベースモデル名（gpt2, gpt2-medium, meta-llama/Llama-2-7b-hf など）
            device: デバイス（cuda/cpu）
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

        # Padding token設定（GPT-2にはデフォルトで無い）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 1.2
    ) -> List[str]:
        """
        プロンプトからテキスト生成.

        Args:
            prompt: 生成の開始テキスト
            max_length: 最大トークン数
            num_return_sequences: 生成する候補数
            temperature: 温度（高いほど多様、低いほど確実）
            top_k: Top-Kサンプリング
            top_p: Nucleus sampling (top-p)
            do_sample: サンプリング有効化
            repetition_penalty: 繰り返しペナルティ

        Returns:
            生成されたテキストのリスト
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def generate_item_description(
        self,
        item_name: str,
        item_type: str = "weapon",
        rarity: str = "rare"
    ) -> str:
        """
        アイテム説明文を生成.

        Args:
            item_name: アイテム名
            item_type: アイテムタイプ
            rarity: レアリティ

        Returns:
            生成された説明文
        """
        prompt = f"Item: {item_name}\nType: {item_type}\nRarity: {rarity}\nDescription: "

        results = self.generate(
            prompt,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7
        )

        # プロンプト部分を除去
        description = results[0].replace(prompt, "").strip()
        return description

    def generate_quest_text(
        self,
        quest_type: str = "fetch",
        location: str = "forest",
        reward: str = "gold"
    ) -> Dict[str, str]:
        """
        クエストテキスト生成.

        Args:
            quest_type: クエストタイプ
            location: 場所
            reward: 報酬

        Returns:
            クエストタイトルと説明文
        """
        title_prompt = f"Quest Title ({quest_type}, {location}): "
        title = self.generate(title_prompt, max_length=50, temperature=0.7)[0]
        title = title.replace(title_prompt, "").split("\n")[0].strip()

        desc_prompt = f"Quest: {title}\nObjective: "
        description = self.generate(desc_prompt, max_length=200, temperature=0.7)[0]
        description = description.replace(desc_prompt, "").strip()

        return {
            "title": title,
            "description": description,
            "type": quest_type,
            "location": location,
            "reward": reward
        }

    def generate_dialogue(
        self,
        character: str,
        context: str = "",
        num_lines: int = 3
    ) -> List[str]:
        """
        NPC会話を生成.

        Args:
            character: キャラクター名
            context: 文脈
            num_lines: 生成する行数

        Returns:
            会話のリスト
        """
        prompt = f"Character: {character}\nContext: {context}\nDialogue:\n"

        result = self.generate(
            prompt,
            max_length=200,
            num_return_sequences=1,
            temperature=0.8
        )[0]

        # 行に分割
        lines = result.replace(prompt, "").strip().split("\n")
        return [line.strip() for line in lines if line.strip()][:num_lines]

    def fine_tune(
        self,
        train_data: List[str],
        output_dir: str = "fine_tuned",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 100,
        save_steps: int = 500
    ):
        """
        ゲームテキストでファインチューニング.

        Args:
            train_data: 学習用テキストのリスト
            output_dir: モデル保存ディレクトリ
            epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
            warmup_steps: ウォームアップステップ
            save_steps: 保存間隔
        """
        print(f"Fine-tuning on {len(train_data)} examples")

        # データセット準備
        dataset = Dataset.from_dict({"text": train_data})

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM (not masked LM)
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            save_total_limit=2,
            logging_steps=100,
            prediction_loss_only=True,
            fp16=torch.cuda.is_available(),
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Save final model
        trainer.save_model(output_dir + "/final")
        self.tokenizer.save_pretrained(output_dir + "/final")

        print(f"Model saved to {output_dir}/final")

    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 100,
        batch_size: int = 8
    ) -> List[str]:
        """
        複数プロンプトから一括生成.

        Args:
            prompts: プロンプトのリスト
            max_length: 最大長
            batch_size: バッチサイズ

        Returns:
            生成されたテキストのリスト
        """
        all_results = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch_prompts = prompts[i:i+batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decode
            for output in outputs:
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                all_results.append(text)

        return all_results


def load_training_data(file_path: str) -> List[str]:
    """
    学習データをJSONファイルから読み込み.

    フォーマット例:
    [
        "Item: Sword of Light. A legendary blade...",
        "Quest: Find the lost artifact...",
        ...
    ]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "texts" in data:
        return data["texts"]
    else:
        raise ValueError("Invalid JSON format. Expected list of strings or {'texts': [...]}")


def main():
    parser = argparse.ArgumentParser(description="Game Text Generator")
    parser.add_argument("--prompt", type=str, help="Prompt for generation")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--train", type=str, help="Training data JSON file")
    parser.add_argument("--batch-generate", type=str, help="File with prompts (one per line)")
    parser.add_argument("--max-length", type=int, default=100, help="Max generation length")
    parser.add_argument("--num-sequences", type=int, default=1, help="Number of sequences")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--output", type=str, default="generated.txt", help="Output file")

    args = parser.parse_args()

    generator = GameTextGenerator(model_name=args.model)

    # 単一生成
    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print("-" * 50)

        results = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            num_return_sequences=args.num_sequences,
            temperature=args.temperature
        )

        for i, text in enumerate(results, 1):
            print(f"\nGeneration {i}:")
            print(text)

    # ファインチューニング
    elif args.train:
        print(f"Loading training data from {args.train}")
        train_data = load_training_data(args.train)

        generator.fine_tune(
            train_data=train_data,
            epochs=args.epochs
        )

    # バッチ生成
    elif args.batch_generate:
        print(f"Loading prompts from {args.batch_generate}")

        with open(args.batch_generate, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        results = generator.batch_generate(prompts, max_length=args.max_length)

        # Save results
        with open(args.output, "w", encoding="utf-8") as f:
            for prompt, result in zip(prompts, results):
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Generated: {result}\n")
                f.write("-" * 50 + "\n")

        print(f"Results saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
