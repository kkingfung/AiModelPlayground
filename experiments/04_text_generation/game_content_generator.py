"""
Game Content Text Generation

ゲームコンテンツ（クエスト、アイテム説明、対話など）の自動生成システム。
GPT-2ベースのモデルをFine-tuningしてゲーム固有のテキストを生成します。

特徴:
    - Quest generation (クエスト生成)
    - Item descriptions (アイテム説明)
    - NPC dialogue (NPC対話)
    - Character names (キャラクター名)
    - Lore text (背景設定)
    - Template-based + Neural hybrid

使い方:
    from game_content_generator import GameContentGenerator

    generator = GameContentGenerator()

    # クエスト生成
    quest = generator.generate_quest(
        quest_type='fetch',
        difficulty='medium'
    )

    # アイテム説明生成
    description = generator.generate_item_description(
        item_type='sword',
        rarity='legendary'
    )

参考:
    - GPT-2: https://openai.com/blog/better-language-models/
    - Fine-tuning for domain-specific text
    - Controlled text generation
"""

import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from typing import Dict, List, Optional, Tuple
import random
import json
import re
from pathlib import Path


class TemplateEngine:
    """
    テンプレートベースの生成エンジン.

    ニューラルモデルと組み合わせて使用します.
    """

    def __init__(self):
        # クエストテンプレート
        self.quest_templates = {
            'fetch': [
                "Retrieve {item} from {location}",
                "Collect {count} {item} scattered across {location}",
                "Find the legendary {item} hidden in {location}"
            ],
            'kill': [
                "Defeat {count} {enemy} in {location}",
                "Hunt down the {enemy} leader in {location}",
                "Eliminate all {enemy} threatening {location}"
            ],
            'escort': [
                "Escort {npc} safely to {location}",
                "Protect {npc} from {enemy} while traveling to {location}",
                "Guide {npc} through dangerous {location}"
            ],
            'explore': [
                "Explore the uncharted {location}",
                "Discover the secrets of {location}",
                "Map the entire {location} region"
            ]
        }

        # アイテム修飾語
        self.item_prefixes = {
            'common': ['Simple', 'Basic', 'Plain', 'Worn'],
            'uncommon': ['Quality', 'Fine', 'Sturdy', 'Reliable'],
            'rare': ['Superior', 'Exceptional', 'Masterwork', 'Enhanced'],
            'epic': ['Legendary', 'Mythical', 'Ancient', 'Powerful'],
            'legendary': ['Godlike', 'Divine', 'Eternal', 'Transcendent']
        }

        # ロケーション
        self.locations = [
            'Ancient Forest', 'Dark Cave', 'Misty Mountains',
            'Abandoned Temple', 'Frozen Wasteland', 'Haunted Castle',
            'Crystal Cavern', 'Volcanic Rift', 'Sunken City'
        ]

        # 敵
        self.enemies = [
            'Goblin', 'Orc', 'Skeleton', 'Dragon',
            'Demon', 'Zombie', 'Bandit', 'Beast'
        ]

        # NPC名
        self.npc_names = [
            'Aldric', 'Beatrice', 'Cedric', 'Diana',
            'Elara', 'Finn', 'Gwendolyn', 'Henrik'
        ]

    def fill_template(self, template: str, **kwargs) -> str:
        """テンプレートを埋める."""
        # ランダム選択のデフォルト値
        defaults = {
            'item': random.choice(['Sword', 'Shield', 'Potion', 'Scroll', 'Gem']),
            'location': random.choice(self.locations),
            'enemy': random.choice(self.enemies),
            'npc': random.choice(self.npc_names),
            'count': random.randint(3, 10)
        }

        # ユーザー指定で上書き
        defaults.update(kwargs)

        return template.format(**defaults)


class GameContentGenerator:
    """
    ゲームコンテンツ生成システム.

    GPT-2ベースのニューラルモデルとテンプレートエンジンを組み合わせて
    高品質なゲームテキストを生成します.
    """

    def __init__(
        self,
        model_name: str = 'gpt2',
        use_custom_model: bool = False,
        custom_model_path: Optional[str] = None,
        device: str = 'auto'
    ):
        """
        Args:
            model_name: ベースモデル名
            use_custom_model: カスタムモデルを使用
            custom_model_path: カスタムモデルのパス
            device: デバイス
        """
        # デバイス設定
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Loading model: {model_name}...")

        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        if use_custom_model and custom_model_path:
            self.model = GPT2LMHeadModel.from_pretrained(custom_model_path)
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        # テンプレートエンジン
        self.template_engine = TemplateEngine()

        print(f"✓ GameContentGenerator initialized on {self.device}")

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        テキストを生成.

        Args:
            prompt: プロンプト
            max_length: 最大長
            temperature: 温度（高いほどランダム）
            top_k: Top-Kサンプリング
            top_p: Top-Pサンプリング
            num_return_sequences: 生成数

        Returns:
            generated_texts: 生成されたテキストのリスト
        """
        # Encode
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated_texts = []
        for sequence in output:
            text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            # プロンプトを除去
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            generated_texts.append(text)

        return generated_texts

    def generate_quest(
        self,
        quest_type: str = 'fetch',
        difficulty: str = 'medium',
        use_template: bool = True,
        **kwargs
    ) -> Dict[str, str]:
        """
        クエストを生成.

        Args:
            quest_type: クエストタイプ（fetch, kill, escort, explore）
            difficulty: 難易度
            use_template: テンプレートを使用
            **kwargs: テンプレートパラメータ

        Returns:
            quest: クエスト情報
        """
        if use_template:
            # テンプレートから目標を生成
            templates = self.template_engine.quest_templates.get(quest_type, [])
            if not templates:
                templates = self.template_engine.quest_templates['fetch']

            template = random.choice(templates)
            objective = self.template_engine.fill_template(template, **kwargs)

            # ニューラルモデルで説明を生成
            prompt = f"Quest Objective: {objective}\nDescription:"
            descriptions = self.generate_text(
                prompt,
                max_length=150,
                temperature=0.7,
                num_return_sequences=1
            )
            description = descriptions[0]

        else:
            # 完全にニューラル生成
            prompt = f"Generate a {difficulty} difficulty {quest_type} quest:\nQuest:"
            quests = self.generate_text(
                prompt,
                max_length=200,
                temperature=0.8,
                num_return_sequences=1
            )
            full_text = quests[0]

            # パース（簡易）
            lines = full_text.split('\n')
            objective = lines[0] if lines else full_text
            description = '\n'.join(lines[1:]) if len(lines) > 1 else ""

        return {
            'type': quest_type,
            'difficulty': difficulty,
            'objective': objective,
            'description': description
        }

    def generate_item_description(
        self,
        item_type: str = 'sword',
        rarity: str = 'rare',
        use_template: bool = True
    ) -> Dict[str, str]:
        """
        アイテム説明を生成.

        Args:
            item_type: アイテムタイプ
            rarity: レアリティ
            use_template: テンプレートを使用

        Returns:
            item: アイテム情報
        """
        if use_template:
            # テンプレートから名前を生成
            prefixes = self.template_engine.item_prefixes.get(rarity, ['Fine'])
            prefix = random.choice(prefixes)
            item_name = f"{prefix} {item_type.capitalize()}"

            # ニューラルモデルで説明を生成
            prompt = f"Item: {item_name}\nRarity: {rarity}\nDescription:"
            descriptions = self.generate_text(
                prompt,
                max_length=100,
                temperature=0.7,
                num_return_sequences=1
            )
            description = descriptions[0]

        else:
            # 完全にニューラル生成
            prompt = f"Generate a {rarity} {item_type} description:\nItem:"
            items = self.generate_text(
                prompt,
                max_length=150,
                temperature=0.8,
                num_return_sequences=1
            )
            full_text = items[0]

            # パース
            lines = full_text.split('\n')
            item_name = lines[0] if lines else f"{rarity.capitalize()} {item_type.capitalize()}"
            description = '\n'.join(lines[1:]) if len(lines) > 1 else full_text

        return {
            'name': item_name,
            'type': item_type,
            'rarity': rarity,
            'description': description
        }

    def generate_dialogue(
        self,
        npc_name: str,
        context: str = "",
        mood: str = "neutral",
        max_lines: int = 3
    ) -> List[str]:
        """
        NPC対話を生成.

        Args:
            npc_name: NPC名
            context: 文脈
            mood: 気分（happy, sad, angry, neutral）
            max_lines: 最大行数

        Returns:
            dialogue_lines: 対話行のリスト
        """
        prompt = f"{npc_name} ({mood})"
        if context:
            prompt += f" talking about {context}"
        prompt += ":\n"

        dialogues = self.generate_text(
            prompt,
            max_length=150,
            temperature=0.9,
            num_return_sequences=1
        )

        # 行に分割
        lines = dialogues[0].split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        lines = lines[:max_lines]

        return lines

    def generate_character_name(
        self,
        race: str = 'human',
        gender: str = 'any',
        num_names: int = 5
    ) -> List[str]:
        """
        キャラクター名を生成.

        Args:
            race: 種族
            gender: 性別
            num_names: 生成数

        Returns:
            names: 名前のリスト
        """
        prompt = f"Generate {race} {gender} character names:\n"

        names_text = self.generate_text(
            prompt,
            max_length=100,
            temperature=0.9,
            num_return_sequences=1
        )

        # 名前を抽出
        text = names_text[0]
        names = []

        # パターンマッチ
        matches = re.findall(r'\b[A-Z][a-z]+\b', text)
        names.extend(matches)

        # 重複除去
        names = list(dict.fromkeys(names))

        return names[:num_names]

    def generate_lore(
        self,
        topic: str,
        style: str = 'epic',
        length: int = 200
    ) -> str:
        """
        背景設定テキストを生成.

        Args:
            topic: トピック
            style: スタイル（epic, mysterious, dark, light）
            length: 長さ

        Returns:
            lore: 背景設定テキスト
        """
        prompt = f"In a {style} fantasy world, the legend of {topic} tells:"

        lore_texts = self.generate_text(
            prompt,
            max_length=length,
            temperature=0.8,
            num_return_sequences=1
        )

        return lore_texts[0]

    def fine_tune(
        self,
        train_data_path: str,
        output_dir: str = './fine_tuned_model',
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-5
    ):
        """
        カスタムデータでFine-tuning.

        Args:
            train_data_path: 訓練データのパス（.txt）
            output_dir: 出力ディレクトリ
            epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
        """
        print(f"\nFine-tuning on {train_data_path}...")

        # データセット作成
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=train_data_path,
            block_size=128
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # 訓練設定
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=2,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=100,
            report_to=[]
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset
        )

        # 訓練
        trainer.train()

        # 保存
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"✓ Model fine-tuned and saved to {output_dir}")

    def save_generated_content(
        self,
        content: Dict,
        filepath: str
    ):
        """生成コンテンツを保存."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)

        print(f"✓ Content saved to {filepath}")


# ============================================================================
# デモ
# ============================================================================

if __name__ == "__main__":
    print("Game Content Text Generation Demo")
    print("=" * 60)

    # ジェネレーター作成
    generator = GameContentGenerator(use_custom_model=False)

    print("\n" + "=" * 60)
    print("1. Quest Generation")
    print("=" * 60)

    quest_types = ['fetch', 'kill', 'escort', 'explore']
    for quest_type in quest_types:
        quest = generator.generate_quest(
            quest_type=quest_type,
            difficulty='medium',
            use_template=True
        )

        print(f"\n[{quest_type.upper()}] Quest:")
        print(f"  Objective: {quest['objective']}")
        print(f"  Description: {quest['description'][:100]}...")

    print("\n" + "=" * 60)
    print("2. Item Description Generation")
    print("=" * 60)

    item_types = ['sword', 'shield', 'potion', 'ring']
    rarities = ['common', 'rare', 'legendary']

    for item_type in item_types[:2]:
        for rarity in rarities[:2]:
            item = generator.generate_item_description(
                item_type=item_type,
                rarity=rarity,
                use_template=True
            )

            print(f"\n{item['name']} ({item['rarity']})")
            print(f"  {item['description'][:80]}...")

    print("\n" + "=" * 60)
    print("3. NPC Dialogue Generation")
    print("=" * 60)

    npc_contexts = [
        ("Guard", "the recent monster attacks", "worried"),
        ("Merchant", "rare items for sale", "friendly"),
        ("Wizard", "ancient prophecy", "mysterious")
    ]

    for npc_name, context, mood in npc_contexts:
        print(f"\n{npc_name} ({mood}):")
        dialogue = generator.generate_dialogue(
            npc_name=npc_name,
            context=context,
            mood=mood,
            max_lines=2
        )

        for line in dialogue:
            print(f"  \"{line}\"")

    print("\n" + "=" * 60)
    print("4. Character Name Generation")
    print("=" * 60)

    races = ['elf', 'dwarf', 'orc']
    for race in races:
        names = generator.generate_character_name(
            race=race,
            gender='any',
            num_names=5
        )
        print(f"\n{race.capitalize()} names: {', '.join(names)}")

    print("\n" + "=" * 60)
    print("5. Lore Generation")
    print("=" * 60)

    topics = ['the Dragon King', 'the Lost City', 'the Sacred Sword']
    for topic in topics:
        lore = generator.generate_lore(
            topic=topic,
            style='epic',
            length=150
        )
        print(f"\n{topic}:")
        print(f"  {lore[:120]}...")

    print("\n" + "=" * 60)
    print("✓ Demo completed!")
    print("\nNext steps:")
    print("  1. Fine-tune on your game-specific text data")
    print("  2. Integrate with game editor")
    print("  3. Add more content types (skills, achievements, etc.)")
