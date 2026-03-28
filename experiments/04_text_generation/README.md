# Experiment 04: Game Content Text Generation

GPT-2ベースのゲームコンテンツ自動生成システム

---

## 概要

クエスト、アイテム説明、NPC対話など、ゲームに必要なテキストコンテンツを自動生成します。
テンプレートベースとニューラル生成のハイブリッドアプローチで高品質なコンテンツを作成できます。

### 生成可能なコンテンツ

1. **Quests** - Fetch, Kill, Escort, Explore quests
2. **Item Descriptions** - Weapons, armor, consumables
3. **NPC Dialogue** - Context-aware conversations
4. **Character Names** - Race and gender-specific names
5. **Lore Text** - World-building and backstory

---

## クイックスタート

```bash
pip install -r requirements.txt
python game_content_generator.py
```

### 基本的な使用

```python
from game_content_generator import GameContentGenerator

generator = GameContentGenerator()

# クエスト生成
quest = generator.generate_quest(
    quest_type='fetch',
    difficulty='medium'
)

print(quest['objective'])
print(quest['description'])
```

---

## 使用例

### 1. アイテム説明の一括生成

```python
# 100個のアイテム説明を生成
items = []
for i in range(100):
    item = generator.generate_item_description(
        item_type=random.choice(['sword', 'shield', 'potion']),
        rarity=random.choice(['common', 'rare', 'legendary'])
    )
    items.append(item)

# JSON保存
generator.save_generated_content(items, 'items.json')
```

### 2. カスタムデータでFine-tuning

```python
# ゲーム固有のテキストで訓練
generator.fine_tune(
    train_data_path='my_game_text.txt',
    output_dir='./my_game_model',
    epochs=5
)

# カスタムモデルを使用
custom_gen = GameContentGenerator(
    use_custom_model=True,
    custom_model_path='./my_game_model'
)
```

---

## Fine-tuning用データ準備

```
my_game_text.txt:

Quest: Retrieve the Ancient Sword
Description: A legendary blade...

Quest: Defeat 10 Goblins
Description: The goblin horde...

Item: Flaming Sword
Description: A sword imbued with...
```

---

## パラメータチューニング

```python
# より創造的
text = generator.generate_text(
    prompt,
    temperature=1.0,  # 高い = ランダム
    top_p=0.95
)

# より保守的
text = generator.generate_text(
    prompt,
    temperature=0.5,  # 低い = 決定的
    top_k=40
)
```

---

**Create unlimited game content with AI! 🎮✍️**
