# NLP Guide for Game Development

## 概要

ゲーム開発で実用的なNLP技術を学びます:
- テキスト生成（Text Generation）
- 感情分析（Sentiment Analysis）
- 固有名詞抽出（Named Entity Recognition）
- 会話システム（Dialogue Systems）

---

## 1. テキスト生成（Text Generator）

### 用途
- アイテム説明文自動生成
- クエストテキスト作成
- NPC会話生成
- ゲーム内テキストのローカライゼーション

### 使い方

#### 単純な生成
```bash
python text_generator.py \
    --prompt "The legendary sword known as" \
    --model gpt2 \
    --max-length 100 \
    --temperature 0.8
```

#### アイテム説明文生成
```python
from text_generator import GameTextGenerator

generator = GameTextGenerator(model_name="gpt2")

description = generator.generate_item_description(
    item_name="Flameblade",
    item_type="weapon",
    rarity="legendary"
)

print(description)
# Output: "A legendary sword forged in dragon fire,
#          capable of igniting enemies with every strike..."
```

#### クエストテキスト生成
```python
quest = generator.generate_quest_text(
    quest_type="rescue",
    location="ancient ruins",
    reward="magic artifact"
)

print(f"Title: {quest['title']}")
print(f"Description: {quest['description']}")
```

### ファインチューニング

#### 学習データ準備 (game_texts.json)
```json
[
  "Item: Sword of Light. A legendary blade that shines with divine power...",
  "Quest: The Lost Artifact. Find the ancient relic hidden in the cursed tomb...",
  "Dialogue: 'Greetings, traveler. I have a quest for you...'",
  ...
]
```

#### 学習実行
```bash
python text_generator.py \
    --train data/game_texts.json \
    --model gpt2 \
    --epochs 3
```

### モデル選択

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| GPT-2 | 500MB | 速い | 中 | プロトタイプ・軽量生成 |
| GPT-2 Medium | 1.5GB | 中 | 高 | バランス型 |
| LLaMA-2-7B | 13GB | 遅い | 最高 | 高品質生成 |

### パラメータチューニング

#### Temperature（多様性）
```python
# 確実（同じような出力）
result = generator.generate(prompt, temperature=0.3)

# バランス
result = generator.generate(prompt, temperature=0.7)

# クリエイティブ（多様な出力）
result = generator.generate(prompt, temperature=1.2)
```

#### Top-K / Top-P（語彙選択）
```python
# 保守的（一般的な単語のみ）
result = generator.generate(prompt, top_k=10, top_p=0.8)

# バランス（デフォルト）
result = generator.generate(prompt, top_k=50, top_p=0.95)

# 冒険的（珍しい単語も）
result = generator.generate(prompt, top_k=100, top_p=0.99)
```

---

## 2. 感情分析（Sentiment Analyzer）

### 用途
- プレイヤーレビュー分析
- フィードバック自動分類
- コミュニティ感情モニタリング
- 問題点の早期発見

### 使い方

#### 単一レビュー分析
```bash
python sentiment_analyzer.py \
    --text "This game is amazing! Best RPG I've played in years."
```

出力:
```
Sentiment: POSITIVE
Confidence: 99.8%
Scores: Positive=99.8%, Negative=0.2%
```

#### バッチ分析
```bash
python sentiment_analyzer.py \
    --analyze data/player_reviews.json \
    --output sentiment_report.txt
```

#### レビューデータ準備 (player_reviews.json)
```json
[
  {"text": "Great game! Love the graphics and gameplay.", "rating": 5},
  {"text": "Terrible bugs, unplayable.", "rating": 1},
  {"text": "Good concept but needs improvement.", "rating": 3},
  ...
]
```

### プログラムからの使用
```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

# レビュー分析
reviews = load_reviews("player_feedback.json")
analysis = analyzer.analyze_reviews(reviews)

print(f"Positive ratio: {analysis['statistics']['positive_ratio']:.1f}%")
print(f"Top complaints: {analysis['top_negative']}")

# アクション可能なインサイト
if analysis['statistics']['negative_ratio'] > 30:
    print("⚠️ High negative sentiment detected!")

    # ネガティブトピック抽出
    texts = [r["text"] for r in reviews]
    issues = analyzer.extract_topics(texts, sentiment_filter="NEGATIVE")
    print(f"Main issues: {issues[:5]}")
```

### レポート例
```
==========================================================
SENTIMENT ANALYSIS REPORT
==========================================================

Total Reviews Analyzed: 1,247

Sentiment Distribution:
  Positive: 856 (68.6%)
  Negative: 298 (23.9%)
  Neutral:  93 (7.5%)

Average Confidence: 94.3%

----------------------------------------------------------
TOP NEGATIVE REVIEWS (Areas for Improvement):
----------------------------------------------------------

1. [98.5% confident]
   "The game crashes constantly on level 3. Can't progress..."

2. [97.2% confident]
   "Controls are terrible, especially for combat..."

----------------------------------------------------------
TOPIC ANALYSIS:
----------------------------------------------------------

Most Mentioned (Negative Reviews):
  crash, bug, control, combat, save, level, enemy, balance
```

---

## 3. 実用例

### Example 1: 自動コンテンツ生成パイプライン

```python
from text_generator import GameTextGenerator

generator = GameTextGenerator()

# 100個のアイテム説明文を自動生成
items = [
    {"name": "Fire Sword", "type": "weapon", "rarity": "rare"},
    {"name": "Ice Shield", "type": "armor", "rarity": "epic"},
    # ... 98 more items
]

for item in items:
    description = generator.generate_item_description(**item)

    # ゲームデータベースに保存
    save_to_db(item["name"], description)

    print(f"Generated: {item['name']}")
```

### Example 2: リアルタイムフィードバックモニタリング

```python
from sentiment_analyzer import SentimentAnalyzer
import time

analyzer = SentimentAnalyzer()

def monitor_reviews():
    """新着レビューを監視してアラート."""

    while True:
        # 新着レビュー取得（例: Steam API）
        new_reviews = fetch_new_reviews()

        if new_reviews:
            analysis = analyzer.analyze_batch([r["text"] for r in new_reviews])

            negative_count = sum(1 for a in analysis if a["sentiment"] == "NEGATIVE")

            if negative_count / len(analysis) > 0.5:
                # 50%以上がネガティブならアラート
                send_alert_to_team(f"⚠️ High negative sentiment: {negative_count}/{len(analysis)}")

        time.sleep(300)  # 5分ごとにチェック
```

### Example 3: 多言語対応

```python
# 英語で生成
en_generator = GameTextGenerator(model_name="gpt2")
en_text = en_generator.generate("The hero embarks on")

# 日本語で生成
ja_generator = GameTextGenerator(model_name="rinna/japanese-gpt2-medium")
ja_text = ja_generator.generate("勇者は冒険に")

# 翻訳品質チェック
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
en_sentiment = analyzer.analyze(en_text)["sentiment"]
ja_sentiment = analyzer.analyze(ja_text)["sentiment"]

if en_sentiment != ja_sentiment:
    print("⚠️ Translation may have altered sentiment!")
```

---

## 4. パフォーマンス最適化

### メモリ削減

#### 8-bit量子化
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    load_in_8bit=True,  # 8-bit量子化
    device_map="auto"
)

# メモリ使用量: 500MB → 125MB
```

#### Gradient Checkpointing（学習時）
```python
model.gradient_checkpointing_enable()

# メモリ使用量削減、速度は若干低下
```

### 推論速度向上

#### ONNX変換（詳細は3_optimization/参照）
```bash
# PyTorch → ONNX
python -m transformers.onnx \
    --model=gpt2 \
    --feature=causal-lm \
    onnx/gpt2/

# 推論: 2-3x高速化
```

#### バッチ処理
```python
# 遅い（1個ずつ）
for text in texts:
    result = analyzer.analyze(text)

# 速い（バッチ）
results = analyzer.analyze_batch(texts, batch_size=32)
```

---

## 5. トラブルシューティング

### 生成品質が低い
**症状**: 無意味なテキスト、繰り返しが多い

**対策**:
- `repetition_penalty`を上げる（1.2 → 1.5）
- `temperature`を下げる（1.0 → 0.7）
- より大きなモデルを使用（gpt2 → gpt2-medium）
- ゲーム固有データでファインチューニング

### メモリ不足エラー
**症状**: CUDA out of memory

**対策**:
- バッチサイズを減らす
- `max_length`を短くする
- 量子化を使用（8-bit/4-bit）
- Gradient checkpointingを有効化

### 感情分析の精度が低い
**対策**:
- ゲーム固有データでファインチューニング
- より大きなモデルを使用（distilbert → roberta-large）
- 複数モデルのアンサンブル
- ドメイン固有の辞書を追加

---

## 6. 次のステップ

NLPをマスターしたら:
1. **Optimization** (3_optimization/) - モデル最適化とONNX変換
2. **Integration** (4_integration/) - ゲームへの組み込み
3. **Advanced NLP** - ファインチューニング、RAGシステム構築

---

## リソース

- **Hugging Face Course**: https://huggingface.co/course
- **Transformers Docs**: https://huggingface.co/docs/transformers
- **Papers with Code (NLP)**: https://paperswithcode.com/area/natural-language-processing

---

**Happy NLP! 🎮📝**
