# Experiment 07 - Quick Start Guide

## 5分で始める実用的なゲームAI

### Step 1: 環境セットアップ（2分）

```bash
cd experiments/07_practical_game_ai
pip install -r requirements.txt
```

---

## 🖼️ Computer Vision - 画像分類

### アセット分類器を5分で作成

```bash
# 1. サンプルデータ準備（フォルダ構造）
mkdir -p data/assets/{character,ui,background,item,weapon}

# 2. 各フォルダに画像を配置
# character/*.png, ui/*.png, etc.

# 3. 学習（10エポック、約5分）
python 1_computer_vision/image_classifier.py \
    --train data/assets \
    --epochs 10 \
    --model-name mobilenet_v3_small  # 軽量モデル

# 4. 推論
python 1_computer_vision/image_classifier.py \
    --predict new_asset.png \
    --model checkpoints/best.pth
```

**出力例**:
```
Top class: character
Confidence: 95.3%

Top 3 predictions:
  1. character: 95.30%
  2. item: 3.20%
  3. weapon: 1.50%
```

---

## 📝 NLP - テキスト生成

### ゲームテキストを瞬時に生成

```bash
# アイテム説明文生成
python 2_nlp/text_generator.py \
    --prompt "The Sword of Flames is a legendary weapon that" \
    --model gpt2 \
    --max-length 100
```

**Pythonから使用**:
```python
from text_generator import GameTextGenerator

gen = GameTextGenerator()

# アイテム説明
desc = gen.generate_item_description(
    item_name="Flameblade",
    item_type="weapon",
    rarity="legendary"
)
print(desc)
# "A legendary sword forged in dragon fire, capable of igniting..."

# クエストテキスト
quest = gen.generate_quest_text(
    quest_type="rescue",
    location="dark forest"
)
print(quest["title"])
print(quest["description"])
```

---

## 😊 NLP - 感情分析

### プレイヤーレビューを自動分析

```bash
# 単一レビュー
python 2_nlp/sentiment_analyzer.py \
    --text "This game is amazing! Best RPG ever!"

# バッチ分析
python 2_nlp/sentiment_analyzer.py \
    --analyze data/reviews.json \
    --output report.txt
```

**レビューデータ例** (data/reviews.json):
```json
[
  {"text": "Great game! Love it.", "rating": 5},
  {"text": "Too many bugs, unplayable", "rating": 1},
  {"text": "Good but needs improvement", "rating": 3}
]
```

**出力レポート**:
```
SENTIMENT ANALYSIS REPORT
========================================
Total Reviews: 150
Positive: 102 (68.0%)
Negative: 36 (24.0%)
Neutral: 12 (8.0%)

Top Negative Reviews (改善点):
1. "Too many crashes on level 3..."
2. "Controls are terrible..."
```

---

## ⚡ Optimization - モデル高速化

### ONNX変換で3-5x高速化

```bash
# PyTorch → ONNX変換
python 3_optimization/onnx_export.py \
    --model checkpoints/classifier.pth \
    --output classifier.onnx \
    --input-shape 1 3 224 224

# ベンチマーク
python 3_optimization/onnx_export.py \
    --benchmark \
    --model checkpoints/classifier.pth \
    --onnx classifier.onnx
```

**ベンチマーク結果**:
```
PyTorch (FP32):  100ms → 10 FPS
ONNX (CPU):      30ms  → 33 FPS  (3.3x faster!)
ONNX (GPU):      10ms  → 100 FPS (10x faster!)
```

### 量子化で75%サイズ削減

```bash
# FP16量子化（50%削減）
python 3_optimization/quantization.py \
    --model checkpoints/model.pth \
    --quantize fp16 \
    --output model_fp16.pth

# INT8量子化（75%削減）
python 3_optimization/quantization.py \
    --model checkpoints/model.pth \
    --quantize int8 \
    --output model_int8.pth
```

**サイズ比較**:
```
Original (FP32): 100 MB
FP16:            50 MB  (50% reduction)
INT8:            25 MB  (75% reduction)
```

---

## 🎯 実践例: 完全ワークフロー

### ゲームアセット管理パイプライン

```python
from image_classifier import GameAssetClassifier
from pathlib import Path

# 1. 分類器準備
classifier = GameAssetClassifier()
classifier.load("checkpoints/asset_classifier.pth")

# 2. 新規アセットを自動分類
for asset in Path("imports/new_assets").glob("*.png"):
    result = classifier.predict(str(asset))
    category = result["top_class"]

    # 自動整理
    dest = Path(f"organized/{category}/{asset.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    asset.rename(dest)

    print(f"{asset.name} → {category}")
```

### プレイヤーフィードバック監視

```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()

def analyze_feedback(reviews):
    analysis = analyzer.analyze_reviews(reviews)

    # アラート判定
    if analysis["statistics"]["negative_ratio"] > 30:
        print("⚠️ Warning: High negative sentiment!")

        # 主な問題抽出
        issues = analyzer.extract_topics(
            [r["text"] for r in reviews],
            sentiment_filter="NEGATIVE"
        )
        print(f"Main issues: {issues[:5]}")

        # チームに通知
        send_alert(f"Negative sentiment detected: {issues}")
```

---

## 📊 パフォーマンス目安

### Computer Vision
- **学習時間**: 10-20分（10エポック、軽量モデル）
- **推論速度**: 10-50ms/画像（CPU）
- **精度**: 85-95%（少数データでも）

### NLP
- **生成速度**: 1-3秒/テキスト（GPT-2）
- **感情分析**: 100-500レビュー/秒
- **精度**: 90-95%

### Optimization
- **ONNX変換**: 数分
- **量子化**: 数分
- **速度向上**: 2-10x
- **サイズ削減**: 50-75%

---

## 🚀 次のステップ

### 学習を深める
1. **詳細ガイド**を読む:
   - [CV_GUIDE.md](1_computer_vision/CV_GUIDE.md)
   - [NLP_GUIDE.md](2_nlp/NLP_GUIDE.md)
   - [OPTIMIZATION_GUIDE.md](3_optimization/OPTIMIZATION_GUIDE.md)

2. **ファインチューニング**で精度向上:
   - ゲーム固有データで再学習
   - 小規模データセットでも効果的

3. **プロダクション環境**へデプロイ:
   - ONNX → Unity/Unreal統合
   - API化してクラウドデプロイ
   - リアルタイム推論システム構築

### プロジェクトアイデア
- 🎨 アセット自動タグ付けツール
- 📊 レビュー分析ダッシュボード
- 🤖 動的NPC会話生成システム
- 🎮 UI自動テストツール

---

## 💡 Tips

### よくある質問

**Q: GPUが無くても使える？**
A: はい！全てCPUで動作します。ONNX+量子化で十分高速です。

**Q: データが少ない場合は？**
A: 転移学習（事前学習済みモデル）で100枚以下でも高精度！

**Q: ゲームエンジンと統合できる？**
A: ONNX形式にすればUnity/Unrealで使用可能です。

**Q: 商用利用可能？**
A: はい。PyTorchとHugging Faceモデルはライセンス確認必要。

---

## 📞 サポート

- **エラーが出たら**: 各ディレクトリのGUIDEファイル参照
- **質問**: GitHubのIssueで質問可能
- **貢献**: Pull Request歓迎！

---

**さあ、実用的なゲームAIを作りましょう！ 🎮🤖**
