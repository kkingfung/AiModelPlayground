# Experiment 07: Practical AI for Game Development

## 概要 (Overview)

ゲーム開発で実際に使える3つのAI技術を実装:
- **Computer Vision (CV)**: 画像分類・物体検出・スタイル転送
- **Natural Language Processing (NLP)**: テキスト生成・感情分析・会話AI
- **Model Optimization**: 軽量化・高速化・デプロイ最適化

## 🎯 学習目標

### Computer Vision
- ✅ 画像分類器の構築（ゲームアセット分類）
- ✅ UI要素の検出（YOLOv8使用）
- ✅ スタイル転送（Neural Style Transfer）
- ✅ セマンティックセグメンテーション

### NLP
- ✅ ゲームテキスト生成（GPT-2/LLaMA）
- ✅ プレイヤーフィードバック分析（感情分析）
- ✅ Named Entity Recognition（ゲーム固有名詞抽出）
- ✅ 会話システム構築

### Model Optimization
- ✅ 量子化（INT8, FP16）
- ✅ ONNX変換とデプロイ
- ✅ TorchScript変換
- ✅ モデル圧縮（pruning, knowledge distillation）

## 📁 ディレクトリ構造

```
07_practical_game_ai/
├── README.md
├── requirements.txt
│
├── 1_computer_vision/
│   ├── image_classifier.py          # ゲームアセット分類器
│   ├── ui_detector.py                # UI要素検出（YOLO）
│   ├── style_transfer.py             # Neural Style Transfer
│   ├── segmentation.py               # セマンティックセグメンテーション
│   └── CV_GUIDE.md
│
├── 2_nlp/
│   ├── text_generator.py             # ゲームテキスト生成
│   ├── sentiment_analyzer.py         # 感情分析
│   ├── ner_extractor.py              # 固有名詞抽出
│   ├── dialogue_system.py            # 会話システム
│   └── NLP_GUIDE.md
│
├── 3_optimization/
│   ├── quantization.py               # モデル量子化
│   ├── onnx_export.py                # ONNX変換
│   ├── torchscript_export.py         # TorchScript変換
│   ├── model_compression.py          # モデル圧縮
│   └── OPTIMIZATION_GUIDE.md
│
├── 4_integration/
│   ├── game_asset_pipeline.py        # CV統合例
│   ├── review_analyzer_api.py        # NLP API例
│   ├── optimized_inference.py        # 最適化推論
│   └── INTEGRATION_GUIDE.md
│
└── data/
    ├── sample_assets/                # サンプル画像
    ├── sample_reviews.json           # サンプルレビュー
    └── sample_dialogues.json         # サンプル会話
```

## 🚀 クイックスタート

### 1. 依存関係インストール

```bash
cd experiments/07_practical_game_ai
pip install -r requirements.txt
```

### 2. Computer Vision例

```bash
# 画像分類
python 1_computer_vision/image_classifier.py \
    --train data/sample_assets \
    --epochs 10

# UI検出
python 1_computer_vision/ui_detector.py \
    --image screenshot.png \
    --detect-buttons

# スタイル転送
python 1_computer_vision/style_transfer.py \
    --content game_scene.png \
    --style art_style.jpg \
    --output stylized.png
```

### 3. NLP例

```bash
# テキスト生成
python 2_nlp/text_generator.py \
    --prompt "The ancient sword glows with" \
    --model gpt2

# 感情分析
python 2_nlp/sentiment_analyzer.py \
    --reviews data/sample_reviews.json \
    --batch-size 32

# 会話システム
python 2_nlp/dialogue_system.py \
    --context "fantasy RPG" \
    --interactive
```

### 4. Model Optimization例

```bash
# 量子化
python 3_optimization/quantization.py \
    --model my_model.pth \
    --quantize int8 \
    --benchmark

# ONNX変換
python 3_optimization/onnx_export.py \
    --model my_model.pth \
    --input-shape 1,3,224,224 \
    --output model.onnx
```

## 💡 実用例

### Example 1: ゲームアセット自動分類

```python
from image_classifier import GameAssetClassifier

classifier = GameAssetClassifier()
classifier.train("data/sample_assets", epochs=10)

# 新しい画像を分類
result = classifier.predict("new_asset.png")
print(f"Category: {result['category']}, Confidence: {result['confidence']}")
```

### Example 2: プレイヤーレビュー分析

```python
from sentiment_analyzer import ReviewAnalyzer

analyzer = ReviewAnalyzer()
reviews = analyzer.load_reviews("player_reviews.json")

# 感情分析
results = analyzer.analyze_batch(reviews)
print(f"Positive: {results['positive_ratio']}%")
print(f"Top complaints: {results['negative_topics']}")
```

### Example 3: 最適化モデルデプロイ

```python
from onnx_export import export_to_onnx
from optimized_inference import ONNXInference

# モデルをONNXに変換
export_to_onnx(model, "classifier.onnx", input_shape=(1, 3, 224, 224))

# 高速推論
inference = ONNXInference("classifier.onnx")
result = inference.predict(image)  # 3-10x faster than PyTorch!
```

## 🎮 ゲーム開発での活用例

### Computer Vision
- **アセット管理**: 自動でアセットをカテゴリ分類
- **UI検証**: スクリーンショットからUI要素を自動検出
- **アート生成**: スタイル転送で新しいアートバリエーション作成
- **品質チェック**: セグメンテーションでアセット品質検証

### NLP
- **コンテンツ生成**: アイテム説明文、クエストテキスト自動生成
- **フィードバック分析**: プレイヤーレビューから改善点抽出
- **ローカライゼーション**: 翻訳品質チェック
- **NPC会話**: 動的な会話システム

### Optimization
- **モバイルデプロイ**: 量子化で軽量化
- **リアルタイム推論**: ONNX/TorchScriptで高速化
- **クラウドコスト削減**: 圧縮モデルでサーバーコスト削減
- **Unity/Unreal統合**: ONNX経由でゲームエンジンに統合

## 📊 パフォーマンス比較

### Model Size Reduction
- **Original PyTorch**: 100 MB
- **FP16 Quantization**: 50 MB (50% reduction)
- **INT8 Quantization**: 25 MB (75% reduction)
- **Pruned + INT8**: 15 MB (85% reduction)

### Inference Speed (CPU)
- **PyTorch (FP32)**: 100ms
- **ONNX (FP32)**: 30ms (3.3x faster)
- **ONNX (INT8)**: 10ms (10x faster)
- **TensorRT (INT8)**: 5ms (20x faster)

### Accuracy Impact
- **FP32 (Baseline)**: 95.0%
- **FP16**: 94.9% (-0.1%)
- **INT8**: 94.5% (-0.5%)
- **Pruned + INT8**: 93.8% (-1.2%)

## 🛠️ 技術スタック

### Computer Vision
- **PyTorch**: ディープラーニングフレームワーク
- **torchvision**: 画像処理・モデル
- **Ultralytics (YOLOv8)**: 物体検出
- **Pillow**: 画像処理
- **OpenCV**: 画像操作

### NLP
- **Transformers (Hugging Face)**: 事前学習モデル
- **spaCy**: NERとテキスト処理
- **NLTK**: 自然言語処理ツール
- **Sentence-Transformers**: 文埋め込み

### Optimization
- **ONNX**: モデル変換・最適化
- **ONNX Runtime**: 高速推論エンジン
- **TorchScript**: PyTorchモデル最適化
- **Neural Compressor**: Intel量子化ツール

## 📚 各セクションの詳細ガイド

- [Computer Vision Guide](1_computer_vision/CV_GUIDE.md)
- [NLP Guide](2_nlp/NLP_GUIDE.md)
- [Optimization Guide](3_optimization/OPTIMIZATION_GUIDE.md)
- [Integration Guide](4_integration/INTEGRATION_GUIDE.md)

## 🎓 学習パス

### 初級（1-2週間）
1. 画像分類器を構築
2. 基本的な感情分析
3. FP16量子化を試す

### 中級（2-4週間）
1. YOLOv8で物体検出
2. テキスト生成モデル構築
3. ONNXでデプロイ

### 上級（1-2ヶ月）
1. カスタムセグメンテーション
2. マルチモーダル会話AI
3. モデル圧縮とプルーニング

## 🚀 次のステップ

このエクスペリメントを完了したら:
- **Experiment 08**: Reinforcement Learning for Game AI
- **Experiment 09**: Generative AI (Stable Diffusion, ControlNet)
- **Experiment 10**: Production ML Pipelines (MLOps)

## 📞 リソース

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Hugging Face Course](https://huggingface.co/course)
- [ONNX Documentation](https://onnx.ai/onnx/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)

---

**始めましょう！実用的なAIスキルを身につけて、ゲーム開発を加速させよう！**
