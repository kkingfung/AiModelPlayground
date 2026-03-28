# Computer Vision Guide for Game Development

## 概要

ゲーム開発で実用的なComputer Vision技術を学びます:
- 画像分類（Image Classification）
- 物体検出（Object Detection）
- スタイル転送（Style Transfer）
- セグメンテーション（Segmentation）

---

## 1. 画像分類（Image Classifier）

### 用途
- ゲームアセット自動分類（キャラクター、UI、背景、アイテム）
- スクリーンショット分類
- アート品質チェック

### 使い方

#### 学習データ準備
```
data/assets/
├── character/
│   ├── hero01.png
│   ├── enemy01.png
│   └── ...
├── ui/
│   ├── button01.png
│   └── ...
├── background/
│   └── ...
├── item/
│   └── ...
└── weapon/
    └── ...
```

#### 学習
```bash
python image_classifier.py \
    --train data/assets \
    --val data/assets_val \
    --epochs 20 \
    --batch-size 32 \
    --model-name resnet50
```

#### 推論
```bash
python image_classifier.py \
    --predict new_asset.png \
    --model checkpoints/best.pth
```

### モデル選択

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| MobileNetV3 | 小 | 速い | 中 | モバイル・リアルタイム |
| ResNet50 | 中 | 中 | 高 | バランス型 |
| EfficientNet-B0 | 小-中 | 中-速 | 高 | 効率重視 |

### Tips
- **データ拡張**: 色調変化、反転、回転で精度向上
- **転移学習**: ImageNet事前学習済みモデルで少ないデータでも高精度
- **クラスバランス**: 各クラスのデータ数を均等に

---

## 2. UI要素検出（UI Detector）

### 用途
- スクリーンショットからUI要素自動検出
- UI/UXテスト自動化
- アクセシビリティチェック
- ゲームプレイ動画分析

### 使い方

#### データセット準備（YOLO形式）
```
data/ui_dataset/
├── images/
│   ├── train/
│   │   ├── screenshot001.png
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── screenshot001.txt
    │   └── ...
    └── val/
        └── ...
```

**ラベルフォーマット** (screenshot001.txt):
```
0 0.5 0.2 0.1 0.05  # class x_center y_center width height (normalized)
1 0.8 0.9 0.15 0.08
```

#### 学習
```bash
python ui_detector.py \
    --train data/ui_dataset.yaml \
    --epochs 50 \
    --imgsz 640 \
    --batch 16
```

#### 検出
```bash
# 画像
python ui_detector.py \
    --detect screenshot.png \
    --model runs/detect/train/weights/best.pt \
    --conf 0.25

# 動画
python ui_detector.py \
    --detect-video gameplay.mp4 \
    --model best.pt
```

#### レイアウト分析
```bash
python ui_detector.py \
    --analyze screenshot.png \
    --model best.pt
```

### 検出可能なUI要素例
- ボタン（Button）
- ヘルスバー（Health Bar）
- メニュー（Menu）
- アイコン（Icon）
- テキストフィールド（Text Field）
- ミニマップ（Minimap）

### アノテーションツール
- **LabelImg**: https://github.com/heartexlabs/labelImg
- **CVAT**: https://cvat.org/
- **Roboflow**: https://roboflow.com/

---

## 3. パフォーマンス最適化

### 推論速度改善

#### 1. 軽量モデル使用
```python
# 速度重視
classifier = GameAssetClassifier(model_name="mobilenet_v3_small")

# 精度重視
classifier = GameAssetClassifier(model_name="resnet50")
```

#### 2. バッチ処理
```python
# 複数画像を一度に処理
images = [img1, img2, img3, ...]
results = model.predict_batch(images, batch_size=16)
```

#### 3. ONNX変換（詳細は3_optimization/参照）
```python
detector.export_onnx("ui_detector.onnx", imgsz=640)
```

### メモリ使用量削減

#### Mixed Precision (FP16)
```python
from torch.cuda.amp import autocast

with autocast():
    outputs = model(inputs)
```

#### Gradient Checkpointing
```python
import torch.utils.checkpoint as checkpoint

# 学習時のメモリ削減
outputs = checkpoint.checkpoint(model_segment, inputs)
```

---

## 4. 実用例

### Example 1: アセット管理パイプライン

```python
from image_classifier import GameAssetClassifier
from pathlib import Path

classifier = GameAssetClassifier()
classifier.load("checkpoints/asset_classifier.pth")

# 新規アセットを自動分類
new_assets = Path("imports/new_assets").glob("*.png")

for asset in new_assets:
    result = classifier.predict(str(asset))
    category = result["top_class"]

    # カテゴリ別に整理
    dest = Path(f"organized_assets/{category}/{asset.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    asset.rename(dest)

    print(f"{asset.name} -> {category} ({result['top_probability']:.2%})")
```

### Example 2: スクリーンショット自動検証

```python
from ui_detector import UIElementDetector

detector = UIElementDetector(model_path="ui_detector.pt")

# UI要素が正しく配置されているかチェック
analysis = detector.analyze_ui_layout("screenshot.png")

# 検証ルール
if analysis["quadrant_distribution"]["top_left"] == 0:
    print("Warning: No UI elements in top-left (where menu should be)")

if analysis["class_distribution"].get("health_bar", 0) == 0:
    print("Warning: Health bar not detected")

if analysis["layout_density"] > 0.5:
    print("Warning: UI too crowded")
```

### Example 3: バッチ推論API

```python
from fastapi import FastAPI, UploadFile, File
from image_classifier import GameAssetClassifier
import io
from PIL import Image

app = FastAPI()
classifier = GameAssetClassifier()
classifier.load("checkpoints/best.pth")

@app.post("/classify")
async def classify_asset(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))

    # 一時保存して推論
    temp_path = f"temp/{file.filename}"
    image.save(temp_path)

    result = classifier.predict(temp_path)

    return {
        "filename": file.filename,
        "category": result["top_class"],
        "confidence": result["top_probability"],
        "top_3": result["predictions"]
    }
```

---

## 5. トラブルシューティング

### 過学習（Overfitting）
**症状**: Train accuracyは高いが、Val accuracyが低い

**対策**:
- データ拡張を強化
- Dropoutを追加
- Early Stoppingを使用
- 正則化（L2 regularization）

### 学習が進まない
**症状**: Lossが下がらない

**対策**:
- 学習率を調整（0.001 → 0.0001）
- バッチサイズを変更
- オプティマイザを変更（Adam → AdamW）
- 事前学習済みモデルを使用

### 検出精度が低い（YOLO）
**対策**:
- アノテーションの品質チェック
- エポック数を増やす（50 → 100）
- 画像サイズを大きく（640 → 1280）
- データ拡張を調整

---

## 6. 次のステップ

Computer Visionをマスターしたら:
1. **NLP** (2_nlp/) - テキスト生成と感情分析
2. **Optimization** (3_optimization/) - モデル最適化とデプロイ
3. **Integration** (4_integration/) - ゲームエンジンとの統合

---

## リソース

- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **torchvision Docs**: https://pytorch.org/vision/stable/index.html
- **Ultralytics YOLOv8**: https://docs.ultralytics.com/
- **Papers with Code**: https://paperswithcode.com/

---

**Happy Computer Vision! 🎮👁️**
