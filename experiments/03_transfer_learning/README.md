# Experiment 03: Transfer Learning for Game Assets

事前学習済みモデルを活用して、少量のデータで高精度なゲームアセット分類器を構築

---

## 概要

Transfer Learningを使用することで、ImageNetなどの大規模データセットで訓練された知識を活用し、
わずか数百枚の画像で高精度な分類器を構築できます。

### なぜTransfer Learningか?

**従来の方法:**
- 数万〜数十万枚の画像が必要
- 訓練に数日〜数週間
- 高性能なGPUが必須

**Transfer Learning:**
- 数百〜数千枚の画像で十分
- 訓練は数時間
- 一般的なGPUで可能

---

## クイックスタート

### インストール

```bash
cd experiments/03_transfer_learning
pip install -r requirements.txt
```

### データ準備

```
data/
├── train/
│   ├── weapons/
│   │   ├── sword_01.png
│   │   └── bow_02.png
│   ├── characters/
│   └── items/
└── val/
    ├── weapons/
    ├── characters/
    └── items/
```

### 基本的な使用

```python
from asset_classifier_transfer import (
    AssetClassifierTransfer,
    GameAssetDataset,
    GameAssetTransforms
)

# データセット作成
train_dataset = GameAssetDataset(
    root_dir='data/train',
    transform=GameAssetTransforms.get_train_transforms()
)

val_dataset = GameAssetDataset(
    root_dir='data/val',
    transform=GameAssetTransforms.get_val_transforms()
)

# モデル作成
classifier = AssetClassifierTransfer(
    backbone='resnet50',
    num_classes=10,
    pretrained=True
)

# Fine-tuning
classifier.fine_tune(
    train_dataset,
    val_dataset,
    strategy='progressive',
    epochs=20
)

# 保存
classifier.save('asset_classifier.pth')
```

---

## Fine-tuning戦略

### 1. Head Only (ヘッドのみ訓練)

最も速く、データが非常に少ない場合に有効。

```python
classifier.fine_tune(
    train_dataset,
    val_dataset,
    strategy='head_only',
    epochs=10
)
```

**メリット:**
- 最速（数分〜数十分）
- メモリ使用量が少ない
- 過学習しにくい

**デメリット:**
- 精度が限定的
- ドメインが大きく異なる場合は不十分

**推奨:**
- データ: 100-500枚
- ImageNetと似たドメイン

### 2. Full (全層訓練)

最も高精度、データが十分ある場合に有効。

```python
classifier.fine_tune(
    train_dataset,
    val_dataset,
    strategy='full',
    epochs=30
)
```

**メリット:**
- 最高精度
- ドメイン適応が良い

**デメリット:**
- 時間がかかる
- 過学習のリスク
- メモリ使用量が大きい

**推奨:**
- データ: 1000枚以上
- ドメインが異なる場合

### 3. Progressive (段階的訓練)

バランスが良く、ほとんどの場合に推奨。

```python
classifier.fine_tune(
    train_dataset,
    val_dataset,
    strategy='progressive',
    epochs=20
)
```

**ステージ:**
1. ヘッドのみ訓練（高学習率）
2. 上位層をアンフリーズ（中学習率）
3. 全層を訓練（低学習率）

**メリット:**
- 安定した学習
- 過学習を防ぐ
- 高精度

**推奨:**
- データ: 500-5000枚
- ほとんどの場合

---

## バックボーンモデル

### ResNet50

```python
classifier = AssetClassifierTransfer(
    backbone='resnet50',
    num_classes=10
)
```

**特徴:**
- 汎用性が高い
- 安定した性能
- 中程度の速度

**推奨:** 汎用的なアセット分類

### EfficientNet-B0

```python
classifier = AssetClassifierTransfer(
    backbone='efficientnet_b0',
    num_classes=10
)
```

**特徴:**
- 高効率
- 小さいモデルサイズ
- 高速

**推奨:** モバイル、リアルタイム推論

### Vision Transformer (ViT)

```python
classifier = AssetClassifierTransfer(
    backbone='vit_b_16',
    num_classes=10
)
```

**特徴:**
- 最新アーキテクチャ
- 高精度（データが十分な場合）
- 遅い

**推奨:** 高精度が必要な場合

---

## データ拡張

### 訓練用変換

```python
transforms = GameAssetTransforms.get_train_transforms(img_size=224)
```

**適用される変換:**
- リサイズ
- ランダムフリップ
- ランダム回転 (±15度)
- 色調整 (明度、コントラスト、彩度)
- アフィン変換 (平行移動、スケール)

### カスタム変換

```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    # ゲーム固有の変換
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## パフォーマンス最適化

### 学習率調整

```python
# 高学習率（ヘッドのみ）
classifier.fine_tune(..., lr=1e-2)

# 低学習率（全層）
classifier.fine_tune(..., lr=1e-4)
```

### バッチサイズ

```python
# 小さいバッチ（メモリ制約）
classifier.fine_tune(..., batch_size=16)

# 大きいバッチ（高速）
classifier.fine_tune(..., batch_size=64)
```

### Mixed Precision Training

```python
# PyTorch AMP使用
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 実践例

### 1. 武器アセット分類

```python
# データ構造
data/train/
├── swords/      (150 images)
├── bows/        (120 images)
├── staffs/      (100 images)
└── guns/        (180 images)

# 訓練
classifier = AssetClassifierTransfer(
    backbone='resnet50',
    num_classes=4
)

classifier.fine_tune(
    train_dataset, val_dataset,
    strategy='progressive',
    epochs=15,
    batch_size=32
)

# 結果: 95%+ accuracy
```

### 2. キャラクタースタイル分類

```python
# スタイル分類（cartoon, realistic, pixel-art, etc.）
classifier = AssetClassifierTransfer(
    backbone='efficientnet_b0',
    num_classes=5
)

classifier.fine_tune(
    train_dataset, val_dataset,
    strategy='head_only',  # スタイルはImageNetと近い
    epochs=10
)

# 結果: 90%+ accuracy with only 300 images
```

### 3. Few-Shot Learning

```python
# 各クラス20-50枚のみ
from torch.utils.data import Subset
import numpy as np

# 各クラスから少数サンプリング
def create_few_shot_dataset(dataset, shots_per_class=20):
    indices_by_class = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in indices_by_class:
            indices_by_class[label] = []
        indices_by_class[label].append(idx)

    few_shot_indices = []
    for label, indices in indices_by_class.items():
        sampled = np.random.choice(indices, size=shots_per_class, replace=False)
        few_shot_indices.extend(sampled)

    return Subset(dataset, few_shot_indices)

few_shot_train = create_few_shot_dataset(train_dataset, shots_per_class=20)

classifier.fine_tune(
    few_shot_train, val_dataset,
    strategy='progressive',
    epochs=30  # より多くのエポック
)
```

---

## 評価とデバッグ

### 学習曲線の可視化

```python
import matplotlib.pyplot as plt

history = classifier.history

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.tight_layout()
plt.savefig('training_curves.png')
```

### 混同行列

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 予測
y_true = []
y_pred = []

for img_path, label in val_dataset.samples:
    result = classifier.predict(img_path)
    y_true.append(label)
    y_pred.append(result['class_idx'])

# 混同行列
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# 詳細レポート
print(classification_report(
    y_true, y_pred,
    target_names=classifier.class_names
))
```

---

## トラブルシューティング

### 過学習

**症状:** Train accuracyは高いがVal accuracyが低い

**対策:**
```python
# データ拡張を強化
transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),  # より大きく
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    # Dropoutを追加（モデル改造）
])

# Regularization
classifier.fine_tune(
    ...,
    weight_decay=1e-3,  # 大きく
    strategy='head_only'  # より保守的
)
```

### 低精度

**症状:** Val accuracyが上がらない

**対策:**
```python
# より多くのデータ
# より多くのエポック
# 学習率調整
classifier.fine_tune(..., lr=1e-4)

# 別のバックボーン
classifier = AssetClassifierTransfer(
    backbone='efficientnet_b0',  # ResNetから変更
    num_classes=num_classes
)
```

### メモリ不足

**対策:**
```python
# バッチサイズを削減
classifier.fine_tune(..., batch_size=8)

# 小さいモデル
classifier = AssetClassifierTransfer(
    backbone='efficientnet_b0',
    num_classes=num_classes
)

# Mixed precision
# (コード内で実装)
```

---

## ベストプラクティス

1. **データ分割**: Train 70%, Val 15%, Test 15%
2. **データ拡張**: 常に使用
3. **Early Stopping**: 過学習を防ぐ
4. **学習率スケジューリング**: 段階的に削減
5. **アンサンブル**: 複数モデルで投票

---

## 参考文献

- Transfer Learning: https://cs231n.github.io/transfer-learning/
- ResNet: https://arxiv.org/abs/1512.03385
- EfficientNet: https://arxiv.org/abs/1905.11946
- Vision Transformer: https://arxiv.org/abs/2010.11929

---

**Transfer Learning = データ効率 × 高精度 🚀**
