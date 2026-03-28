"""
Transfer Learning for Game Asset Classification

事前学習済みモデルを使用してゲームアセット分類器を迅速に構築します。
少量のデータで高精度なモデルを訓練できます。

特徴:
    - Pre-trained models (ResNet, EfficientNet, ViT)
    - Few-shot learning (少量データで学習)
    - Fine-tuning strategies (full, partial, head-only)
    - Data augmentation for games
    - Progressive unfreezing

使い方:
    from asset_classifier_transfer import AssetClassifierTransfer

    # モデル作成
    classifier = AssetClassifierTransfer(
        backbone='resnet50',
        num_classes=10,
        pretrained=True
    )

    # データ準備
    train_data = ...
    val_data = ...

    # Fine-tuning
    classifier.fine_tune(
        train_data,
        val_data,
        strategy='progressive',
        epochs=20
    )

    # 保存
    classifier.save('game_asset_classifier.pth')

参考:
    - Transfer Learning: https://cs231n.github.io/transfer-learning/
    - Fine-tuning strategies
    - Domain adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_B0_Weights,
    ViT_B_16_Weights
)
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import time


class GameAssetDataset(Dataset):
    """
    ゲームアセット用データセット.

    ディレクトリ構造:
        data/
        ├── weapons/
        │   ├── sword_01.png
        │   └── bow_02.png
        ├── characters/
        │   └── hero_03.png
        └── items/
            └── potion_04.png
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            root_dir: ルートディレクトリ
            transform: 画像変換
            class_to_idx: クラス名からインデックスへのマッピング
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # クラスとサンプルを収集
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])

        if class_to_idx is None:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx

        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # 画像パスとラベルを収集
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]

            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    self.samples.append((str(img_path), class_idx))

        print(f"Found {len(self.samples)} images in {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 画像読み込み
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class GameAssetTransforms:
    """
    ゲームアセット用の画像変換.

    訓練時と評価時で異なる変換を使用します.
    """

    @staticmethod
    def get_train_transforms(img_size: int = 224) -> transforms.Compose:
        """
        訓練用変換（データ拡張あり）.

        Args:
            img_size: 画像サイズ

        Returns:
            transform: 変換パイプライン
        """
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @staticmethod
    def get_val_transforms(img_size: int = 224) -> transforms.Compose:
        """
        評価用変換（データ拡張なし）.

        Args:
            img_size: 画像サイズ

        Returns:
            transform: 変換パイプライン
        """
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class AssetClassifierTransfer:
    """
    Transfer Learning ベースのゲームアセット分類器.

    事前学習済みモデルをFine-tuningして
    少量のデータで高精度な分類器を構築します.
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        num_classes: int = 10,
        pretrained: bool = True,
        img_size: int = 224,
        device: str = 'auto'
    ):
        """
        Args:
            backbone: バックボーンモデル ('resnet50', 'efficientnet_b0', 'vit_b_16')
            num_classes: クラス数
            pretrained: 事前学習済み重みを使用
            img_size: 入力画像サイズ
            device: デバイス
        """
        # デバイス設定
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.img_size = img_size

        # モデル作成
        self.model = self._create_model(backbone, num_classes, pretrained)
        self.model.to(self.device)

        # クラス名マッピング
        self.class_names = None

        # 訓練履歴
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        print(f"✓ AssetClassifierTransfer initialized")
        print(f"  Backbone: {backbone}")
        print(f"  Classes: {num_classes}")
        print(f"  Device: {self.device}")

    def _create_model(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool
    ) -> nn.Module:
        """モデルを作成."""
        if backbone == 'resnet50':
            if pretrained:
                model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                model = models.resnet50(weights=None)

            # 最終層を置き換え
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)

        elif backbone == 'efficientnet_b0':
            if pretrained:
                model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                model = models.efficientnet_b0(weights=None)

            # 最終層を置き換え
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, num_classes)

        elif backbone == 'vit_b_16':
            if pretrained:
                model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            else:
                model = models.vit_b_16(weights=None)

            # 最終層を置き換え
            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, num_classes)

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        return model

    def fine_tune(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        strategy: str = 'progressive',
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        save_best: bool = True,
        checkpoint_path: str = 'best_model.pth'
    ):
        """
        モデルをFine-tuning.

        Args:
            train_dataset: 訓練データセット
            val_dataset: 検証データセット
            strategy: Fine-tuning戦略
                - 'head_only': 最終層のみ訓練
                - 'full': 全層を訓練
                - 'progressive': 段階的にアンフリーズ
            epochs: エポック数
            batch_size: バッチサイズ
            lr: 学習率
            weight_decay: 重み減衰
            save_best: ベストモデルを保存
            checkpoint_path: チェックポイントパス
        """
        # クラス名を保存
        if hasattr(train_dataset, 'idx_to_class'):
            self.class_names = train_dataset.idx_to_class

        # DataLoader作成
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 損失関数
        criterion = nn.CrossEntropyLoss()

        # 戦略に応じた訓練
        if strategy == 'head_only':
            self._fine_tune_head_only(
                train_loader, val_loader, criterion,
                epochs, lr, weight_decay, save_best, checkpoint_path
            )
        elif strategy == 'full':
            self._fine_tune_full(
                train_loader, val_loader, criterion,
                epochs, lr, weight_decay, save_best, checkpoint_path
            )
        elif strategy == 'progressive':
            self._fine_tune_progressive(
                train_loader, val_loader, criterion,
                epochs, lr, weight_decay, save_best, checkpoint_path
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _fine_tune_head_only(
        self,
        train_loader, val_loader, criterion,
        epochs, lr, weight_decay, save_best, checkpoint_path
    ):
        """最終層のみをFine-tuning."""
        print("\n=== Fine-tuning Strategy: Head Only ===")

        # バックボーンをフリーズ
        self._freeze_backbone()

        # 最終層のみのオプティマイザ
        if self.backbone_name == 'resnet50':
            params = self.model.fc.parameters()
        elif self.backbone_name == 'efficientnet_b0':
            params = self.model.classifier.parameters()
        elif self.backbone_name == 'vit_b_16':
            params = self.model.heads.parameters()

        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        best_val_acc = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self._validate(val_loader, criterion)

            scheduler.step()

            # 履歴を記録
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # ベストモデルを保存
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save(checkpoint_path)
                print(f"  → Best model saved (Val Acc: {val_acc:.4f})")

    def _fine_tune_full(
        self,
        train_loader, val_loader, criterion,
        epochs, lr, weight_decay, save_best, checkpoint_path
    ):
        """全層をFine-tuning."""
        print("\n=== Fine-tuning Strategy: Full ===")

        # 全層をアンフリーズ
        self._unfreeze_all()

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self._validate(val_loader, criterion)

            scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save(checkpoint_path)
                print(f"  → Best model saved (Val Acc: {val_acc:.4f})")

    def _fine_tune_progressive(
        self,
        train_loader, val_loader, criterion,
        epochs, lr, weight_decay, save_best, checkpoint_path
    ):
        """段階的にアンフリーズしてFine-tuning."""
        print("\n=== Fine-tuning Strategy: Progressive ===")

        # Stage 1: Head only
        print("\nStage 1: Training head only...")
        self._freeze_backbone()

        if self.backbone_name == 'resnet50':
            params = self.model.fc.parameters()
        elif self.backbone_name == 'efficientnet_b0':
            params = self.model.classifier.parameters()
        elif self.backbone_name == 'vit_b_16':
            params = self.model.heads.parameters()

        optimizer = optim.Adam(params, lr=lr * 10, weight_decay=weight_decay)

        stage1_epochs = epochs // 3
        for epoch in range(stage1_epochs):
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self._validate(val_loader, criterion)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"[Stage 1] Epoch {epoch+1}/{stage1_epochs} - "
                  f"Val Acc: {val_acc:.4f}")

        # Stage 2: Unfreeze top layers
        print("\nStage 2: Unfreezing top layers...")
        self._unfreeze_top_layers()

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        stage2_epochs = epochs // 3
        best_val_acc = 0.0

        for epoch in range(stage2_epochs):
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self._validate(val_loader, criterion)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"[Stage 2] Epoch {epoch+1}/{stage2_epochs} - "
                  f"Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_best:
                    self.save(checkpoint_path)

        # Stage 3: Full fine-tuning
        print("\nStage 3: Full fine-tuning...")
        self._unfreeze_all()

        optimizer = optim.Adam(self.model.parameters(), lr=lr / 10, weight_decay=weight_decay)

        stage3_epochs = epochs - stage1_epochs - stage2_epochs
        for epoch in range(stage3_epochs):
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self._validate(val_loader, criterion)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"[Stage 3] Epoch {epoch+1}/{stage3_epochs} - "
                  f"Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_best:
                    self.save(checkpoint_path)
                    print(f"  → Best model saved (Val Acc: {val_acc:.4f})")

    def _train_epoch(self, dataloader, criterion, optimizer):
        """1エポック訓練."""
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Training")
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': correct / total})

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def _validate(self, dataloader, criterion):
        """検証."""
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(dataloader)
        val_acc = correct / total

        return val_loss, val_acc

    def _freeze_backbone(self):
        """バックボーンをフリーズ."""
        for param in self.model.parameters():
            param.requires_grad = False

        # 最終層のみアンフリーズ
        if self.backbone_name == 'resnet50':
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif self.backbone_name == 'efficientnet_b0':
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif self.backbone_name == 'vit_b_16':
            for param in self.model.heads.parameters():
                param.requires_grad = True

    def _unfreeze_top_layers(self):
        """上位層をアンフリーズ."""
        # 簡易実装: 最後の2ブロックをアンフリーズ
        if self.backbone_name == 'resnet50':
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        # 他のモデルも同様に実装可能

    def _unfreeze_all(self):
        """全層をアンフリーズ."""
        for param in self.model.parameters():
            param.requires_grad = True

    def predict(self, image_path: str) -> Dict:
        """画像を分類."""
        self.model.eval()

        # 画像読み込み
        image = Image.open(image_path).convert('RGB')

        # 変換
        transform = GameAssetTransforms.get_val_transforms(self.img_size)
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # 推論
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        confidence = confidence.item()

        result = {
            'class_idx': predicted_class,
            'confidence': confidence
        }

        if self.class_names:
            result['class_name'] = self.class_names[predicted_class]

        return result

    def save(self, filepath: str):
        """モデルを保存."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'backbone': self.backbone_name,
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'class_names': self.class_names,
            'history': self.history
        }

        torch.save(save_dict, filepath)

    def load(self, filepath: str):
        """モデルを読み込み."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint.get('class_names')
        self.history = checkpoint.get('history', {})

        print(f"✓ Model loaded from {filepath}")


# ============================================================================
# デモ
# ============================================================================

if __name__ == "__main__":
    print("Transfer Learning for Game Assets Demo")
    print("=" * 60)

    # このデモでは実際のデータセットが必要です
    # 以下は使用例を示しています

    print("\n✓ Demo setup (requires actual dataset)")
    print("\nExample usage:")
    print("""
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
        epochs=20,
        batch_size=32
    )

    # 推論
    result = classifier.predict('test_image.png')
    print(f"Class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2%}")
    """)

    print("\n✓ See README.md for complete setup instructions")
