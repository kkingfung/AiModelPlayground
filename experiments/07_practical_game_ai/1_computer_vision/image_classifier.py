"""
Game Asset Image Classifier

ゲームアセット（キャラクター、UI、背景、アイテムなど）を自動分類する画像分類器.
転移学習を使用して少ないデータで高精度を実現.

使い方:
    # 学習
    python image_classifier.py --train data/assets --epochs 10

    # 推論
    python image_classifier.py --predict new_asset.png --model checkpoints/best.pth

    # 評価
    python image_classifier.py --evaluate data/test --model checkpoints/best.pth
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from tqdm import tqdm


class GameAssetClassifier:
    """
    ゲームアセット分類器.

    事前学習済みResNet50をファインチューニングして、
    ゲームアセットを自動分類します.
    """

    def __init__(
        self,
        num_classes: int = 5,
        model_name: str = "resnet50",
        pretrained: bool = True,
        device: Optional[str] = None
    ):
        """
        Args:
            num_classes: 分類クラス数
            model_name: ベースモデル名（resnet50, efficientnet_b0など）
            pretrained: ImageNetで事前学習済みの重みを使用
            device: デバイス（cuda/cpu）
        """
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # モデル構築
        self.model = self._build_model(pretrained)
        self.model.to(self.device)

        # データ変換
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.class_names: List[str] = []

    def _build_model(self, pretrained: bool) -> nn.Module:
        """モデル構築."""
        if self.model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            # 最終層を置き換え
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)

        elif self.model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)

        elif self.model_name == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(pretrained=pretrained)
            num_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(num_features, self.num_classes)

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model

    def train(
        self,
        train_dir: str,
        val_dir: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
        save_dir: str = "checkpoints"
    ) -> Dict[str, List[float]]:
        """
        モデルを学習.

        Args:
            train_dir: 学習データディレクトリ（サブフォルダがクラス名）
            val_dir: 検証データディレクトリ
            epochs: エポック数
            batch_size: バッチサイズ
            lr: 学習率
            save_dir: モデル保存ディレクトリ

        Returns:
            学習履歴（loss, accuracy）
        """
        # データセット準備
        train_dataset = ImageFolder(train_dir, transform=self.train_transform)
        self.class_names = train_dataset.classes

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = None
        if val_dir:
            val_dataset = ImageFolder(val_dir, transform=self.val_transform)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )

        # 損失関数と最適化
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )

        # 学習ループ
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 40)

            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Validation phase
            if val_loader:
                val_loss, val_acc = self._validate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

                scheduler.step(val_loss)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save(Path(save_dir) / "best.pth")
                    print(f"Saved best model (acc: {best_val_acc:.4f})")

        # Save final model
        self.save(Path(save_dir) / "final.pth")

        # Save history
        with open(Path(save_dir) / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> Tuple[float, float]:
        """1エポック学習."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc="Training")
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def _validate(
        self,
        loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """検証."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Validating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def predict(
        self,
        image_path: str,
        top_k: int = 3
    ) -> Dict[str, any]:
        """
        画像を分類.

        Args:
            image_path: 画像ファイルパス
            top_k: 上位K個の予測を返す

        Returns:
            予測結果（クラス名、確率）
        """
        self.model.eval()

        # 画像読み込みと変換
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.val_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Top-K取得
        top_probs, top_indices = torch.topk(probabilities, top_k)

        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                "class": self.class_names[idx] if self.class_names else f"class_{idx}",
                "probability": prob.item(),
                "confidence": prob.item() * 100
            })

        return {
            "predictions": results,
            "top_class": results[0]["class"],
            "top_probability": results[0]["probability"]
        }

    def save(self, path: str):
        """モデル保存."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "model_name": self.model_name
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """モデル読み込み."""
        checkpoint = torch.load(path, map_location=self.device)

        self.class_names = checkpoint["class_names"]
        self.num_classes = checkpoint["num_classes"]
        self.model_name = checkpoint["model_name"]

        # モデル再構築
        self.model = self._build_model(pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        print(f"Model loaded from {path}")


def main():
    parser = argparse.ArgumentParser(description="Game Asset Image Classifier")
    parser.add_argument("--train", type=str, help="Training data directory")
    parser.add_argument("--val", type=str, help="Validation data directory")
    parser.add_argument("--predict", type=str, help="Image path to predict")
    parser.add_argument("--model", type=str, default="checkpoints/best.pth", help="Model path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--model-name", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet_b0", "mobilenet_v3_small"],
                        help="Base model architecture")
    parser.add_argument("--num-classes", type=int, default=5, help="Number of classes")

    args = parser.parse_args()

    # 学習モード
    if args.train:
        print("Training Game Asset Classifier...")
        classifier = GameAssetClassifier(
            num_classes=args.num_classes,
            model_name=args.model_name
        )

        history = classifier.train(
            train_dir=args.train,
            val_dir=args.val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

        print("\nTraining completed!")
        print(f"Final train accuracy: {history['train_acc'][-1]:.4f}")
        if history["val_acc"]:
            print(f"Final val accuracy: {history['val_acc'][-1]:.4f}")

    # 推論モード
    elif args.predict:
        print(f"Predicting: {args.predict}")

        classifier = GameAssetClassifier(num_classes=args.num_classes)
        classifier.load(args.model)

        result = classifier.predict(args.predict, top_k=3)

        print("\nPrediction Results:")
        print(f"Top class: {result['top_class']}")
        print(f"Confidence: {result['top_probability']*100:.2f}%\n")

        print("Top 3 predictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  {i}. {pred['class']}: {pred['confidence']:.2f}%")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
