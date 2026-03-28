# Experiment 01: MNIST Basics

深層学習の基礎を学ぶための入門実験

---

## 概要

MNIST手書き数字認識を通じて、ニューラルネットワークの基礎を学びます。
このは、AI/MLを初めて学ぶ方に最適な実験です。

### 学習内容

1. **ニューラルネットワークの基礎**: 層、ニューロン、活性化関数
2. **訓練ループ**: 機械学習の基本パターン
3. **逆伝播**: モデルが学習する仕組み
4. **評価**: モデルの性能測定
5. **畳み込みニューラルネットワーク (CNN)**: 画像認識の標準手法

---

## クイックスタート

### インストール

```bash
cd experiments/01_mnist_basics
pip install -r requirements.txt
```

### 基本実験の実行

```bash
# シンプルなニューラルネットワーク
python mnist_simple.py

# CNN (より高精度)
python mnist_cnn.py

# インタラクティブ実験ビルダー
python experiment_builder.py
```

---

## 提供するスクリプト

### 1. `mnist_simple.py` - 基本的なニューラルネットワーク

**対象**: 完全な初心者

最もシンプルな全結合ニューラルネットワーク。

**アーキテクチャ:**
```
Input (784 pixels)
    ↓
Dense Layer (128 neurons) + ReLU
    ↓
Dropout (20%)
    ↓
Dense Layer (64 neurons) + ReLU
    ↓
Dropout (20%)
    ↓
Output Layer (10 classes)
```

**期待される結果:**
- 訓練精度: ~98-99%
- テスト精度: ~97-98%
- 訓練時間: 5-10分 (CPU)

**学べること:**
- 全結合層の仕組み
- 活性化関数 (ReLU)
- Dropout による過学習防止
- 訓練ループの基本構造

### 2. `mnist_cnn.py` - 畳み込みニューラルネットワーク

**対象**: 基礎を理解した方

画像認識の標準手法であるCNNを実装。

**アーキテクチャ (Basic CNN):**
```
Input (1×28×28)
    ↓
Conv2d (1→32) + BatchNorm + ReLU + MaxPool
    ↓  (14×14)
Conv2d (32→64) + BatchNorm + ReLU + MaxPool
    ↓  (7×7)
Flatten (64×7×7 = 3136)
    ↓
Dense (3136→128) + ReLU + Dropout
    ↓
Dense (128→10)
    ↓
Output (10 classes)
```

**期待される結果:**
- 訓練精度: ~99.5%
- テスト精度: ~99.0-99.3%
- 訓練時間: 10-15分 (CPU)

**CNNの利点:**
- 空間構造を保持
- 局所的なパターンを学習 (エッジ、コーナー)
- 平行移動不変性
- パラメータ数が少ない

**学べること:**
- 畳み込み層の仕組み
- Feature mapの可視化
- Batch Normalization
- Data Augmentation
- Learning Rate Scheduling

### 3. `experiment_builder.py` - インタラクティブ実験

**対象**: 実験したい方

ハイパーパラメータをインタラクティブに調整して、効果を理解。

**機能:**
- カスタムアーキテクチャ構築
- ハイパーパラメータ調整
- 複数実験の比較
- 結果の自動保存

**実験できること:**
- 層の数とサイズ
- 活性化関数 (ReLU, Tanh, Sigmoid)
- Dropout率
- 学習率
- バッチサイズ
- Optimizer (Adam, SGD, RMSprop)
- Data Augmentation

---

## 重要な概念

### 損失関数 (Loss Function)

**CrossEntropyLoss**: 分類問題で使用。予測と正解の差を測定。

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(predictions, true_labels)
```

値が小さいほど良い予測。

### Optimizer

モデルの重みを更新して損失を最小化。

**Adam (推奨)**:
- 適応的学習率
- モーメンタム
- 最も汎用的

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**SGD**:
- シンプル
- 大きなバッチサイズで効果的

**RMSprop**:
- RNNで人気

### Dropout

訓練中にランダムにニューロンを無効化。

**目的**: 過学習 (overfitting) を防ぐ

```python
nn.Dropout(0.2)  # 20%のニューロンを無効化
```

**いつ使うか**:
- 訓練精度 >> テスト精度の場合
- データが少ない場合

### Batch Normalization

各層の出力を正規化。

**効果**:
- 訓練の安定化
- 高い学習率が使える
- 一部のRegularization効果

```python
nn.BatchNorm2d(num_channels)
```

### Data Augmentation

訓練データを人工的に増やす。

```python
transforms.Compose([
    transforms.RandomRotation(10),      # ±10度回転
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 平行移動
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

**効果**: 汎化性能の向上

---

## 実験例

### 実験1: アーキテクチャの比較

**目的**: 層の数と精度の関係を理解

```bash
python experiment_builder.py
# Option 2 (Predefined experiments) を選択
```

**結果例**:
| アーキテクチャ | パラメータ数 | テスト精度 |
|----------------|-------------|-----------|
| [784, 64, 10] | ~50K | ~96.5% |
| [784, 128, 64, 10] | ~100K | ~97.8% |
| [784, 256, 128, 64, 10] | ~230K | ~98.1% |
| [784, 512, 256, 128, 64, 10] | ~660K | ~98.2% |

**結論**: ある程度以降は、深くしても精度向上は限定的。

### 実験2: 学習率の影響

**目的**: 学習率が訓練に与える影響を理解

| 学習率 | 結果 |
|--------|------|
| 0.0001 | 遅い収束、15エポックで95% |
| 0.001 | 良い収束、10エポックで97.8% |
| 0.01 | 不安定、振動 |
| 0.1 | 発散、学習しない |

**結論**: 0.001が最適。大きすぎると不安定、小さすぎると遅い。

### 実験3: Dropoutの効果

**データ**: 訓練データを10%に削減 (過学習しやすい状況)

| Dropout率 | 訓練精度 | テスト精度 | 過学習度 |
|-----------|----------|-----------|---------|
| 0.0 | 99.5% | 92.1% | 7.4% |
| 0.2 | 98.2% | 94.5% | 3.7% |
| 0.5 | 96.1% | 95.2% | 0.9% |

**結論**: Dropoutは過学習を効果的に防ぐ。

### 実験4: Data Augmentationの効果

**設定**: [784, 128, 64, 10], Dropout 0.2

| Augmentation | テスト精度 |
|--------------|-----------|
| なし | 97.8% |
| あり | 98.3% |

**結論**: わずかだが精度向上。データが少ない場合により効果的。

---

## CNNの可視化

### Feature Mapsの可視化

```bash
python mnist_cnn.py
# 選択: 1 (Basic CNN)
# 訓練後、feature_maps.png が生成される
```

**見方**:
- 最初の畳み込み層の出力
- エッジ、コーナー、線などを検出
- 各フィルタが異なる特徴を学習

### 学習済みフィルタの可視化

```bash
# cnn訓練後、conv_filters.png が生成される
```

**見方**:
- 3x3の小さなフィルタ
- エッジ検出、ブラー、シャープニングなど
- 自動的に有用な特徴を学習

---

## トラブルシューティング

### 問題1: 精度が10%前後で停滞

**原因**: モデルが全て同じクラスを予測している

**解決策**:
1. 学習率を確認 (0.001推奨)
2. 重みの初期化を確認
3. Optimizerを変更 (SGD → Adam)

### 問題2: 訓練精度 >> テスト精度 (過学習)

**原因**: モデルが訓練データに特化しすぎ

**解決策**:
1. Dropoutを増やす (0.2 → 0.5)
2. Data Augmentationを追加
3. モデルを小さくする
4. 訓練データを増やす

### 問題3: 訓練が非常に遅い

**原因**: CPUで実行、またはバッチサイズが小さい

**解決策**:
1. GPUを使用 (自動検出される)
2. バッチサイズを増やす (64 → 128)
3. エポック数を減らす (10 → 5)

### 問題4: 損失が減らない

**原因**: 学習率が不適切、またはモデルが複雑すぎる

**解決策**:
1. 学習率を変更 (0.001 ↔ 0.0001)
2. モデルをシンプルに
3. バッチサイズを変更
4. データの正規化を確認

---

## パフォーマンス比較

### モデル精度

| モデル | パラメータ数 | テスト精度 | 訓練時間 (CPU) |
|--------|-------------|-----------|---------------|
| Simple NN | ~100K | 97-98% | 5-10分 |
| Basic CNN | ~100K | 99.0-99.3% | 10-15分 |
| Improved CNN | ~200K | 99.3-99.5% | 15-20分 |

### 最先端との比較

MNIST人間の精度: ~97.5%
現在の最高記録: 99.79% (大規模アンサンブル)

このコードで達成可能: **99.3-99.5%** (単一モデル)

---

## 次のステップ

### レベル1: MNIST完了後

1. **Fashion-MNIST** - より難しい画像分類
2. **CIFAR-10** - カラー画像、10クラス
3. **Experiment 02** - 感情分析 (NLP入門)

### レベル2: 応用

1. **Transfer Learning (Exp 03)** - 事前学習モデルの活用
2. **Custom Architecture (Exp 05)** - モバイル向け軽量化
3. **Reinforcement Learning (Exp 08)** - ゲームAI

### レベル3: 実践

1. **Computer Vision (Exp 07)** - 実用的な画像認識
2. **Domain Specific AI (Exp 06)** - マルチモーダルAI
3. **Autonomous Testing (Exp 09)** - 自動テスト

---

## 参考資料

### オンラインコース
- **PyTorch公式チュートリアル**: https://pytorch.org/tutorials/
- **Deep Learning Specialization (Coursera)**: Andrew Ng
- **Fast.ai**: 実践的なディープラーニング

### 書籍
- "Deep Learning" by Ian Goodfellow
- "Neural Networks and Deep Learning" by Michael Nielsen (無料)

### 論文
- Original MNIST Paper: http://yann.lecun.com/exdb/mnist/
- Batch Normalization: https://arxiv.org/abs/1502.03167
- Dropout: https://jmlr.org/papers/v15/srivastava14a.html

---

## よくある質問

**Q: なぜMNISTから始めるのか?**

A: MNIST は「Hello World」のような存在です。
- データが小さい (訓練時間が短い)
- 理解しやすい (手書き数字)
- 基礎が全て含まれる
- すぐに結果が出る (モチベーション維持)

**Q: CNNと全結合層の違いは?**

A: CNNは画像の空間構造を保持します。
- 全結合: 784個の独立したピクセル
- CNN: 28x28の2次元構造

結果、CNNは「隣接ピクセルの関係」を学習可能。

**Q: 99%以上の精度が必要?**

A: 用途次第です。
- 学習目的: 97%で十分
- 実用: 99%+が望ましい
- クリティカル (医療等): 99.9%+必須

**Q: GPUがないと学習できない?**

A: MNISTはCPUで十分です。
- CPU: 10-15分
- GPU: 2-3分

より大きなデータセットではGPUが必須。

---

**楽しく学びましょう! 🎓🚀**
