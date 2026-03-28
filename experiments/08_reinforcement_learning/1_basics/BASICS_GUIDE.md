# RL Basics Guide

## 強化学習の基礎

このガイドでは、Q-LearningとDQNという2つの基本的なRLアルゴリズムを学びます。

---

## 📚 目次

1. [Q-Learning（表形式）](#q-learning)
2. [Deep Q-Network (DQN)](#deep-q-network-dqn)
3. [Experience Replay](#experience-replay)
4. [実践的な使い方](#実践的な使い方)
5. [トラブルシューティング](#トラブルシューティング)

---

## Q-Learning

### 概要

**Q-Learning**は最もシンプルなRLアルゴリズム。
状態と行動の全ての組み合わせをテーブル（Q-table）で管理。

**使用例**:
- グリッドワールド（迷路）
- 小規模なゲーム（Tic-Tac-Toe等）
- プロトタイピング

### 仕組み

#### Q値の更新式

```
Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
```

- `Q(s, a)`: 状態sで行動aを取る価値
- `α` (alpha): 学習率（0.1が一般的）
- `γ` (gamma): 割引率（0.99が一般的）
- `r`: 即座の報酬
- `s'`: 次の状態
- `max Q(s', a')`: 次の状態での最大Q値

#### ε-greedy探索

```python
if random() < epsilon:
    action = random_action()  # 探索
else:
    action = argmax(Q[state])  # 活用
```

- `epsilon`: 探索率（1.0 → 0.01に減衰）
- 初期は探索、後期は活用

### 使い方

```bash
# グリッドワールドで学習
python q_learning.py --episodes 1000 --render

# パラメータ調整
python q_learning.py \
    --episodes 2000 \
    --lr 0.1 \
    --gamma 0.99 \
    --epsilon 1.0

# 学習済みポリシーで実行
python q_learning.py --play --load q_table.pkl
```

### コード例

```python
from q_learning import GridWorld, QLearningAgent

# 環境とエージェント作成
env = GridWorld(size=5)
agent = QLearningAgent(
    n_actions=4,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0
)

# 学習
agent.train(env, n_episodes=1000)

# 保存
agent.save("q_table.pkl")

# ポリシー可視化
agent.visualize_policy(env)
# Output:
# S → → ↓ ↓
# ↓ X → ↓ ↓
# ↓ ↓ X → ↓
# ↓ → ↑ X ↓
# → → → → G

# プレイ
agent.play(env, render=True)
```

### 学習曲線の例

```
Episode 0:    Reward = -50 (ランダムに動く)
Episode 100:  Reward = -20 (ゴールに近づく)
Episode 500:  Reward = 5   (ゴール到達！)
Episode 1000: Reward = 8   (最適経路発見)
```

### Q-Learningの限界

❌ **状態空間が大きい環境では使えない**
- 例: 囲碁（3^361通りの状態）
- 例: Atari（210x160x3ピクセル）

✅ **解決策**: Deep Q-Network (DQN)

---

## Deep Q-Network (DQN)

### 概要

**DQN**は、Q-tableをニューラルネットワークに置き換えたアルゴリズム。
大規模・連続的な状態空間でも学習可能。

**ブレークスルー**:
- 2013年DeepMindがAtariゲームで人間超えを達成
- 論文: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

**主要技術**:
1. **Experience Replay**: 過去の経験を再利用
2. **Target Network**: 学習を安定化
3. **CNN**: 画像から特徴抽出（Atari等）

### アーキテクチャ

#### 標準的なDQN（CartPole等）

```
Input: State (4次元) → [128] → [128] → Output: Q-values (2次元)
```

#### CNNベースDQN（Atari等）

```
Input: Image (84x84x4)
  ↓
Conv2D(32, 8x8, stride=4) → ReLU
  ↓
Conv2D(64, 4x4, stride=2) → ReLU
  ↓
Conv2D(64, 3x3, stride=1) → ReLU
  ↓
Flatten → FC(512) → ReLU
  ↓
Output: Q-values (action_size)
```

### 使い方

```bash
# CartPole環境で学習
python dqn.py --env CartPole-v1 --episodes 500

# より複雑な環境
python dqn.py --env LunarLander-v2 --episodes 1000

# Atariゲーム（CNN使用）
python dqn.py --env ALE/Breakout-v5 --episodes 2000 --use-cnn

# パラメータ調整
python dqn.py \
    --env CartPole-v1 \
    --episodes 500 \
    --lr 0.001 \
    --gamma 0.99 \
    --buffer-size 10000 \
    --batch-size 64 \
    --target-update 10

# 学習済みモデルで実行
python dqn.py --env CartPole-v1 --play --model checkpoints/dqn_model.pth
```

### コード例

```python
from dqn import DQNAgent
import gymnasium as gym

# 環境作成
env = gym.make("CartPole-v1")

# エージェント作成
agent = DQNAgent(
    state_size=4,
    action_size=2,
    learning_rate=0.001,
    discount_factor=0.99,
    buffer_size=10000,
    batch_size=64
)

# 学習
agent.train(env, n_episodes=500)

# 保存
agent.save("checkpoints/dqn_cartpole.pth")

# プロット
agent.plot_training()

# 実行
agent.play(env, n_episodes=5, render=True)
```

### 学習曲線の例（CartPole-v1）

```
Episode 0:    Reward = 10  (すぐに倒れる)
Episode 50:   Reward = 25
Episode 100:  Reward = 50
Episode 200:  Reward = 150
Episode 300:  Reward = 300
Episode 500:  Reward = 500 ✓ Solved! (平均500 = 完璧)
```

### ハイパーパラメータ調整

| Parameter | 推奨値 | 説明 |
|-----------|-------|------|
| learning_rate | 0.0001-0.001 | 大きすぎると発散、小さすぎると学習遅い |
| discount_factor | 0.99 | 未来の報酬をどれだけ重視するか |
| epsilon | 1.0 → 0.01 | 探索率（徐々に減衰） |
| buffer_size | 10000-100000 | 大きいほど多様な経験、メモリ使用量↑ |
| batch_size | 32-128 | 大きいほど安定、計算コスト↑ |
| target_update | 10-100 | 小さいほど頻繁に更新、不安定 |

---

## Experience Replay

### 概要

Experience Replayは、過去の経験`(s, a, r, s', done)`をバッファに保存し、
ランダムサンプリングで学習する手法。

### なぜ必要？

❌ **問題**: 連続した経験は相関が高い
```
時刻t:   (state1, action1, reward1, state2, done=False)
時刻t+1: (state2, action2, reward2, state3, done=False)
           ↑ state2が重複 → データの相関
```

✅ **解決**: ランダムサンプリングで相関を減らす
```
Buffer: [経験1, 経験2, ..., 経験10000]
         ↓ ランダムサンプリング
Batch: [経験42, 経験1337, 経験999, ...]
```

### 種類

#### 1. Standard Replay Buffer

最もシンプル。固定サイズのdequeで先入れ先出し。

```python
from experience_replay import ReplayBuffer

buffer = ReplayBuffer(capacity=10000)

# 経験を追加
buffer.push(state, action, reward, next_state, done)

# サンプリング
batch = buffer.sample(batch_size=64)

# 統計
stats = buffer.get_statistics()
print(stats)
# {'size': 10000, 'capacity': 10000, 'usage': 1.0, 'avg_reward': 0.5}
```

#### 2. Prioritized Experience Replay (PER)

TD誤差が大きい経験を優先的にサンプリング。
より効率的な学習が可能。

**特徴**:
- 重要な経験（TD誤差大）を多く学習
- Importance Sampling補正で偏りを修正

```python
from experience_replay import PrioritizedReplayBuffer

per = PrioritizedReplayBuffer(
    capacity=10000,
    alpha=0.6,  # 優先度の指数
    beta=0.4    # IS補正の指数（徐々に1.0へ）
)

# 経験を追加（TD誤差付き）
per.push(state, action, reward, next_state, done, td_error=0.5)

# サンプリング
transitions, indices, weights = per.sample(batch_size=64)

# 学習後、優先度を更新
per.update_priorities(indices, new_td_errors)
```

#### 3. Hindsight Experience Replay (HER)

失敗した経験を「別のゴール」として再解釈。
スパース報酬環境（ゴール到達時のみ報酬）で有効。

**例**:
```
目標ゴール: (5, 5)
実際の軌跡: (0,0) → (1,1) → (2,2) [失敗、報酬0]

HER再解釈:
「(2,2)に到達する」を新しいゴールとして学習
→ 報酬1を与える（成功経験として）
```

```python
from experience_replay import HindsightReplayBuffer

her = HindsightReplayBuffer(
    capacity=10000,
    her_ratio=0.8  # 80%をHER経験として生成
)

# 経験を追加（ゴール情報付き）
her.push(state, action, reward, next_state, done,
         goal=target_goal, achieved_goal=actual_goal)

# エピソード終了時に自動でHER経験を生成
```

### デモ実行

```bash
# 各Replay Bufferのデモ
python experience_replay.py
```

**出力例**:
```
============================================================
EXPERIENCE REPLAY BUFFER DEMO
============================================================

1. Standard Replay Buffer
------------------------------------------------------------
Buffer size: 100 (capacity: 100)
Sampled batch size: 10
Statistics: {'size': 100, 'usage': 1.0, 'avg_reward': 0.02}

2. Prioritized Experience Replay
------------------------------------------------------------
PER Buffer size: 100
Sampled batch size: 10
IS weights (first 5): [0.82 0.91 1.00 0.75 0.88]

3. Hindsight Experience Replay
------------------------------------------------------------
HER Buffer size: 200
Average reward in buffer: 0.400
```

---

## 実践的な使い方

### 1. グリッドワールドで学習（Q-Learning）

```bash
# 学習
python q_learning.py --episodes 1000 --render

# ポリシー可視化
python q_learning.py --visualize --load q_table.pkl
```

**何が起こる？**
- エージェントが開始地点(0,0)からゴール(4,4)への最短経路を学習
- 障害物を避けるようになる
- 学習曲線とQ値ヒートマップが保存される

### 2. CartPoleで学習（DQN）

```bash
# 学習
python dqn.py --env CartPole-v1 --episodes 500

# 学習済みモデルで実行
python dqn.py --env CartPole-v1 --play --model checkpoints/dqn_model.pth
```

**何が起こる？**
- 棒を立て続けることを学習
- 最初は10-20ステップで倒れるが、徐々に長く保つようになる
- 500エピソードで平均500ステップ（完璧）に到達

### 3. Experience Replayの比較

```python
import gymnasium as gym
from dqn import DQNAgent
from experience_replay import ReplayBuffer, PrioritizedReplayBuffer

env = gym.make("CartPole-v1")

# Standard Replay
agent1 = DQNAgent(state_size=4, action_size=2, buffer_size=10000)
agent1.train(env, n_episodes=500)

# Prioritized Replay
# (DQNAgentにPERを統合する必要あり - 実装例は省略)
```

---

## トラブルシューティング

### Q-Learning

#### 問題: 学習しない（報酬が上がらない）

**原因**:
- 学習率αが小さすぎる
- 探索率εが減衰しすぎ（早期に活用のみになる）
- 報酬設計が悪い

**解決策**:
```bash
# 学習率を上げる
python q_learning.py --lr 0.2

# ε減衰を遅くする
python q_learning.py --epsilon-decay 0.999
```

#### 問題: 学習が不安定

**原因**:
- 学習率αが大きすぎる

**解決策**:
```bash
python q_learning.py --lr 0.05
```

### DQN

#### 問題: 学習が発散（報酬が暴れる）

**原因**:
- 学習率が高すぎる
- Target Networkの更新頻度が高すぎる

**解決策**:
```bash
# 学習率を下げる
python dqn.py --lr 0.0001

# Target Network更新を減らす
python dqn.py --target-update 20
```

#### 問題: 学習が遅い

**原因**:
- バッファサイズが小さい
- バッチサイズが小さい
- 学習率が低すぎる

**解決策**:
```bash
python dqn.py --buffer-size 50000 --batch-size 128 --lr 0.001
```

#### 問題: メモリ不足

**原因**:
- バッファサイズが大きすぎる

**解決策**:
```bash
python dqn.py --buffer-size 5000
```

#### 問題: GPUが使われない

**確認**:
```python
import torch
print(torch.cuda.is_available())  # True?
```

**解決策**:
- PyTorch GPUバージョンをインストール
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## まとめ

### Q-Learning vs DQN

| 特徴 | Q-Learning | DQN |
|------|-----------|-----|
| 状態表現 | 表形式（Q-table） | ニューラルネット |
| 適用範囲 | 小規模・離散 | 大規模・連続的 |
| メモリ | 小 | 大 |
| 学習速度 | 速い（小規模） | 遅い（大規模でも動く） |
| 実装難易度 | 簡単 | 中程度 |

### 次のステップ

1. ✅ Q-Learning実装 → グリッドワールドで学習
2. ✅ DQN実装 → CartPoleで学習
3. **次**: カスタムゲーム環境の構築（`2_environments/`）
4. **発展**: PPO, A3C等の高度なアルゴリズム（`3_algorithms/`）

---

## 参考資料

- **Q-Learning**: Watkins, 1989 "Learning from Delayed Rewards"
- **DQN**: Mnih et al., 2013 "Playing Atari with Deep Reinforcement Learning"
- **PER**: Schaul et al., 2016 "Prioritized Experience Replay"
- **HER**: Andrychowicz et al., 2017 "Hindsight Experience Replay"

---

**さあ、RLの基礎を習得しましょう！ 🎮🤖🚀**
