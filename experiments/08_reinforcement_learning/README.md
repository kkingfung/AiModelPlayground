# Experiment 08: Reinforcement Learning for Game AI

## 概要 (Overview)

ゲームAIのための強化学習（Reinforcement Learning）を実装:
- **Q-Learning & DQN**: 基礎的なRL手法
- **PPO & A3C**: 最先端アルゴリズム
- **カスタムゲーム環境**: 独自ゲームでAI訓練
- **実践的な応用**: NPC行動、難易度調整、プレイテスト自動化

## 🎯 学習目標

### 基礎
- ✅ Q-Learning（表形式）
- ✅ Deep Q-Network (DQN)
- ✅ Experience Replay
- ✅ Target Network

### 発展
- ✅ Policy Gradient (REINFORCE)
- ✅ Actor-Critic (A2C/A3C)
- ✅ Proximal Policy Optimization (PPO)
- ✅ Soft Actor-Critic (SAC)

### ゲーム応用
- ✅ カスタムゲーム環境構築
- ✅ NPC行動学習
- ✅ 難易度自動調整
- ✅ プレイテスト自動化

## 📁 ディレクトリ構造

```
08_reinforcement_learning/
├── README.md
├── requirements.txt
│
├── 1_basics/
│   ├── q_learning.py              # 表形式Q-Learning
│   ├── dqn.py                     # Deep Q-Network
│   ├── experience_replay.py       # リプレイバッファ
│   └── BASICS_GUIDE.md
│
├── 2_environments/
│   ├── grid_world.py              # グリッドワールド環境
│   ├── platformer_env.py          # 2Dプラットフォーマー
│   ├── combat_env.py              # 戦闘シミュレーション
│   ├── gym_wrapper.py             # OpenAI Gym互換ラッパー
│   └── ENV_GUIDE.md
│
├── 3_algorithms/
│   ├── policy_gradient.py         # REINFORCE
│   ├── actor_critic.py            # A2C/A3C
│   ├── ppo.py                     # PPO（最も実用的）
│   ├── sac.py                     # SAC（連続制御）
│   └── ALGORITHMS_GUIDE.md
│
├── 4_applications/
│   ├── npc_behavior.py            # NPC行動学習
│   ├── difficulty_tuning.py       # 難易度自動調整
│   ├── playtester_bot.py          # プレイテストボット
│   ├── curriculum_learning.py     # カリキュラム学習
│   └── APPLICATIONS_GUIDE.md
│
└── trained_agents/                # 学習済みエージェント
    ├── platformer_agent.pth
    ├── combat_agent.pth
    └── ...
```

## 🚀 クイックスタート

### 1. 依存関係インストール

```bash
cd experiments/08_reinforcement_learning
pip install -r requirements.txt
```

### 2. 基礎: Q-Learning（5分）

```bash
# グリッドワールドでQ-Learningを実行
python 1_basics/q_learning.py \
    --episodes 1000 \
    --render
```

**何が起こる?**
- エージェントがゴールへの最短経路を学習
- 最初はランダムに動き、徐々に賢くなる
- 視覚化でリアルタイム学習を観察

### 3. Deep Q-Network (DQN)

```bash
# Atariゲームで学習
python 1_basics/dqn.py \
    --env CartPole-v1 \
    --episodes 500 \
    --render
```

### 4. カスタムゲーム環境

```bash
# 2Dプラットフォーマーで学習
python 2_environments/platformer_env.py \
    --train \
    --algorithm ppo \
    --episodes 5000
```

### 5. 学習済みエージェントで遊ぶ

```bash
# 学習済みモデルをロードして実行
python 3_algorithms/ppo.py \
    --play \
    --model trained_agents/platformer_agent.pth \
    --render
```

## 💡 強化学習の基礎

### RL用語集

- **Agent（エージェント）**: 学習するAI
- **Environment（環境）**: ゲーム世界
- **State（状態）**: 現在のゲーム状態
- **Action（行動）**: エージェントの行動選択
- **Reward（報酬）**: 行動の良し悪しを示す数値
- **Policy（方策）**: 状態→行動の写像（エージェントの戦略）

### RLサイクル

```
┌─────────┐  action   ┌─────────────┐
│  Agent  │ ────────> │ Environment │
│         │           │   (Game)    │
└─────────┘           └─────────────┘
     ^                      │
     │   state, reward      │
     └──────────────────────┘
```

### 報酬設計の例

#### プラットフォーマーゲーム
```python
reward = 0

# ゴール到達
if reached_goal:
    reward += 1000

# 死亡
if dead:
    reward -= 100

# 時間ペナルティ（早くゴールするよう促す）
reward -= 1

# 前進ボーナス
reward += (new_x - old_x) * 10
```

#### 戦闘AI
```python
reward = 0

# ダメージを与えた
reward += damage_dealt * 10

# ダメージを受けた
reward -= damage_taken * 5

# 勝利
if won:
    reward += 500

# 敗北
if lost:
    reward -= 200

# HP維持ボーナス
reward += current_hp * 0.1
```

## 🎮 実用例

### Example 1: NPCパトロール行動

```python
from npc_behavior import NPCAgent

# 環境作成
env = PatrolEnvironment(map_size=(10, 10))

# エージェント作成
agent = NPCAgent(state_size=env.state_size, action_size=env.action_size)

# 学習
agent.train(env, episodes=1000)

# 保存
agent.save("trained_agents/patrol_npc.pth")

# ゲームで使用
action = agent.get_action(current_state)
```

### Example 2: 動的難易度調整

```python
from difficulty_tuning import DifficultyAgent

# プレイヤースキルに基づいて敵の強さを調整
difficulty_agent = DifficultyAgent()

# プレイヤーデータ
player_skill = {
    "win_rate": 0.65,
    "avg_health": 0.4,
    "avg_time": 120
}

# 最適な難易度を取得
difficulty = difficulty_agent.get_optimal_difficulty(player_skill)
# Output: {"enemy_count": 5, "enemy_health": 1.2, "spawn_rate": 0.8}
```

### Example 3: プレイテスト自動化

```python
from playtester_bot import PlaytesterBot

# レベルテスト
bot = PlaytesterBot()
results = bot.test_level("levels/level_5.json", num_runs=100)

print(f"Completion rate: {results['completion_rate']}%")
print(f"Average time: {results['avg_time']}s")
print(f"Difficulty: {results['difficulty_rating']}")

# 問題箇所特定
if results['completion_rate'] < 50:
    print(f"Difficult sections: {results['hard_sections']}")
```

## 📊 アルゴリズム比較

| Algorithm | 学習速度 | 安定性 | メモリ | 用途 |
|-----------|---------|--------|--------|------|
| Q-Learning | 遅い | 高 | 小 | 離散・小規模 |
| DQN | 中 | 中 | 中 | 離散・中規模 |
| PPO | 速い | 高 | 大 | 汎用・実用的 |
| A3C | 非常に速い | 中 | 大 | 並列学習 |
| SAC | 速い | 高 | 大 | 連続制御 |

### 選び方ガイド

**初心者 / プロトタイプ**:
- Q-Learning（シンプル）
- DQN（汎用的）

**実用 / プロダクション**:
- **PPO**（最もバランスが良い）
- A3C（並列学習で高速）

**連続制御（移動、エイム）**:
- SAC
- TD3

## 🛠️ 技術スタック

### Core RL
- **PyTorch**: ニューラルネットワーク
- **Gym**: 環境インターフェース
- **Stable-Baselines3**: 実装済みアルゴリズム（参考用）
- **TensorBoard**: 学習可視化

### Game Integration
- **Pygame**: 2Dゲーム環境
- **Unity ML-Agents**: Unityとの統合（オプション）
- **Gym-Retro**: レトロゲーム環境

## 🎓 学習パス

### Week 1: 基礎（1週間）
1. Q-Learning実装
2. GridWorld環境で学習
3. DQNで複雑な環境に挑戦

### Week 2-3: 発展（2週間）
1. PPO実装
2. カスタムゲーム環境構築
3. NPC行動学習

### Week 4+: 応用（継続）
1. 難易度調整システム
2. プレイテスト自動化
3. プロダクション統合

## 📈 学習曲線の例

### CartPole-v1（バランスゲーム）
```
Episode 0:    Reward = 10
Episode 100:  Reward = 50
Episode 300:  Reward = 200
Episode 500:  Reward = 500 ✓ Solved!
```

### カスタムプラットフォーマー
```
Episode 0:     死亡（穴に落ちる）
Episode 500:   ゴールの半分まで到達
Episode 2000:  ゴール到達！
Episode 5000:  最適ルートで高速クリア
```

## 🚀 次のステップ

このエクスペリメントを完了したら:
- **Experiment 09**: Generative AI (Stable Diffusion)
- **Experiment 10**: Production ML Pipelines (MLOps)
- **Unity/Unreal統合**: 実際のゲームエンジンへ組み込み

## 📚 リソース

- **OpenAI Spinning Up**: https://spinningup.openai.com/
- **Stable-Baselines3 Docs**: https://stable-baselines3.readthedocs.io/
- **DeepMind RL Course**: https://www.deepmind.com/learning-resources
- **Unity ML-Agents**: https://github.com/Unity-Technologies/ml-agents

---

**強化学習でゲームAIを次のレベルへ！ 🎮🤖🚀**
