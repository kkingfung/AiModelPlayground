"""
Experience Replay Buffer

DQNの重要な構成要素.
過去の経験を保存・再利用することで学習を効率化・安定化.

主な利点:
- データの相関を減らす（連続した経験はi.i.d.ではない）
- サンプル効率向上（1つの経験を複数回使用）
- レアな経験を保持（報酬の高いエピソードなど）

使い方:
    from experience_replay import ReplayBuffer, PrioritizedReplayBuffer

    # 基本的なReplay Buffer
    buffer = ReplayBuffer(capacity=10000)
    buffer.push(state, action, reward, next_state, done)
    batch = buffer.sample(batch_size=64)

    # Prioritized Experience Replay
    per_buffer = PrioritizedReplayBuffer(capacity=10000)
    per_buffer.push(state, action, reward, next_state, done)
    batch, indices, weights = per_buffer.sample(batch_size=64)
    per_buffer.update_priorities(indices, td_errors)
"""

import random
import numpy as np
from collections import deque, namedtuple
from typing import List, Tuple, Optional


# 経験を保存するための名前付きタプル
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    標準的なExperience Replay Buffer.

    固定サイズのdequeで経験を保存し、ランダムサンプリング.
    最も基本的で実装が簡単.
    """

    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: バッファの最大サイズ（古い経験から削除）
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        経験を追加.

        Args:
            state: 現在の状態
            action: 実行した行動
            reward: 得られた報酬
            next_state: 次の状態
            done: エピソード終了フラグ
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """
        ランダムにバッチをサンプリング.

        Args:
            batch_size: サンプル数

        Returns:
            Transitionのリスト
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """バッファ内の経験数."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """学習可能か（バッチサイズ分の経験があるか）."""
        return len(self.buffer) >= batch_size

    def clear(self):
        """バッファをクリア."""
        self.buffer.clear()

    def get_statistics(self) -> dict:
        """バッファの統計情報."""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'usage': 0.0,
                'avg_reward': 0.0,
                'positive_ratio': 0.0
            }

        rewards = [t.reward for t in self.buffer]

        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'usage': len(self.buffer) / self.capacity,
            'avg_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'positive_ratio': sum(1 for r in rewards if r > 0) / len(rewards)
        }


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER).

    TD誤差が大きい経験を優先的にサンプリング.
    より効率的な学習が可能.

    参考: Schaul et al. 2016 "Prioritized Experience Replay"
    https://arxiv.org/abs/1511.05952
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 0.01
    ):
        """
        Args:
            capacity: バッファの最大サイズ
            alpha: 優先度の指数（0=uniform, 1=full prioritization）
            beta: Importance Sampling補正の指数（初期値）
            beta_increment: βの増加量（徐々に1.0に近づける）
            epsilon: 優先度の最小値（0除算防止）
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: Optional[float] = None
    ):
        """
        経験を追加.

        Args:
            state: 現在の状態
            action: 実行した行動
            reward: 得られた報酬
            next_state: 次の状態
            done: エピソード終了フラグ
            td_error: TD誤差（Noneの場合は最大優先度を使用）
        """
        # 新しい経験の優先度
        if td_error is not None:
            priority = (abs(td_error) + self.epsilon) ** self.alpha
        else:
            # TD誤差が不明な場合は最大優先度を使用（確実にサンプリングされる）
            max_priority = self.priorities.max() if self.size > 0 else 1.0
            priority = max_priority

        # バッファに追加
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = Transition(state, action, reward, next_state, done)

        self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        優先度に基づいてサンプリング.

        Args:
            batch_size: サンプル数

        Returns:
            (transitions, indices, weights):
                - transitions: Transitionのリスト
                - indices: サンプルのインデックス（優先度更新用）
                - weights: Importance Sampling補正ウェイト
        """
        # 優先度に基づくサンプリング確率
        priorities = self.priorities[:self.size]
        probabilities = priorities / priorities.sum()

        # サンプリング
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)

        # Importance Sampling補正ウェイト
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 正規化（最大値を1にする）

        # β を徐々に増加（学習後期にIS補正を強める）
        self.beta = min(1.0, self.beta + self.beta_increment)

        # サンプルを取得
        transitions = [self.buffer[idx] for idx in indices]

        return transitions, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        優先度を更新.

        学習後にTD誤差に基づいて優先度を更新.

        Args:
            indices: 更新する経験のインデックス
            td_errors: TD誤差（絶対値）
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self) -> int:
        """バッファ内の経験数."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """学習可能か."""
        return self.size >= batch_size

    def clear(self):
        """バッファをクリア."""
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def get_statistics(self) -> dict:
        """バッファの統計情報."""
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'usage': 0.0,
                'beta': self.beta
            }

        rewards = [t.reward for t in self.buffer[:self.size]]
        priorities = self.priorities[:self.size]

        return {
            'size': self.size,
            'capacity': self.capacity,
            'usage': self.size / self.capacity,
            'beta': self.beta,
            'avg_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'avg_priority': np.mean(priorities),
            'priority_std': np.std(priorities),
            'max_priority': priorities.max()
        }


class HindsightReplayBuffer:
    """
    Hindsight Experience Replay (HER).

    失敗した経験を「別のゴール」として再解釈することで学習.
    スパース報酬環境（ゴール到達時のみ報酬）で有効.

    参考: Andrychowicz et al. 2017 "Hindsight Experience Replay"
    https://arxiv.org/abs/1707.01495

    例:
        ゴール: (5, 5) に到達
        実際の軌跡: (0,0) → (1,1) → (2,2) [失敗、報酬0]

        HER:
        「(2,2)に到達する」を新しいゴールとして再解釈
        → 報酬1を与えて学習（成功経験として）
    """

    def __init__(self, capacity: int = 10000, her_ratio: float = 0.8):
        """
        Args:
            capacity: バッファの最大サイズ
            her_ratio: HERで生成する経験の割合（0.8 = 80%がHER経験）
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.her_ratio = her_ratio

        # エピソードバッファ（1エピソード分の経験を一時保存）
        self.episode_buffer = []

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        goal: Optional[np.ndarray] = None,
        achieved_goal: Optional[np.ndarray] = None
    ):
        """
        経験を一時保存（エピソード終了時にHER処理）.

        Args:
            state: 現在の状態
            action: 実行した行動
            reward: 得られた報酬
            next_state: 次の状態
            done: エピソード終了フラグ
            goal: 目標（ゴール状態）
            achieved_goal: 実際に到達した状態
        """
        self.episode_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'goal': goal,
            'achieved_goal': achieved_goal
        })

        # エピソード終了時にバッファに追加
        if done:
            self._process_episode()
            self.episode_buffer = []

    def _process_episode(self):
        """
        エピソード終了時にHER処理してバッファに追加.

        元の経験 + HERで生成した経験をバッファに追加.
        """
        if len(self.episode_buffer) == 0:
            return

        # 元の経験を追加（オリジナルゴール）
        for exp in self.episode_buffer:
            self.buffer.append(Transition(
                exp['state'],
                exp['action'],
                exp['reward'],
                exp['next_state'],
                exp['done']
            ))

        # HER経験を生成
        n_her_samples = int(len(self.episode_buffer) * self.her_ratio)

        for _ in range(n_her_samples):
            # ランダムにステップを選択
            t = random.randint(0, len(self.episode_buffer) - 1)

            # 未来のステップからゴールを選択（Future strategy）
            future_t = random.randint(t, len(self.episode_buffer) - 1)
            new_goal = self.episode_buffer[future_t]['achieved_goal']

            # 新しいゴールで報酬を再計算
            exp = self.episode_buffer[t]
            achieved = exp['achieved_goal']

            # ゴールに到達したかチェック
            if np.array_equal(achieved, new_goal):
                new_reward = 1.0  # 成功報酬
            else:
                new_reward = 0.0

            # HER経験を追加
            self.buffer.append(Transition(
                exp['state'],
                exp['action'],
                new_reward,
                exp['next_state'],
                exp['done']
            ))

    def sample(self, batch_size: int) -> List[Transition]:
        """ランダムにバッチをサンプリング."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


def demo_replay_buffers():
    """デモ: 各Replay Bufferの使い方."""

    print("=" * 60)
    print("EXPERIENCE REPLAY BUFFER DEMO")
    print("=" * 60)

    # サンプルデータ生成
    def generate_sample_experience():
        state = np.random.randn(4)
        action = random.randint(0, 1)
        reward = random.uniform(-1, 1)
        next_state = np.random.randn(4)
        done = random.random() < 0.1
        return state, action, reward, next_state, done

    # ===== 1. Standard Replay Buffer =====
    print("\n1. Standard Replay Buffer")
    print("-" * 60)

    buffer = ReplayBuffer(capacity=100)

    # 経験を追加
    for _ in range(150):  # capacityより多く追加
        buffer.push(*generate_sample_experience())

    print(f"Buffer size: {len(buffer)} (capacity: {buffer.capacity})")

    # サンプリング
    batch = buffer.sample(batch_size=10)
    print(f"Sampled batch size: {len(batch)}")

    # 統計
    stats = buffer.get_statistics()
    print(f"Statistics: {stats}")

    # ===== 2. Prioritized Experience Replay =====
    print("\n2. Prioritized Experience Replay")
    print("-" * 60)

    per_buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)

    # 経験を追加
    for _ in range(150):
        state, action, reward, next_state, done = generate_sample_experience()
        # TD誤差を模擬（報酬が大きいほど高い優先度）
        td_error = abs(reward)
        per_buffer.push(state, action, reward, next_state, done, td_error)

    print(f"PER Buffer size: {len(per_buffer)}")

    # サンプリング
    transitions, indices, weights = per_buffer.sample(batch_size=10)
    print(f"Sampled batch size: {len(transitions)}")
    print(f"IS weights (first 5): {weights[:5]}")

    # 優先度更新（模擬）
    td_errors = np.random.rand(len(indices))
    per_buffer.update_priorities(indices, td_errors)

    # 統計
    stats = per_buffer.get_statistics()
    print(f"Statistics: {stats}")

    # ===== 3. Hindsight Experience Replay =====
    print("\n3. Hindsight Experience Replay")
    print("-" * 60)

    her_buffer = HindsightReplayBuffer(capacity=100, her_ratio=0.8)

    # エピソードを模擬
    for episode in range(10):
        for step in range(10):
            state = np.random.randn(4)
            action = random.randint(0, 1)
            reward = 0.0 if step < 9 else 1.0  # 最後のステップのみ報酬
            next_state = np.random.randn(4)
            done = step == 9

            goal = np.array([1.0, 1.0])
            achieved_goal = next_state[:2]

            her_buffer.push(state, action, reward, next_state, done, goal, achieved_goal)

    print(f"HER Buffer size: {len(her_buffer)}")

    # サンプリング
    batch = her_buffer.sample(batch_size=10)
    print(f"Sampled batch size: {len(batch)}")
    print(f"Average reward in buffer: {np.mean([t.reward for t in her_buffer.buffer]):.3f}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo_replay_buffers()
