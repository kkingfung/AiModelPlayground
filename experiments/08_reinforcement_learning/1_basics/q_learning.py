"""
Q-Learning - 表形式強化学習

最もシンプルなRLアルゴリズム.
グリッドワールドなど離散的で小規模な環境に最適.

使い方:
    # 学習
    python q_learning.py --episodes 1000 --render

    # Q-テーブル可視化
    python q_learning.py --load q_table.npy --visualize
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import pickle
from collections import defaultdict


class GridWorld:
    """
    シンプルなグリッドワールド環境.

    エージェントが開始地点(S)からゴール(G)を目指す.
    - 障害物(X)は通過不可
    - ゴール到達で報酬+100
    - 各ステップで-1（早くゴールするよう促す）
    """

    def __init__(self, size: int = 5):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)

        # 障害物配置
        self.obstacles = {(1, 1), (2, 2), (3, 1)}

        self.reset()

    def reset(self) -> Tuple[int, int]:
        """環境をリセット."""
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        行動を実行.

        Args:
            action: 0=上, 1=右, 2=下, 3=左

        Returns:
            (next_state, reward, done)
        """
        # 移動方向
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上, 右, 下, 左
        move = moves[action]

        # 新しい位置
        new_pos = (
            self.agent_pos[0] + move[0],
            self.agent_pos[1] + move[1]
        )

        # 境界チェック
        if (0 <= new_pos[0] < self.size and
            0 <= new_pos[1] < self.size and
            new_pos not in self.obstacles):
            self.agent_pos = new_pos

        # 報酬計算
        if self.agent_pos == self.goal:
            reward = 100.0
            done = True
        else:
            reward = -1.0  # 時間ペナルティ
            done = False

        return self.agent_pos, reward, done

    def render(self):
        """環境を可視化."""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        # 障害物
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'

        # スタートとゴール
        grid[self.start[0]][self.start[1]] = 'S'
        grid[self.goal[0]][self.goal[1]] = 'G'

        # エージェント
        if self.agent_pos != self.start and self.agent_pos != self.goal:
            grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'

        # 表示
        for row in grid:
            print(' '.join(row))
        print()


class QLearningAgent:
    """
    Q-Learning エージェント.

    Q(s, a) = Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]

    - α (alpha): 学習率
    - γ (gamma): 割引率
    - ε (epsilon): 探索率
    """

    def __init__(
        self,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table（辞書形式）
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        # 統計
        self.training_rewards = []

    def get_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        行動選択（ε-greedy方策）.

        Args:
            state: 現在の状態
            training: 学習中かどうか

        Returns:
            選択された行動
        """
        if training and np.random.random() < self.epsilon:
            # 探索: ランダム行動
            return np.random.randint(self.n_actions)
        else:
            # 活用: Q値が最大の行動
            return np.argmax(self.q_table[state])

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool
    ):
        """
        Q-tableを更新.

        Q(s, a) ← Q(s, a) + α[r + γ max Q(s', a') - Q(s, a)]
        """
        current_q = self.q_table[state][action]

        if done:
            # 終端状態では次の状態の価値は0
            target_q = reward
        else:
            # Bellman方程式
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q

        # Q値更新
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)

    def train(
        self,
        env: GridWorld,
        n_episodes: int = 1000,
        render_every: int = 100,
        verbose: bool = True
    ):
        """
        学習ループ.

        Args:
            env: 環境
            n_episodes: エピソード数
            render_every: 描画間隔
            verbose: 詳細表示
        """
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0

            while True:
                # 行動選択
                action = self.get_action(state, training=True)

                # 環境ステップ
                next_state, reward, done = env.step(action)

                # Q-table更新
                self.update(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state
                steps += 1

                # 描画
                if render_every > 0 and episode % render_every == 0:
                    env.render()

                if done:
                    break

            # ε減衰（徐々に探索を減らす）
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # 統計記録
            self.training_rewards.append(total_reward)

            if verbose and episode % 100 == 0:
                avg_reward = np.mean(self.training_rewards[-100:])
                print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, Steps: {steps}")

        print(f"\n✓ Training completed! Final avg reward: {np.mean(self.training_rewards[-100:]):.2f}")

    def play(self, env: GridWorld, render: bool = True) -> float:
        """
        学習済みポリシーで実行.

        Args:
            env: 環境
            render: 描画するか

        Returns:
            合計報酬
        """
        state = env.reset()
        total_reward = 0
        steps = 0

        print("\nPlaying with learned policy:")

        while steps < 100:  # 無限ループ防止
            if render:
                env.render()

            action = self.get_action(state, training=False)
            next_state, reward, done = env.step(action)

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                if render:
                    env.render()
                print(f"✓ Goal reached in {steps} steps! Total reward: {total_reward:.2f}")
                break

        return total_reward

    def save(self, filepath: str):
        """Q-tableを保存."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'training_rewards': self.training_rewards,
                'epsilon': self.epsilon
            }, f)
        print(f"Q-table saved to {filepath}")

    def load(self, filepath: str):
        """Q-tableを読み込み."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data['q_table'])
        self.training_rewards = data['training_rewards']
        self.epsilon = data.get('epsilon', self.epsilon_min)

        print(f"Q-table loaded from {filepath}")

    def visualize_policy(self, env: GridWorld):
        """
        学習済みポリシーを可視化.

        各マスでの最適行動を矢印で表示.
        """
        action_symbols = ['↑', '→', '↓', '←']

        print("\nLearned Policy:")
        for i in range(env.size):
            row = []
            for j in range(env.size):
                state = (i, j)

                if state == env.goal:
                    row.append('G')
                elif state in env.obstacles:
                    row.append('X')
                elif state == env.start:
                    row.append('S')
                else:
                    best_action = np.argmax(self.q_table[state])
                    row.append(action_symbols[best_action])

            print(' '.join(row))
        print()

    def plot_training(self):
        """学習曲線をプロット."""
        plt.figure(figsize=(12, 5))

        # Raw rewards
        plt.subplot(1, 2, 1)
        plt.plot(self.training_rewards, alpha=0.3, label='Raw')
        # Moving average
        window = 100
        if len(self.training_rewards) >= window:
            moving_avg = np.convolve(
                self.training_rewards,
                np.ones(window)/window,
                mode='valid'
            )
            plt.plot(range(window-1, len(self.training_rewards)), moving_avg,
                    label=f'{window}-episode MA')

        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)

        # Q-value heatmap
        plt.subplot(1, 2, 2)
        q_values_grid = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                q_values_grid[i, j] = np.max(self.q_table[(i, j)])

        plt.imshow(q_values_grid, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Max Q-value')
        plt.title('Learned Q-values (Heatmap)')
        plt.xlabel('Column')
        plt.ylabel('Row')

        plt.tight_layout()
        plt.savefig('q_learning_results.png')
        print("Training plot saved to q_learning_results.png")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Q-Learning in GridWorld")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--play", action="store_true", help="Play with learned policy")
    parser.add_argument("--load", type=str, help="Load Q-table from file")
    parser.add_argument("--save", type=str, default="q_table.pkl", help="Save Q-table to file")
    parser.add_argument("--visualize", action="store_true", help="Visualize policy")

    args = parser.parse_args()

    # 環境作成
    env = GridWorld(size=5)

    # エージェント作成
    agent = QLearningAgent(
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=args.epsilon
    )

    # Q-table読み込み
    if args.load:
        agent.load(args.load)

    # 学習または実行
    if args.play:
        # 学習済みポリシーで実行
        agent.play(env, render=True)

    else:
        # 学習
        print("Starting Q-Learning training...")
        print(f"Episodes: {args.episodes}")
        print(f"Learning rate: {args.lr}")
        print(f"Discount factor: {args.gamma}")
        print()

        render_every = 100 if args.render else 0
        agent.train(env, n_episodes=args.episodes, render_every=render_every)

        # 保存
        agent.save(args.save)

        # 可視化
        if args.visualize:
            agent.visualize_policy(env)

        # プロット
        agent.plot_training()

        # 最終テスト
        print("\nFinal test:")
        agent.play(env, render=True)


if __name__ == "__main__":
    main()
