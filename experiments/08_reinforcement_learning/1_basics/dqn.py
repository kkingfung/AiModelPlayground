"""
Deep Q-Network (DQN)

ニューラルネットワークでQ値を近似する強化学習アルゴリズム.
Q-Learningの限界（状態空間が大きい場合）を克服.

主要技術:
- Experience Replay: 過去の経験を再利用して学習
- Target Network: 学習を安定化
- ε-greedy探索: 探索と活用のバランス

使い方:
    # CartPole環境で学習
    python dqn.py --env CartPole-v1 --episodes 500 --render

    # Atariゲームで学習
    python dqn.py --env ALE/Breakout-v5 --episodes 1000 --use-cnn

    # 学習済みモデルで実行
    python dqn.py --env CartPole-v1 --play --model checkpoints/dqn_cartpole.pth
"""

import argparse
import random
from collections import deque, namedtuple
from pathlib import Path
from typing import Tuple, Optional, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 経験を保存するための名前付きタプル
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    Experience Replay Buffer.

    過去の経験（s, a, r, s', done）を保存し、ランダムサンプリングで学習.
    これにより:
    - データの相関を減らす
    - サンプル効率を向上
    - 学習の安定性を向上
    """

    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: バッファの最大サイズ
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """経験を追加."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """ランダムにバッチをサンプリング."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """
    Deep Q-Network.

    状態を入力として、各行動のQ値を出力するニューラルネットワーク.
    """

    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 128]):
        """
        Args:
            state_size: 状態の次元数
            action_size: 行動の数
            hidden_sizes: 隠れ層のサイズリスト
        """
        super(DQN, self).__init__()

        layers = []
        prev_size = state_size

        # 隠れ層を構築
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # 出力層
        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        順伝播.

        Args:
            x: 状態 [batch_size, state_size]

        Returns:
            Q値 [batch_size, action_size]
        """
        return self.network(x)


class CNNDQNAtari(nn.Module):
    """
    CNNベースのDQN（Atariゲーム用）.

    画像（84x84x4）を入力として、Q値を出力.
    DQN論文のアーキテクチャを使用.
    """

    def __init__(self, action_size: int):
        """
        Args:
            action_size: 行動の数
        """
        super(CNNDQNAtari, self).__init__()

        # 畳み込み層
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 84x84x4 → 20x20x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 20x20x32 → 9x9x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 9x9x64 → 7x7x64
            nn.ReLU()
        )

        # 全結合層
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        """
        順伝播.

        Args:
            x: 画像 [batch_size, 4, 84, 84]

        Returns:
            Q値 [batch_size, action_size]
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


class DQNAgent:
    """
    DQNエージェント.

    Experience ReplayとTarget Networkを使用したQ学習.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        use_cnn: bool = False,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: Optional[str] = None
    ):
        """
        Args:
            state_size: 状態の次元数
            action_size: 行動の数
            use_cnn: CNNを使用するか（Atari用）
            learning_rate: 学習率
            discount_factor: 割引率γ
            epsilon: 探索率（初期値）
            epsilon_decay: εの減衰率
            epsilon_min: εの最小値
            buffer_size: Replay Bufferのサイズ
            batch_size: ミニバッチサイズ
            target_update_freq: Target Networkの更新頻度（エピソード単位）
            device: 使用デバイス（cuda/cpu）
        """
        self.state_size = state_size
        self.action_size = action_size
        self.use_cnn = use_cnn
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # デバイス設定
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # ネットワーク作成
        if use_cnn:
            self.q_network = CNNDQNAtari(action_size).to(self.device)
            self.target_network = CNNDQNAtari(action_size).to(self.device)
        else:
            self.q_network = DQN(state_size, action_size).to(self.device)
            self.target_network = DQN(state_size, action_size).to(self.device)

        # Target Networkの重みをコピー
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 推論モード

        # オプティマイザ
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience Replay Buffer
        self.memory = ReplayBuffer(buffer_size)

        # 統計
        self.training_rewards = []
        self.training_losses = []
        self.episode_count = 0

    def get_action(self, state, training: bool = True) -> int:
        """
        行動選択（ε-greedy方策）.

        Args:
            state: 現在の状態
            training: 学習中かどうか

        Returns:
            選択された行動
        """
        if training and random.random() < self.epsilon:
            # 探索: ランダム行動
            return random.randrange(self.action_size)
        else:
            # 活用: Q値が最大の行動
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(1).item()

    def remember(self, state, action, reward, next_state, done):
        """経験をReplay Bufferに保存."""
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """
        Experience Replayで学習.

        Replay Bufferからランダムサンプリングして、Q-networkを更新.
        """
        if len(self.memory) < self.batch_size:
            return None

        # ミニバッチをサンプリング
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # バッチをテンソルに変換
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # 現在のQ値: Q(s, a)
        current_q_values = self.q_network(state_batch).gather(1, action_batch).squeeze(1)

        # 次の状態の最大Q値: max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            # 終端状態では次の価値は0
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # 損失計算（Huber Loss）
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

        # 最適化
        self.optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピング（学習の安定化）
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Target NetworkをQ-networkの重みで更新."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(
        self,
        env,
        n_episodes: int = 500,
        max_steps: int = 500,
        render: bool = False,
        verbose: bool = True
    ):
        """
        学習ループ.

        Args:
            env: Gym環境
            n_episodes: エピソード数
            max_steps: 1エピソードの最大ステップ数
            render: 描画するか
            verbose: 詳細表示
        """
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            total_loss = 0
            steps = 0

            for step in range(max_steps):
                if render:
                    env.render()

                # 行動選択
                action = self.get_action(state, training=True)

                # 環境ステップ
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # 経験を保存
                self.remember(state, action, reward, next_state, done)

                # Experience Replayで学習
                loss = self.replay()
                if loss is not None:
                    total_loss += loss

                total_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            # ε減衰
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Target Network更新
            self.episode_count += 1
            if self.episode_count % self.target_update_freq == 0:
                self.update_target_network()

            # 統計記録
            self.training_rewards.append(total_reward)
            if total_loss > 0:
                self.training_losses.append(total_loss / steps)

            # 進捗表示
            if verbose and episode % 10 == 0:
                avg_reward = np.mean(self.training_rewards[-10:])
                avg_loss = np.mean(self.training_losses[-10:]) if self.training_losses else 0
                print(f"Episode {episode}/{n_episodes}, "
                      f"Reward: {total_reward:.2f}, "
                      f"Avg Reward (last 10): {avg_reward:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, "
                      f"Epsilon: {self.epsilon:.3f}, "
                      f"Steps: {steps}, "
                      f"Buffer: {len(self.memory)}")

        print(f"\n✓ Training completed! Final avg reward: {np.mean(self.training_rewards[-100:]):.2f}")

    def play(
        self,
        env,
        n_episodes: int = 5,
        render: bool = True,
        max_steps: int = 500
    ) -> List[float]:
        """
        学習済みポリシーで実行.

        Args:
            env: Gym環境
            n_episodes: 実行エピソード数
            render: 描画するか
            max_steps: 1エピソードの最大ステップ数

        Returns:
            各エピソードの報酬リスト
        """
        episode_rewards = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0

            print(f"\nEpisode {episode + 1}/{n_episodes}")

            for step in range(max_steps):
                if render:
                    env.render()

                action = self.get_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                total_reward += reward
                state = next_state
                steps += 1

                if done:
                    break

            episode_rewards.append(total_reward)
            print(f"  Steps: {steps}, Total Reward: {total_reward:.2f}")

        avg_reward = np.mean(episode_rewards)
        print(f"\n✓ Average reward over {n_episodes} episodes: {avg_reward:.2f}")

        return episode_rewards

    def save(self, filepath: str):
        """モデルを保存."""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """モデルを読み込み."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.training_losses = checkpoint.get('training_losses', [])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.episode_count = checkpoint.get('episode_count', 0)

        print(f"Model loaded from {filepath}")

    def plot_training(self, save_path: str = "dqn_training.png"):
        """学習曲線をプロット."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Reward curve
        axes[0].plot(self.training_rewards, alpha=0.3, label='Raw')
        # Moving average
        window = 50
        if len(self.training_rewards) >= window:
            moving_avg = np.convolve(
                self.training_rewards,
                np.ones(window) / window,
                mode='valid'
            )
            axes[0].plot(range(window - 1, len(self.training_rewards)), moving_avg,
                        label=f'{window}-episode MA')

        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Training Rewards')
        axes[0].legend()
        axes[0].grid(True)

        # Loss curve
        if self.training_losses:
            axes[1].plot(self.training_losses, alpha=0.6)
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Training Loss')
            axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training plot saved to {save_path}")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Deep Q-Network (DQN)")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment name")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--target-update", type=int, default=10, help="Target network update frequency")
    parser.add_argument("--use-cnn", action="store_true", help="Use CNN (for Atari)")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--play", action="store_true", help="Play with learned policy")
    parser.add_argument("--model", type=str, help="Model path to load")
    parser.add_argument("--save", type=str, default="dqn_model.pth", help="Save model path")

    args = parser.parse_args()

    # 環境作成
    env = gym.make(args.env, render_mode="human" if args.render else None)

    # 状態・行動サイズ取得
    if args.use_cnn:
        state_size = None  # CNNでは不要
    else:
        state_size = env.observation_space.shape[0]

    action_size = env.action_space.n

    # エージェント作成
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        use_cnn=args.use_cnn,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon=args.epsilon,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update
    )

    # モデル読み込み
    if args.model:
        agent.load(args.model)

    # 学習または実行
    if args.play:
        # 学習済みポリシーで実行
        agent.play(env, n_episodes=5, render=True, max_steps=args.max_steps)
    else:
        # 学習
        print(f"Starting DQN training on {args.env}...")
        print(f"Episodes: {args.episodes}")
        print(f"Learning rate: {args.lr}")
        print(f"Discount factor: {args.gamma}")
        print(f"Buffer size: {args.buffer_size}")
        print(f"Batch size: {args.batch_size}\n")

        agent.train(
            env,
            n_episodes=args.episodes,
            max_steps=args.max_steps,
            render=args.render
        )

        # 保存
        Path("checkpoints").mkdir(exist_ok=True)
        save_path = f"checkpoints/{args.save}"
        agent.save(save_path)

        # プロット
        agent.plot_training(f"checkpoints/dqn_{args.env.replace('/', '_')}_training.png")

        # 最終テスト
        print("\nFinal test:")
        agent.play(env, n_episodes=3, render=True, max_steps=args.max_steps)

    env.close()


if __name__ == "__main__":
    main()
