"""
SAC (Soft Actor-Critic)

SACは連続制御タスクに特化した最先端のアルゴリズム.
最大エントロピー強化学習フレームワークに基づく.

特徴:
    - 連続行動空間に特化
    - サンプル効率が非常に高い
    - 安定した学習
    - 自動温度調整（エントロピー係数の自動調整）

使い方:
    from sac import SACAgent
    import gymnasium as gym

    env = gym.make('Pendulum-v1')
    agent = SACAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        action_range=(env.action_space.low[0], env.action_space.high[0])
    )

    agent.train(env, n_episodes=500)
    agent.save('sac_pendulum.pth')

参考:
    - Paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (Haarnoja et al., 2018)
    - https://arxiv.org/abs/1801.01290
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, List
from collections import deque
import random


class ReplayBuffer:
    """経験再生バッファ."""

    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """
    ポリシーネットワーク（Actor）.

    連続行動のための確率的ポリシー.
    平均と標準偏差を出力し、正規分布からサンプリング.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] = [256, 256],
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super(Actor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # 共有層
        layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        self.shared = nn.Sequential(*layers)

        # 出力層
        self.mean = nn.Linear(prev_size, action_size)
        self.log_std = nn.Linear(prev_size, action_size)

    def forward(self, state):
        """順伝播."""
        x = self.shared(state)
        mean = self.mean(x)
        log_std = self.log_std(x)

        # log_stdをクリップ（安定性のため）
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state):
        """
        行動をサンプリング.

        Reparameterization Trickを使用:
        a = μ + σ * ε, ε ~ N(0, 1)

        Returns:
            action: サンプリングされた行動（tanh適用前）
            log_prob: 対数確率
            mean: 平均
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 正規分布からサンプリング
        normal = Normal(mean, std)
        z = normal.rsample()  # reparameterization trick

        # tanh変換（行動を[-1, 1]に制限）
        action = torch.tanh(z)

        # 対数確率（tanh変換の補正含む）
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    """
    Q関数ネットワーク（Critic）.

    Twin Q-networks（2つのQ関数）を使用して過大評価を防ぐ.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] = [256, 256]
    ):
        super(Critic, self).__init__()

        # Q1ネットワーク
        layers1 = []
        prev_size = state_size + action_size
        for hidden_size in hidden_sizes:
            layers1.append(nn.Linear(prev_size, hidden_size))
            layers1.append(nn.ReLU())
            prev_size = hidden_size
        layers1.append(nn.Linear(prev_size, 1))
        self.q1 = nn.Sequential(*layers1)

        # Q2ネットワーク
        layers2 = []
        prev_size = state_size + action_size
        for hidden_size in hidden_sizes:
            layers2.append(nn.Linear(prev_size, hidden_size))
            layers2.append(nn.ReLU())
            prev_size = hidden_size
        layers2.append(nn.Linear(prev_size, 1))
        self.q2 = nn.Sequential(*layers2)

    def forward(self, state, action):
        """Q値を計算."""
        x = torch.cat([state, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2


class SACAgent:
    """
    SAC (Soft Actor-Critic) エージェント.

    主要コンセプト:
        - Maximum Entropy RL: エントロピーを報酬に組み込む
        - Twin Q-networks: 過大評価を防ぐ
        - Automatic Temperature Tuning: エントロピー係数の自動調整
        - Off-policy学習: 高サンプル効率
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        action_range: Tuple[float, float] = (-1.0, 1.0),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_temperature: bool = True,
        buffer_size: int = 1000000,
        batch_size: int = 256,
        device: str = 'auto'
    ):
        """
        Args:
            state_size: 状態空間の次元
            action_size: 行動空間の次元
            action_range: 行動の範囲 (min, max)
            lr: 学習率
            gamma: 割引率
            tau: ターゲットネットワーク更新率
            alpha: 温度パラメータ（エントロピー係数）
            auto_temperature: 自動温度調整
            buffer_size: リプレイバッファサイズ
            batch_size: バッチサイズ
            device: デバイス
        """
        # デバイス
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.state_size = state_size
        self.action_size = action_size
        self.action_low, self.action_high = action_range
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Actor
        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Critic (Twin Q-networks)
        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # 温度パラメータ
        self.auto_temperature = auto_temperature
        if auto_temperature:
            # 目標エントロピー = -action_size
            self.target_entropy = -action_size
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

        # リプレイバッファ
        self.buffer = ReplayBuffer(buffer_size)

        # 統計
        self.episode_rewards = []
        self.critic_losses = []
        self.actor_losses = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        行動を選択.

        Args:
            state: 状態
            deterministic: 決定的行動（テスト用）

        Returns:
            action: 選択された行動
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(state_tensor)
            else:
                action, _, _ = self.actor.sample(state_tensor)

        action = action.cpu().numpy()[0]

        # 行動範囲にスケーリング
        action = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)

        return action

    def update(self):
        """ネットワークを更新."""
        if len(self.buffer) < self.batch_size:
            return

        # バッチサンプリング
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)

        # 行動を[-1, 1]に正規化
        actions_normalized = (actions - self.action_low) / (self.action_high - self.action_low) * 2.0 - 1.0

        # === Critic更新 ===
        with torch.no_grad():
            # 次状態の行動をサンプリング
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

            # ターゲットQ値（Twin Q-networksの最小値）
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)

            # ターゲット値（エントロピー項を含む）
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha.detach()
            else:
                alpha = self.alpha

            target_value = rewards + (1 - dones) * self.gamma * (q_target - alpha * next_log_probs)

        # 現在のQ値
        q1, q2 = self.critic(states, actions_normalized)

        # Critic損失
        critic_loss = nn.MSELoss()(q1, target_value) + nn.MSELoss()(q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # === Actor更新 ===
        sampled_actions, log_probs, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, sampled_actions)
        q_new = torch.min(q1_new, q2_new)

        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.detach()
        else:
            alpha = self.alpha

        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # === 温度パラメータ更新（自動調整）===
        if self.auto_temperature:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # === ターゲットネットワーク更新（ソフト更新）===
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 統計記録
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())

    def train(
        self,
        env,
        n_episodes: int = 500,
        max_steps: int = 1000,
        update_every: int = 1,
        verbose: bool = True
    ):
        """
        環境で学習.

        Args:
            env: Gym環境
            n_episodes: エピソード数
            max_steps: 最大ステップ数
            update_every: 更新間隔
            verbose: 詳細ログ
        """
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # 行動選択
                action = self.select_action(state)

                # 環境でステップ
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # バッファに追加
                self.buffer.add(state, action, reward, next_state, done)

                # 更新
                if step % update_every == 0:
                    self.update()

                episode_reward += reward
                state = next_state

                if done:
                    break

            self.episode_rewards.append(episode_reward)

            if verbose and (episode + 1) % 10 == 0:
                recent_rewards = self.episode_rewards[-100:]
                avg_reward = np.mean(recent_rewards)
                print(f"Episode {episode + 1}/{n_episodes} - "
                      f"Reward: {episode_reward:.2f} - "
                      f"Avg (100): {avg_reward:.2f}")

    def save(self, filepath: str):
        """モデルを保存."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards
        }, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str):
        """モデルを読み込み."""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.episode_rewards = checkpoint['episode_rewards']
        print(f"✓ Model loaded from {filepath}")


if __name__ == "__main__":
    # デモ: Pendulum-v1でSAC学習
    import gymnasium as gym

    print("SAC Demo - Pendulum-v1")
    print("=" * 60)

    # 環境作成
    env = gym.make('Pendulum-v1')

    # エージェント作成
    agent = SACAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        action_range=(env.action_space.low[0], env.action_space.high[0]),
        auto_temperature=True
    )

    # 学習
    print("\nTraining...")
    agent.train(env, n_episodes=100, verbose=True)

    # 保存
    agent.save('sac_pendulum.pth')

    # テスト
    print("\nTesting...")
    state, _ = env.reset()
    episode_reward = 0

    for _ in range(200):
        action = agent.select_action(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            break

    print(f"\nTest Episode Reward: {episode_reward:.2f}")
    print("\n✓ Demo completed!")
