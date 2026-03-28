"""
Proximal Policy Optimization (PPO)

PPOは現代の強化学習で最も人気のあるアルゴリズムの一つ.
OpenAI FiveやDota 2のボットなど、多くの実用例で使用されている.

特徴:
    - 安定した学習（Trust Region最適化）
    - 高いサンプル効率
    - 連続・離散行動空間の両方に対応
    - 実装が比較的シンプル

使い方:
    from ppo import PPOAgent
    import gymnasium as gym

    env = gym.make('CartPole-v1')
    agent = PPOAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        continuous=False
    )

    agent.train(env, n_iterations=500)
    agent.save('ppo_cartpole.pth')

参考:
    - Paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    - https://arxiv.org/abs/1707.06347
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import List, Tuple, Optional
from collections import deque
import time


class ActorCriticNetwork(nn.Module):
    """
    Actor-Criticネットワーク.

    Actor: ポリシー π(a|s) を出力
    Critic: 状態価値 V(s) を出力
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] = [64, 64],
        continuous: bool = False
    ):
        """
        Args:
            state_size: 状態空間の次元
            action_size: 行動空間の次元
            hidden_sizes: 隠れ層のサイズ
            continuous: 連続行動空間かどうか
        """
        super(ActorCriticNetwork, self).__init__()

        self.continuous = continuous

        # 共有層
        layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size

        self.shared_layers = nn.Sequential(*layers)

        # Actor head
        if continuous:
            # 連続行動: 平均と標準偏差を出力
            self.actor_mean = nn.Linear(prev_size, action_size)
            self.actor_log_std = nn.Linear(prev_size, action_size)
        else:
            # 離散行動: 行動確率を出力
            self.actor = nn.Linear(prev_size, action_size)

        # Critic head
        self.critic = nn.Linear(prev_size, 1)

    def forward(self, state):
        """順伝播."""
        x = self.shared_layers(state)

        if self.continuous:
            mean = self.actor_mean(x)
            log_std = self.actor_log_std(x)
            std = torch.exp(log_std)
            value = self.critic(x)
            return mean, std, value
        else:
            logits = self.actor(x)
            value = self.critic(x)
            return logits, value

    def get_action_and_value(self, state, action=None):
        """
        行動とその価値を取得.

        Args:
            state: 状態
            action: 行動（Noneの場合はサンプリング）

        Returns:
            action: 選択された行動
            log_prob: 対数確率
            entropy: エントロピー
            value: 状態価値
        """
        if self.continuous:
            mean, std, value = self.forward(state)
            dist = Normal(mean, std)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        else:
            logits, value = self.forward(state)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class RolloutBuffer:
    """
    ロールアウトバッファ（経験収集用）.

    PPOはオンポリシーアルゴリズムなので、
    現在のポリシーで収集したデータのみを使用.
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, state, action, reward, value, log_prob, done):
        """経験を追加."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self):
        """バッファ内容を取得."""
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards))
        values = torch.FloatTensor(np.array(self.values))
        log_probs = torch.FloatTensor(np.array(self.log_probs))
        dones = torch.FloatTensor(np.array(self.dones))

        return states, actions, rewards, values, log_probs, dones

    def clear(self):
        """バッファをクリア."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) エージェント.

    主要コンセプト:
        - Clipped Surrogate Objective: ポリシー更新を制限
        - Advantage推定: GAE (Generalized Advantage Estimation)
        - 複数エポックの更新: 同じデータで複数回学習
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        continuous: bool = False,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = 'auto'
    ):
        """
        Args:
            state_size: 状態空間の次元
            action_size: 行動空間の次元
            continuous: 連続行動空間
            lr: 学習率
            gamma: 割引率
            gae_lambda: GAEのλ
            clip_epsilon: クリッピング範囲
            value_coef: 価値関数損失の係数
            entropy_coef: エントロピーボーナスの係数
            max_grad_norm: 勾配クリッピング
            n_steps: ロールアウト長
            n_epochs: 更新エポック数
            batch_size: ミニバッチサイズ
            device: デバイス
        """
        # デバイス設定
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # ハイパーパラメータ
        self.state_size = state_size
        self.action_size = action_size
        self.continuous = continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # ネットワーク
        self.network = ActorCriticNetwork(
            state_size,
            action_size,
            continuous=continuous
        ).to(self.device)

        # オプティマイザ
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # バッファ
        self.buffer = RolloutBuffer()

        # 統計
        self.episode_rewards = []
        self.episode_lengths = []

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        行動を選択.

        Args:
            state: 現在の状態

        Returns:
            action: 選択された行動
            log_prob: 対数確率
            value: 状態価値
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(state_tensor)

        action = action.cpu().numpy()[0]
        log_prob = log_prob.cpu().item()
        value = value.cpu().item()

        return action, log_prob, value

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GAE (Generalized Advantage Estimation) を計算.

        GAE = Σ (γλ)^t * δ_t
        δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            rewards: 報酬列
            values: 状態価値列
            dones: 終了フラグ
            next_value: 次状態の価値

        Returns:
            advantages: アドバンテージ
            returns: リターン（価値のターゲット）
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        # 逆方向に計算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            # TD誤差
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]

            # GAE
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage

        # リターン = アドバンテージ + 価値
        returns = advantages + values

        return advantages, returns

    def update(self):
        """
        ポリシーとバリュー関数を更新.

        PPOの核心部分：
            - Clipped Surrogate Objective
            - 複数エポックの更新
            - ミニバッチ学習
        """
        # バッファからデータ取得
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get()

        # 最後の状態の価値を計算
        with torch.no_grad():
            last_state = states[-1].unsqueeze(0).to(self.device)
            _, _, _, last_value = self.network.get_action_and_value(last_state)
            last_value = last_value.cpu().item()

        # GAE計算
        advantages, returns = self.compute_gae(rewards, values, dones, last_value)

        # 正規化（安定化のため）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # デバイスに転送
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # 複数エポック更新
        for epoch in range(self.n_epochs):
            # ミニバッチ作成
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 現在のポリシーで評価
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch_states,
                    batch_actions if self.continuous else batch_actions.long()
                )

                # 重要度サンプリング比
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped Surrogate Objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 価値関数損失
                value_loss = nn.MSELoss()(new_values, batch_returns)

                # エントロピーボーナス
                entropy_loss = -entropy.mean()

                # 総損失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        # バッファクリア
        self.buffer.clear()

    def train(
        self,
        env,
        n_iterations: int = 1000,
        verbose: bool = True,
        save_interval: int = 100
    ):
        """
        環境で学習.

        Args:
            env: Gym環境
            n_iterations: イテレーション数
            verbose: 詳細ログ
            save_interval: 保存間隔
        """
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for iteration in range(n_iterations):
            # ロールアウト収集
            for step in range(self.n_steps):
                action, log_prob, value = self.select_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.buffer.add(state, action, reward, value, log_prob, done)

                episode_reward += reward
                episode_length += 1

                if done:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)

                    state, _ = env.reset()
                    episode_reward = 0
                    episode_length = 0
                else:
                    state = next_state

            # 更新
            self.update()

            # ログ
            if verbose and (iteration + 1) % 10 == 0:
                recent_rewards = self.episode_rewards[-100:]
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                print(f"Iteration {iteration + 1}/{n_iterations} - "
                      f"Avg Reward: {avg_reward:.2f} - "
                      f"Episodes: {len(self.episode_rewards)}")

            # 保存
            if (iteration + 1) % save_interval == 0:
                self.save(f'ppo_checkpoint_{iteration+1}.pth')

    def save(self, filepath: str):
        """モデルを保存."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str):
        """モデルを読み込み."""
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        print(f"✓ Model loaded from {filepath}")


if __name__ == "__main__":
    # デモ: CartPole-v1でPPO学習
    import gymnasium as gym

    print("PPO Demo - CartPole-v1")
    print("=" * 60)

    # 環境作成
    env = gym.make('CartPole-v1')

    # エージェント作成
    agent = PPOAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        continuous=False,
        n_steps=2048,
        n_epochs=10,
        batch_size=64
    )

    # 学習
    print("\nTraining...")
    agent.train(env, n_iterations=100, verbose=True)

    # 保存
    agent.save('ppo_cartpole.pth')

    # テスト
    print("\nTesting...")
    state, _ = env.reset()
    episode_reward = 0

    for _ in range(500):
        action, _, _ = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            break

    print(f"\nTest Episode Reward: {episode_reward}")
    print("\n✓ Demo completed!")
