"""
A3C (Asynchronous Advantage Actor-Critic)

A3Cは非同期に複数の環境で学習する並列アルゴリズム.
DeepMindがAtariゲームで成功を収めた手法.

特徴:
    - 並列学習（複数ワーカー）
    - Experience Replay不要
    - オンポリシー学習
    - 高速な学習

使い方:
    from a3c import A3CAgent
    import gymnasium as gym

    def make_env():
        return gym.make('CartPole-v1')

    agent = A3CAgent(
        state_size=4,
        action_size=2,
        n_workers=4
    )

    agent.train(make_env, n_iterations=1000)
    agent.save('a3c_cartpole.pth')

参考:
    - Paper: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
    - https://arxiv.org/abs/1602.01783
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp
from typing import List, Callable
import time
from collections import deque


class ActorCriticNetwork(nn.Module):
    """
    Actor-Criticネットワーク（A3C用）.

    PPOと似ているが、A3Cでは各ワーカーが独立した環境でロールアウトを収集.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] = [128, 128],
        continuous: bool = False
    ):
        super(ActorCriticNetwork, self).__init__()

        self.continuous = continuous

        # 共有層
        layers = []
        prev_size = state_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        self.shared_layers = nn.Sequential(*layers)

        # Actor
        if continuous:
            self.actor_mean = nn.Linear(prev_size, action_size)
            self.actor_log_std = nn.Parameter(torch.zeros(action_size))
        else:
            self.actor = nn.Linear(prev_size, action_size)

        # Critic
        self.critic = nn.Linear(prev_size, 1)

    def forward(self, state):
        x = self.shared_layers(state)

        if self.continuous:
            mean = self.actor_mean(x)
            std = torch.exp(self.actor_log_std)
            value = self.critic(x)
            return mean, std, value
        else:
            logits = self.actor(x)
            value = self.critic(x)
            return logits, value

    def get_action_and_value(self, state, action=None):
        """行動と価値を取得."""
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


def worker(
    worker_id: int,
    global_network: ActorCriticNetwork,
    optimizer: optim.Optimizer,
    make_env: Callable,
    n_steps: int,
    gamma: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    global_episode: mp.Value,
    global_reward: mp.Value,
    lock: mp.Lock,
    max_episodes: int
):
    """
    ワーカープロセス.

    各ワーカーは独立した環境でロールアウトを収集し、
    グローバルネットワークを更新する.

    Args:
        worker_id: ワーカーID
        global_network: 共有ネットワーク
        optimizer: 共有オプティマイザ
        make_env: 環境生成関数
        n_steps: ロールアウト長
        gamma: 割引率
        value_coef: 価値関数損失の係数
        entropy_coef: エントロピー係数
        max_grad_norm: 勾配クリッピング
        global_episode: グローバルエピソード数
        global_reward: グローバル報酬
        lock: ロック
        max_episodes: 最大エピソード数
    """
    # ローカル環境
    env = make_env()

    # ローカルネットワーク
    local_network = ActorCriticNetwork(
        state_size=global_network.shared_layers[0].in_features,
        action_size=global_network.actor.out_features if hasattr(global_network, 'actor') else global_network.actor_mean.out_features,
        continuous=global_network.continuous
    )

    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0

    while global_episode.value < max_episodes:
        # グローバルネットワークから同期
        local_network.load_state_dict(global_network.state_dict())

        # ロールアウト収集
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        for step in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = local_network.get_action_and_value(state_tensor)

            action_np = action.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(done)

            episode_reward += reward
            episode_length += 1

            if done:
                with lock:
                    global_episode.value += 1
                    global_reward.value = episode_reward

                if worker_id == 0 and global_episode.value % 10 == 0:
                    print(f"[Worker {worker_id}] Episode {global_episode.value} - "
                          f"Reward: {episode_reward:.2f} - "
                          f"Length: {episode_length}")

                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                break
            else:
                state = next_state

        # 最後の状態の価値
        if not dones[-1]:
            with torch.no_grad():
                last_state = torch.FloatTensor(next_state).unsqueeze(0)
                _, _, _, last_value = local_network.get_action_and_value(last_state)
                last_value = last_value.item()
        else:
            last_value = 0.0

        # リターン計算（TD(λ)ではなくN-step return）
        returns = []
        R = last_value
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns)
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        values = torch.FloatTensor(values)

        # 新しいポリシーで評価
        if local_network.continuous:
            actions_eval = actions
        else:
            actions_eval = actions.long()

        _, new_log_probs, entropy, new_values = local_network.get_action_and_value(states, actions_eval)

        # アドバンテージ
        advantages = returns - values

        # 損失計算
        policy_loss = -(new_log_probs * advantages.detach()).mean()
        value_loss = nn.MSELoss()(new_values, returns)
        entropy_loss = -entropy.mean()

        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

        # グローバルネットワーク更新
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(local_network.parameters(), max_grad_norm)

        # 勾配をグローバルネットワークにコピー
        for local_param, global_param in zip(local_network.parameters(), global_network.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone()
            else:
                global_param.grad += local_param.grad

        optimizer.step()


class A3CAgent:
    """
    A3C (Asynchronous Advantage Actor-Critic) エージェント.

    主要コンセプト:
        - 非同期並列学習
        - グローバルネットワークとローカルネットワーク
        - N-step returns
        - Experience Replay不要
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        continuous: bool = False,
        n_workers: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 40.0,
        n_steps: int = 20
    ):
        """
        Args:
            state_size: 状態空間の次元
            action_size: 行動空間の次元
            continuous: 連続行動空間
            n_workers: ワーカー数
            lr: 学習率
            gamma: 割引率
            value_coef: 価値関数損失の係数
            entropy_coef: エントロピー係数
            max_grad_norm: 勾配クリッピング
            n_steps: N-stepリターン
        """
        self.state_size = state_size
        self.action_size = action_size
        self.continuous = continuous
        self.n_workers = n_workers
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps

        # グローバルネットワーク
        self.global_network = ActorCriticNetwork(
            state_size,
            action_size,
            continuous=continuous
        )
        self.global_network.share_memory()  # プロセス間共有

        # オプティマイザ
        self.optimizer = optim.Adam(self.global_network.parameters(), lr=lr)

    def train(
        self,
        make_env: Callable,
        n_episodes: int = 1000,
        verbose: bool = True
    ):
        """
        並列学習.

        Args:
            make_env: 環境生成関数
            n_episodes: 総エピソード数
            verbose: 詳細ログ
        """
        # マルチプロセス共有変数
        global_episode = mp.Value('i', 0)
        global_reward = mp.Value('d', 0.0)
        lock = mp.Lock()

        # ワーカープロセス起動
        processes = []

        for worker_id in range(self.n_workers):
            p = mp.Process(
                target=worker,
                args=(
                    worker_id,
                    self.global_network,
                    self.optimizer,
                    make_env,
                    self.n_steps,
                    self.gamma,
                    self.value_coef,
                    self.entropy_coef,
                    self.max_grad_norm,
                    global_episode,
                    global_reward,
                    lock,
                    n_episodes
                )
            )
            p.start()
            processes.append(p)

        # すべてのワーカーが終了するまで待機
        for p in processes:
            p.join()

        if verbose:
            print("\n✓ Training completed!")

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        行動を選択（推論用）.

        Args:
            state: 状態

        Returns:
            action: 選択された行動
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = self.global_network.get_action_and_value(state_tensor)

        return action.cpu().numpy()[0]

    def save(self, filepath: str):
        """モデルを保存."""
        torch.save({
            'network': self.global_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str):
        """モデルを読み込み."""
        checkpoint = torch.load(filepath)
        self.global_network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"✓ Model loaded from {filepath}")


if __name__ == "__main__":
    # デモ: CartPole-v1でA3C学習
    import gymnasium as gym

    print("A3C Demo - CartPole-v1")
    print("=" * 60)

    # 環境生成関数
    def make_env():
        return gym.make('CartPole-v1')

    # テスト用に1つ環境を作成して次元を取得
    test_env = make_env()
    state_size = test_env.observation_space.shape[0]
    action_size = test_env.action_space.n
    test_env.close()

    # エージェント作成
    agent = A3CAgent(
        state_size=state_size,
        action_size=action_size,
        continuous=False,
        n_workers=4,
        n_steps=20
    )

    # 学習
    print("\nTraining with 4 workers...")
    print("(This may take a few minutes)")

    agent.train(make_env, n_episodes=500, verbose=True)

    # 保存
    agent.save('a3c_cartpole.pth')

    # テスト
    print("\nTesting...")
    env = make_env()
    state, _ = env.reset()
    episode_reward = 0

    for _ in range(500):
        action = agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            break

    env.close()

    print(f"\nTest Episode Reward: {episode_reward}")
    print("\n✓ Demo completed!")
