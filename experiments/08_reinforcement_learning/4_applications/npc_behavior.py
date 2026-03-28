"""
NPC Behavior Training with RL

このモジュールは、強化学習を使用してNPCの行動を訓練するためのアプリケーションです。
PPO、A3C、SACアルゴリズムを使用して、以下のようなNPC行動を学習できます：

使用例:
    - Combat AI（戦闘AI）: 敵と戦うNPC
    - Patrol AI（巡回AI）: 特定のエリアをパトロールするNPC
    - Companion AI（仲間AI）: プレイヤーをフォローするNPC
    - Trader AI（商人AI）: 価格交渉を行うNPC

使い方:
    from npc_behavior import NPCBehaviorTrainer, CombatBehavior
    import gymnasium as gym

    # カスタム環境を作成
    env = gym.make('YourGameEnv-v0')

    # トレーナー作成
    trainer = NPCBehaviorTrainer(
        behavior_type=CombatBehavior,
        algorithm='PPO',
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )

    # 訓練
    trainer.train(env, n_episodes=1000)

    # 保存
    trainer.save('combat_npc.pth')

    # Unity連携
    trainer.export_for_unity('combat_npc_weights.json')

参考:
    - PPO, A3C, SAC implementations in 3_algorithms/
    - Unity ML-Agents: https://github.com/Unity-Technologies/ml-agents
"""

import sys
import os
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent / '3_algorithms'))

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, List, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import json
import time
from collections import deque

# RL アルゴリズムをインポート
from ppo import PPOAgent
from sac import SACAgent


class BehaviorType(Enum):
    """NPCの行動タイプ."""
    COMBAT = "combat"           # 戦闘AI
    PATROL = "patrol"           # 巡回AI
    COMPANION = "companion"     # 仲間AI
    TRADER = "trader"           # 商人AI
    STEALTH = "stealth"         # ステルスAI
    BOSS = "boss"               # ボスAI


class NPCBehavior(ABC):
    """
    NPC行動の基底クラス.

    各NPCタイプはこのクラスを継承して、
    固有の報酬関数と状態処理を実装します.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 行動設定（難易度、攻撃性など）
        """
        self.config = config

    @abstractmethod
    def compute_reward(
        self,
        state: Dict[str, Any],
        action: Any,
        next_state: Dict[str, Any],
        done: bool
    ) -> float:
        """
        報酬を計算.

        Args:
            state: 現在の状態
            action: 実行した行動
            next_state: 次の状態
            done: エピソード終了フラグ

        Returns:
            reward: 報酬値
        """
        pass

    @abstractmethod
    def process_observation(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """
        生の観測を処理してニューラルネットワーク用の状態ベクトルに変換.

        Args:
            raw_obs: 環境からの生の観測

        Returns:
            state_vector: 状態ベクトル
        """
        pass

    @abstractmethod
    def action_to_game_input(self, action: Any) -> Dict[str, Any]:
        """
        RLエージェントの行動をゲーム入力に変換.

        Args:
            action: エージェントの行動

        Returns:
            game_input: ゲームエンジン用の入力
        """
        pass


class CombatBehavior(NPCBehavior):
    """
    戦闘AI行動.

    特徴:
        - 敵を追跡して攻撃
        - 体力が低いときは回避
        - 距離に応じた行動選択
        - ダメージを与えると報酬

    行動空間:
        0: 前進
        1: 後退
        2: 左移動
        3: 右移動
        4: 近接攻撃
        5: 遠距離攻撃
        6: 防御
        7: スキル使用
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'aggression': 0.7,          # 攻撃性（0-1）
            'survival_threshold': 0.3,   # この体力以下で防御的に
            'attack_range': 5.0,         # 攻撃範囲
            'chase_range': 15.0,         # 追跡範囲
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def compute_reward(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        done: bool
    ) -> float:
        """戦闘報酬を計算."""
        reward = 0.0

        # ダメージを与えた報酬
        if 'enemy_health' in state and 'enemy_health' in next_state:
            damage_dealt = state['enemy_health'] - next_state['enemy_health']
            reward += damage_dealt * 10.0

        # ダメージを受けたペナルティ
        if 'npc_health' in state and 'npc_health' in next_state:
            damage_taken = state['npc_health'] - next_state['npc_health']
            reward -= damage_taken * 5.0

        # 距離に応じた報酬
        if 'distance_to_enemy' in next_state:
            distance = next_state['distance_to_enemy']
            optimal_range = self.config['attack_range']

            # 最適距離にいると報酬
            if abs(distance - optimal_range) < 2.0:
                reward += 0.5

            # 遠すぎる場合はペナルティ
            if distance > self.config['chase_range']:
                reward -= 0.2

        # 敵を倒した報酬
        if next_state.get('enemy_defeated', False):
            reward += 100.0

        # 自分が倒されたペナルティ
        if done and next_state.get('npc_health', 1.0) <= 0:
            reward -= 50.0

        # 生存報酬（時間経過）
        reward += 0.01

        return reward

    def process_observation(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """観測を状態ベクトルに変換."""
        state_vector = []

        # NPC状態
        state_vector.append(raw_obs.get('npc_health', 1.0))
        state_vector.append(raw_obs.get('npc_stamina', 1.0))

        # NPC位置（正規化）
        npc_pos = raw_obs.get('npc_position', [0, 0, 0])
        state_vector.extend([p / 100.0 for p in npc_pos])

        # 敵への相対位置
        enemy_pos = raw_obs.get('enemy_position', [0, 0, 0])
        relative_pos = [e - n for e, n in zip(enemy_pos, npc_pos)]
        state_vector.extend([p / 100.0 for p in relative_pos])

        # 敵との距離
        distance = raw_obs.get('distance_to_enemy', 0.0)
        state_vector.append(distance / self.config['chase_range'])

        # 敵の状態
        state_vector.append(raw_obs.get('enemy_health', 1.0))

        # 攻撃可能フラグ
        state_vector.append(1.0 if raw_obs.get('can_attack', False) else 0.0)
        state_vector.append(1.0 if raw_obs.get('can_use_skill', False) else 0.0)

        return np.array(state_vector, dtype=np.float32)

    def action_to_game_input(self, action: int) -> Dict[str, Any]:
        """行動をゲーム入力に変換."""
        action_map = {
            0: {'move': 'forward'},
            1: {'move': 'backward'},
            2: {'move': 'left'},
            3: {'move': 'right'},
            4: {'attack': 'melee'},
            5: {'attack': 'ranged'},
            6: {'defend': True},
            7: {'skill': True},
        }
        return action_map.get(action, {'move': 'idle'})


class PatrolBehavior(NPCBehavior):
    """
    巡回AI行動.

    特徴:
        - 指定されたウェイポイントを巡回
        - 異常を検知すると調査
        - 規則的なパトロールパターン

    行動空間:
        0: 次のウェイポイントへ移動
        1: 周囲を調査
        2: 待機
        3: 前のウェイポイントへ戻る
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'waypoints': [[0, 0], [10, 0], [10, 10], [0, 10]],
            'patrol_speed': 1.0,
            'investigation_range': 8.0,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.current_waypoint_idx = 0

    def compute_reward(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        done: bool
    ) -> float:
        """巡回報酬を計算."""
        reward = 0.0

        # ウェイポイントに到達
        if next_state.get('reached_waypoint', False):
            reward += 10.0
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.config['waypoints'])

        # ウェイポイントに近づいている
        if 'distance_to_waypoint' in state and 'distance_to_waypoint' in next_state:
            distance_change = state['distance_to_waypoint'] - next_state['distance_to_waypoint']
            reward += distance_change * 0.5

        # 異常検知
        if next_state.get('detected_anomaly', False):
            reward += 5.0

        # 時間経過ペナルティ（効率的な巡回を促す）
        reward -= 0.01

        return reward

    def process_observation(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """観測を状態ベクトルに変換."""
        state_vector = []

        # NPC位置
        npc_pos = raw_obs.get('npc_position', [0, 0])
        state_vector.extend(npc_pos)

        # 現在のウェイポイント
        current_wp = self.config['waypoints'][self.current_waypoint_idx]
        state_vector.extend(current_wp)

        # ウェイポイントまでの距離
        distance = np.linalg.norm(np.array(npc_pos) - np.array(current_wp))
        state_vector.append(distance)

        # ウェイポイントインデックス（正規化）
        state_vector.append(self.current_waypoint_idx / len(self.config['waypoints']))

        # 異常検知フラグ
        state_vector.append(1.0 if raw_obs.get('anomaly_detected', False) else 0.0)

        return np.array(state_vector, dtype=np.float32)

    def action_to_game_input(self, action: int) -> Dict[str, Any]:
        """行動をゲーム入力に変換."""
        action_map = {
            0: {'move_to': self.config['waypoints'][self.current_waypoint_idx]},
            1: {'investigate': True},
            2: {'wait': True},
            3: {'move_to': self.config['waypoints'][self.current_waypoint_idx - 1]},
        }
        return action_map.get(action, {'wait': True})


class CompanionBehavior(NPCBehavior):
    """
    仲間AI行動.

    特徴:
        - プレイヤーをフォロー
        - プレイヤーをサポート（回復、バフなど）
        - プレイヤーが攻撃されたら反撃

    行動空間:
        0: プレイヤーに近づく
        1: プレイヤーから適度な距離を保つ
        2: プレイヤーをサポート
        3: 敵を攻撃
        4: アイテムを拾う
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            'follow_distance': 3.0,
            'support_cooldown': 5.0,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def compute_reward(
        self,
        state: Dict[str, Any],
        action: int,
        next_state: Dict[str, Any],
        done: bool
    ) -> float:
        """仲間報酬を計算."""
        reward = 0.0

        # プレイヤーとの距離を適切に保つ
        distance = next_state.get('distance_to_player', 0.0)
        optimal_distance = self.config['follow_distance']
        distance_penalty = abs(distance - optimal_distance) * -0.1
        reward += distance_penalty

        # サポート成功
        if next_state.get('support_successful', False):
            reward += 15.0

        # プレイヤーの体力が回復した
        if 'player_health' in state and 'player_health' in next_state:
            health_increase = next_state['player_health'] - state['player_health']
            if health_increase > 0:
                reward += health_increase * 20.0

        # 敵にダメージを与えた
        if 'enemy_health' in state and 'enemy_health' in next_state:
            damage_dealt = state['enemy_health'] - next_state['enemy_health']
            reward += damage_dealt * 5.0

        # プレイヤーが倒されたペナルティ
        if next_state.get('player_defeated', False):
            reward -= 100.0

        return reward

    def process_observation(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """観測を状態ベクトルに変換."""
        state_vector = []

        # NPC状態
        state_vector.append(raw_obs.get('npc_health', 1.0))

        # プレイヤーへの相対位置
        npc_pos = raw_obs.get('npc_position', [0, 0, 0])
        player_pos = raw_obs.get('player_position', [0, 0, 0])
        relative_pos = [p - n for p, n in zip(player_pos, npc_pos)]
        state_vector.extend([p / 50.0 for p in relative_pos])

        # プレイヤーとの距離
        distance = raw_obs.get('distance_to_player', 0.0)
        state_vector.append(distance / 20.0)

        # プレイヤー状態
        state_vector.append(raw_obs.get('player_health', 1.0))

        # サポート可能フラグ
        state_vector.append(1.0 if raw_obs.get('can_support', False) else 0.0)

        # 敵が近くにいるか
        state_vector.append(1.0 if raw_obs.get('enemy_nearby', False) else 0.0)

        return np.array(state_vector, dtype=np.float32)

    def action_to_game_input(self, action: int) -> Dict[str, Any]:
        """行動をゲーム入力に変換."""
        action_map = {
            0: {'follow': 'close'},
            1: {'follow': 'optimal'},
            2: {'support': True},
            3: {'attack': 'enemy'},
            4: {'pickup': 'item'},
        }
        return action_map.get(action, {'follow': 'optimal'})


class NPCBehaviorTrainer:
    """
    NPC行動トレーナー.

    強化学習アルゴリズム（PPO, A3C, SAC）を使用して
    NPCの行動を訓練します.
    """

    def __init__(
        self,
        behavior_type: type,
        algorithm: str = 'PPO',
        state_size: Optional[int] = None,
        action_size: Optional[int] = None,
        continuous: bool = False,
        behavior_config: Optional[Dict[str, Any]] = None,
        **algorithm_kwargs
    ):
        """
        Args:
            behavior_type: NPCBehavior継承クラス
            algorithm: 使用するアルゴリズム（'PPO', 'A3C', 'SAC'）
            state_size: 状態空間の次元（Noneの場合は自動検出）
            action_size: 行動空間の次元（Noneの場合は自動検出）
            continuous: 連続行動空間
            behavior_config: 行動設定
            **algorithm_kwargs: アルゴリズム固有のパラメータ
        """
        self.behavior_type = behavior_type
        self.behavior = behavior_type(behavior_config or {})
        self.algorithm_name = algorithm
        self.continuous = continuous

        # 状態・行動サイズを自動検出（環境から取得する必要がある）
        self.state_size = state_size
        self.action_size = action_size

        # アルゴリズムを作成
        self.agent = None
        self.algorithm_kwargs = algorithm_kwargs

        # 統計
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
        }

    def _create_agent(self):
        """RLエージェントを作成."""
        if self.state_size is None or self.action_size is None:
            raise ValueError("state_size and action_size must be set before creating agent")

        if self.algorithm_name == 'PPO':
            self.agent = PPOAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                continuous=self.continuous,
                **self.algorithm_kwargs
            )
        elif self.algorithm_name == 'SAC':
            if not self.continuous:
                raise ValueError("SAC requires continuous action space")
            self.agent = SACAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                **self.algorithm_kwargs
            )
        elif self.algorithm_name == 'A3C':
            # A3Cは特別な訓練方法が必要
            raise NotImplementedError("A3C integration coming soon")
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")

    def train(
        self,
        env,
        n_episodes: int = 1000,
        max_steps: int = 1000,
        verbose: bool = True,
        save_interval: int = 100,
        eval_interval: int = 50
    ):
        """
        環境でNPCを訓練.

        Args:
            env: Gym互換環境
            n_episodes: エピソード数
            max_steps: 最大ステップ数
            verbose: 詳細ログ
            save_interval: モデル保存間隔
            eval_interval: 評価間隔
        """
        # 環境から状態・行動サイズを取得
        if self.state_size is None:
            if hasattr(env.observation_space, 'shape'):
                self.state_size = env.observation_space.shape[0]
            else:
                raise ValueError("Cannot infer state_size from environment")

        if self.action_size is None:
            if hasattr(env.action_space, 'n'):
                self.action_size = env.action_space.n
            elif hasattr(env.action_space, 'shape'):
                self.action_size = env.action_space.shape[0]
            else:
                raise ValueError("Cannot infer action_size from environment")

        # エージェント作成
        if self.agent is None:
            self._create_agent()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {self.behavior_type.__name__} with {self.algorithm_name}")
            print(f"State size: {self.state_size}, Action size: {self.action_size}")
            print(f"{'='*60}\n")

        # 訓練ループ
        if self.algorithm_name == 'PPO':
            # PPOの場合は専用の訓練メソッドを使用
            self._train_ppo(env, n_episodes, max_steps, verbose, save_interval, eval_interval)
        elif self.algorithm_name == 'SAC':
            # SACの場合
            self._train_sac(env, n_episodes, max_steps, verbose, save_interval, eval_interval)

    def _train_ppo(
        self,
        env,
        n_episodes: int,
        max_steps: int,
        verbose: bool,
        save_interval: int,
        eval_interval: int
    ):
        """PPOで訓練."""
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_count = 0

        for step in range(n_episodes * max_steps):
            # 行動選択
            if self.continuous:
                action, log_prob, value = self.agent.select_action(state)
            else:
                action, log_prob, value = self.agent.select_action(state)

            # 環境でステップ
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 行動固有の報酬を計算（オプション）
            # reward = self.behavior.compute_reward(state, action, next_state, done)

            # バッファに追加
            self.agent.buffer.add(state, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_length += 1

            # エピソード終了
            if done:
                self.training_stats['episode_rewards'].append(episode_reward)
                self.training_stats['episode_lengths'].append(episode_length)
                episode_count += 1

                if verbose and episode_count % 10 == 0:
                    recent = self.training_stats['episode_rewards'][-100:]
                    avg_reward = np.mean(recent)
                    self.training_stats['avg_rewards'].append(avg_reward)
                    print(f"Episode {episode_count}/{n_episodes} - "
                          f"Reward: {episode_reward:.2f} - "
                          f"Avg (100): {avg_reward:.2f} - "
                          f"Length: {episode_length}")

                # 保存
                if episode_count % save_interval == 0:
                    self.save(f'npc_{self.behavior_type.__name__.lower()}_{episode_count}.pth')

                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0

                if episode_count >= n_episodes:
                    break
            else:
                state = next_state

            # 更新
            if len(self.agent.buffer) >= self.agent.n_steps:
                self.agent.update()

    def _train_sac(
        self,
        env,
        n_episodes: int,
        max_steps: int,
        verbose: bool,
        save_interval: int,
        eval_interval: int
    ):
        """SACで訓練."""
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # 行動選択
                action = self.agent.select_action(state)

                # 環境でステップ
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # バッファに追加
                self.agent.buffer.add(state, action, reward, next_state, done)

                # 更新
                self.agent.update()

                episode_reward += reward
                state = next_state

                if done:
                    break

            self.training_stats['episode_rewards'].append(episode_reward)

            if verbose and (episode + 1) % 10 == 0:
                recent = self.training_stats['episode_rewards'][-100:]
                avg_reward = np.mean(recent)
                print(f"Episode {episode + 1}/{n_episodes} - "
                      f"Reward: {episode_reward:.2f} - "
                      f"Avg (100): {avg_reward:.2f}")

            # 保存
            if (episode + 1) % save_interval == 0:
                self.save(f'npc_{self.behavior_type.__name__.lower()}_{episode+1}.pth')

    def evaluate(self, env, n_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """
        訓練済みモデルを評価.

        Args:
            env: 評価環境
            n_episodes: 評価エピソード数
            render: 描画するか

        Returns:
            stats: 評価統計
        """
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                # 決定的行動
                if self.algorithm_name == 'PPO':
                    action, _, _ = self.agent.select_action(state)
                elif self.algorithm_name == 'SAC':
                    action = self.agent.select_action(state, deterministic=True)

                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1

                if render:
                    env.render()

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
        }

        return stats

    def save(self, filepath: str):
        """モデルを保存."""
        save_data = {
            'algorithm': self.algorithm_name,
            'behavior_type': self.behavior_type.__name__,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'continuous': self.continuous,
            'behavior_config': self.behavior.config,
            'training_stats': self.training_stats,
        }

        # エージェントの状態を保存
        self.agent.save(filepath)

        # メタデータを別ファイルに保存
        meta_filepath = filepath.replace('.pth', '_meta.json')
        with open(meta_filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"✓ NPC behavior model saved to {filepath}")
        print(f"✓ Metadata saved to {meta_filepath}")

    def load(self, filepath: str):
        """モデルを読み込み."""
        # メタデータを読み込み
        meta_filepath = filepath.replace('.pth', '_meta.json')
        with open(meta_filepath, 'r') as f:
            save_data = json.load(f)

        self.algorithm_name = save_data['algorithm']
        self.state_size = save_data['state_size']
        self.action_size = save_data['action_size']
        self.continuous = save_data['continuous']
        self.training_stats = save_data['training_stats']

        # エージェント作成
        self._create_agent()

        # エージェントの状態を読み込み
        self.agent.load(filepath)

        print(f"✓ NPC behavior model loaded from {filepath}")

    def export_for_unity(self, filepath: str):
        """
        Unityで使用できる形式でエクスポート.

        ニューラルネットワークの重みをJSON形式で保存します.
        Unity側でMLAgentsまたはカスタム推論エンジンを使用して読み込みます.

        Args:
            filepath: 出力ファイルパス（.json）
        """
        if self.agent is None:
            raise ValueError("No trained agent to export")

        # ネットワーク構造と重みを抽出
        export_data = {
            'behavior_type': self.behavior_type.__name__,
            'algorithm': self.algorithm_name,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'continuous': self.continuous,
            'behavior_config': self.behavior.config,
            'network': {}
        }

        # PPOの場合
        if self.algorithm_name == 'PPO':
            network = self.agent.network
            export_data['network'] = self._extract_network_weights(network)

        # SACの場合
        elif self.algorithm_name == 'SAC':
            export_data['network']['actor'] = self._extract_network_weights(self.agent.actor)
            export_data['network']['critic'] = self._extract_network_weights(self.agent.critic)

        # JSONに保存
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✓ Unity export saved to {filepath}")
        print(f"  Use this file with Unity ML-Agents or custom inference engine")

    def _extract_network_weights(self, network: nn.Module) -> Dict:
        """ネットワークの重みを抽出."""
        weights = {}

        for name, param in network.named_parameters():
            weights[name] = param.detach().cpu().numpy().tolist()

        return weights


# ============================================================================
# デモ: 戦闘NPCの訓練
# ============================================================================

if __name__ == "__main__":
    import gymnasium as gym

    print("NPC Behavior Training Demo")
    print("=" * 60)

    # カスタム環境の代わりにCartPoleを使用（デモ用）
    env = gym.make('CartPole-v1')

    # 戦闘NPCトレーナー作成
    trainer = NPCBehaviorTrainer(
        behavior_type=CombatBehavior,
        algorithm='PPO',
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        continuous=False,
        behavior_config={
            'aggression': 0.8,
            'survival_threshold': 0.2,
        },
        # PPO parameters
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
    )

    print("\nTraining Combat NPC...")
    trainer.train(env, n_episodes=100, verbose=True, save_interval=50)

    # 保存
    trainer.save('combat_npc_demo.pth')

    # Unity用エクスポート
    trainer.export_for_unity('combat_npc_unity.json')

    # 評価
    print("\n" + "=" * 60)
    print("Evaluating trained NPC...")
    stats = trainer.evaluate(env, n_episodes=10)

    print(f"\nEvaluation Results:")
    print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  Mean Length: {stats['mean_length']:.1f}")
    print(f"  Min/Max Reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")

    env.close()

    print("\n✓ Demo completed!")
    print("\nNext steps:")
    print("  1. Create custom game environment with NPCBehavior interface")
    print("  2. Train NPCs with different behaviors (Combat, Patrol, Companion)")
    print("  3. Export to Unity and integrate with game")
