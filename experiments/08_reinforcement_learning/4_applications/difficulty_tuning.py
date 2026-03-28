"""
Dynamic Difficulty Adjustment (DDA) with RL

このモジュールは、強化学習を使用してゲームの難易度を動的に調整します。
プレイヤーのパフォーマンスをリアルタイムで分析し、
最適なチャレンジレベルを維持するように難易度を自動調整します。

主要概念:
    - Flow理論: プレイヤーをフロー状態に保つ
    - プレイヤープロファイリング: スキルレベルの推定
    - リアルタイム調整: ゲームプレイ中に難易度を変更
    - パーソナライズ: 各プレイヤーに最適な難易度

使い方:
    from difficulty_tuning import DifficultyTuner, PlayerPerformance

    # チューナー作成
    tuner = DifficultyTuner(
        difficulty_levels=5,
        target_win_rate=0.6,
        algorithm='PPO'
    )

    # ゲームループで使用
    while game_running:
        # プレイヤーパフォーマンスを記録
        performance = PlayerPerformance(
            success=player_won,
            completion_time=time_taken,
            deaths=death_count,
            score=final_score
        )

        # 難易度を調整
        new_difficulty = tuner.adjust_difficulty(performance)
        game.set_difficulty(new_difficulty)

    # モデルを訓練（オフライン）
    tuner.train_from_data(player_data, n_epochs=100)

参考:
    - Flow Theory: https://en.wikipedia.org/wiki/Flow_(psychology)
    - Dynamic Difficulty Adjustment: https://ieeexplore.ieee.org/document/5593336
    - Player Modeling: https://dl.acm.org/doi/10.1145/1378773.1378781
"""

import sys
import os
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent / '3_algorithms'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
import time

from ppo import PPOAgent


@dataclass
class PlayerPerformance:
    """
    プレイヤーのパフォーマンス記録.

    このデータを使用して難易度調整を行います.
    """
    success: bool                          # レベル/ミッションをクリアしたか
    completion_time: float                 # 完了時間（秒）
    deaths: int = 0                        # 死亡回数
    score: float = 0.0                     # スコア
    damage_taken: float = 0.0              # 受けたダメージ
    damage_dealt: float = 0.0              # 与えたダメージ
    resources_used: int = 0                # 使用したリソース（ポーション等）
    mistakes: int = 0                      # ミスの回数
    perfect_actions: int = 0               # 完璧な行動の回数
    retry_count: int = 0                   # リトライ回数

    # メタ情報
    timestamp: float = field(default_factory=time.time)
    level_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換."""
        return {
            'success': self.success,
            'completion_time': self.completion_time,
            'deaths': self.deaths,
            'score': self.score,
            'damage_taken': self.damage_taken,
            'damage_dealt': self.damage_dealt,
            'resources_used': self.resources_used,
            'mistakes': self.mistakes,
            'perfect_actions': self.perfect_actions,
            'retry_count': self.retry_count,
            'timestamp': self.timestamp,
            'level_id': self.level_id,
        }


@dataclass
class PlayerProfile:
    """
    プレイヤープロファイル.

    プレイヤーのスキルレベルと特性を追跡します.
    """
    skill_level: float = 0.5               # 推定スキルレベル（0-1）
    total_playtime: float = 0.0            # 総プレイ時間
    total_attempts: int = 0                # 総試行回数
    total_successes: int = 0               # 総成功回数
    avg_completion_time: float = 0.0       # 平均完了時間
    avg_deaths: float = 0.0                # 平均死亡回数

    # 最近のパフォーマンス（移動平均）
    recent_win_rate: float = 0.5           # 最近の勝率
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=10))
    recent_completion_times: deque = field(default_factory=lambda: deque(maxlen=10))

    # プレイスタイル
    playstyle: str = "balanced"            # aggressive, defensive, balanced
    prefers_challenge: bool = False        # チャレンジを好むか
    frustration_tolerance: float = 0.5     # フラストレーション耐性（0-1）

    def update(self, performance: PlayerPerformance):
        """パフォーマンスでプロファイルを更新."""
        self.total_attempts += 1
        if performance.success:
            self.total_successes += 1

        # 勝率更新
        win_rate = self.total_successes / self.total_attempts
        self.recent_win_rate = win_rate

        # 最近のスコア・時間を追加
        self.recent_scores.append(performance.score)
        self.recent_completion_times.append(performance.completion_time)

        # 平均値更新
        self.avg_completion_time = np.mean(list(self.recent_completion_times)) if self.recent_completion_times else 0.0
        self.avg_deaths = (self.avg_deaths * (self.total_attempts - 1) + performance.deaths) / self.total_attempts

        # スキルレベル推定（簡易版）
        # 勝率、死亡回数、完了時間から推定
        skill_estimate = 0.0
        skill_estimate += self.recent_win_rate * 0.4
        skill_estimate += max(0, 1.0 - self.avg_deaths / 10.0) * 0.3
        if self.avg_completion_time > 0:
            time_factor = max(0, 1.0 - self.avg_completion_time / 300.0)  # 5分を基準
            skill_estimate += time_factor * 0.3

        # EMAでスキルレベルを更新
        alpha = 0.1
        self.skill_level = alpha * skill_estimate + (1 - alpha) * self.skill_level
        self.skill_level = np.clip(self.skill_level, 0.0, 1.0)


class DifficultyState:
    """
    難易度調整のための状態表現.

    プレイヤーのパフォーマンスとプロファイルを組み合わせて、
    RLエージェントが使用する状態ベクトルを作成します.
    """

    @staticmethod
    def from_performance_and_profile(
        performance: PlayerPerformance,
        profile: PlayerProfile,
        current_difficulty: int,
        max_difficulty: int
    ) -> np.ndarray:
        """
        パフォーマンスとプロファイルから状態ベクトルを作成.

        Args:
            performance: 最新のパフォーマンス
            profile: プレイヤープロファイル
            current_difficulty: 現在の難易度
            max_difficulty: 最大難易度

        Returns:
            state: 状態ベクトル
        """
        state = []

        # プレイヤープロファイル
        state.append(profile.skill_level)
        state.append(profile.recent_win_rate)
        state.append(profile.avg_deaths / 10.0)  # 正規化
        state.append(min(profile.avg_completion_time / 300.0, 1.0))  # 正規化（5分基準）

        # 最新パフォーマンス
        state.append(1.0 if performance.success else 0.0)
        state.append(performance.deaths / 10.0)
        state.append(min(performance.completion_time / 300.0, 1.0))
        state.append(performance.score / 1000.0)  # スコアは適宜正規化

        # 現在の難易度（正規化）
        state.append(current_difficulty / max_difficulty)

        # 最近のトレンド
        if len(profile.recent_scores) >= 2:
            score_trend = (profile.recent_scores[-1] - profile.recent_scores[-2]) / 1000.0
            state.append(np.clip(score_trend, -1.0, 1.0))
        else:
            state.append(0.0)

        return np.array(state, dtype=np.float32)


class DifficultyReward:
    """
    難易度調整の報酬関数.

    目標:
        - プレイヤーをフロー状態に保つ
        - 勝率を目標値付近に保つ
        - 過度な変化を避ける
        - エンゲージメントを最大化
    """

    @staticmethod
    def compute(
        performance: PlayerPerformance,
        profile: PlayerProfile,
        old_difficulty: int,
        new_difficulty: int,
        target_win_rate: float = 0.6,
        target_completion_time: float = 180.0
    ) -> float:
        """
        報酬を計算.

        Args:
            performance: パフォーマンス
            profile: プロファイル
            old_difficulty: 変更前の難易度
            new_difficulty: 変更後の難易度
            target_win_rate: 目標勝率
            target_completion_time: 目標完了時間

        Returns:
            reward: 報酬値
        """
        reward = 0.0

        # 1. 勝率を目標値に近づける
        win_rate_error = abs(profile.recent_win_rate - target_win_rate)
        win_rate_reward = -win_rate_error * 10.0
        reward += win_rate_reward

        # 2. 完了時間を目標値に近づける
        if performance.completion_time > 0:
            time_error = abs(performance.completion_time - target_completion_time) / target_completion_time
            time_reward = -time_error * 2.0
            reward += time_reward

        # 3. フロー状態を促進
        # スキルと難易度のバランスが取れていると報酬
        difficulty_normalized = new_difficulty / 10.0  # 最大難易度を10と仮定
        skill_difficulty_match = 1.0 - abs(profile.skill_level - difficulty_normalized)
        flow_reward = skill_difficulty_match * 5.0
        reward += flow_reward

        # 4. 過度な難易度変化にペナルティ
        difficulty_change = abs(new_difficulty - old_difficulty)
        if difficulty_change > 2:
            reward -= (difficulty_change - 2) * 2.0

        # 5. プレイヤーのフラストレーションを考慮
        if performance.retry_count > 3:
            # リトライが多い場合、難易度を下げると報酬
            if new_difficulty < old_difficulty:
                reward += 3.0

        # 6. 連続失敗/成功のハンドリング
        if not performance.success and performance.retry_count > 2:
            # 連続失敗→難易度を下げるべき
            if new_difficulty < old_difficulty:
                reward += 2.0
        elif performance.success and performance.deaths == 0:
            # 完璧にクリア→難易度を上げるべき
            if new_difficulty > old_difficulty:
                reward += 2.0

        # 7. エンゲージメント報酬
        # スコアが高い、完璧な行動が多い場合
        engagement_score = performance.score / 1000.0 + performance.perfect_actions / 10.0
        reward += engagement_score * 0.5

        return reward


class DifficultyTuner:
    """
    動的難易度調整システム.

    強化学習を使用してプレイヤーに最適な難易度を自動調整します.
    """

    def __init__(
        self,
        difficulty_levels: int = 5,
        target_win_rate: float = 0.6,
        target_completion_time: float = 180.0,
        algorithm: str = 'PPO',
        use_rl: bool = True,
        **rl_kwargs
    ):
        """
        Args:
            difficulty_levels: 難易度レベル数（例: 1=Easy, 5=Hard）
            target_win_rate: 目標勝率（0-1）
            target_completion_time: 目標完了時間（秒）
            algorithm: 使用するRLアルゴリズム
            use_rl: RLを使用するか（Falseの場合はヒューリスティック）
            **rl_kwargs: RLアルゴリズムのパラメータ
        """
        self.difficulty_levels = difficulty_levels
        self.target_win_rate = target_win_rate
        self.target_completion_time = target_completion_time
        self.algorithm = algorithm
        self.use_rl = use_rl

        # 状態・行動サイズ
        # 状態: [skill_level, win_rate, avg_deaths, avg_time, success, deaths, time, score, current_diff, score_trend] = 10次元
        self.state_size = 10
        # 行動: [難易度を下げる, 維持, 上げる] = 3
        self.action_size = 3

        # プレイヤープロファイル
        self.profiles: Dict[str, PlayerProfile] = {}

        # 現在の難易度（プレイヤーごと）
        self.current_difficulties: Dict[str, int] = {}

        # パフォーマンス履歴
        self.performance_history: List[Tuple[np.ndarray, int, float]] = []  # (state, action, reward)

        # RLエージェント
        self.agent: Optional[PPOAgent] = None
        if use_rl:
            self.agent = PPOAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                continuous=False,
                **rl_kwargs
            )

        # 統計
        self.adjustment_history: List[Dict[str, Any]] = []

    def get_or_create_profile(self, player_id: str) -> PlayerProfile:
        """プレイヤープロファイルを取得または作成."""
        if player_id not in self.profiles:
            self.profiles[player_id] = PlayerProfile()
            # 初期難易度は中程度
            self.current_difficulties[player_id] = self.difficulty_levels // 2
        return self.profiles[player_id]

    def adjust_difficulty(
        self,
        performance: PlayerPerformance,
        player_id: str = "default"
    ) -> int:
        """
        パフォーマンスに基づいて難易度を調整.

        Args:
            performance: プレイヤーのパフォーマンス
            player_id: プレイヤーID

        Returns:
            new_difficulty: 新しい難易度レベル
        """
        # プロファイル取得
        profile = self.get_or_create_profile(player_id)

        # プロファイル更新
        profile.update(performance)

        # 現在の難易度
        current_difficulty = self.current_difficulties[player_id]

        # 状態を作成
        state = DifficultyState.from_performance_and_profile(
            performance,
            profile,
            current_difficulty,
            self.difficulty_levels
        )

        # 難易度調整
        if self.use_rl and self.agent is not None:
            # RLエージェントで決定
            action = self._select_action_rl(state)
        else:
            # ヒューリスティックで決定
            action = self._select_action_heuristic(performance, profile, current_difficulty)

        # 行動を難易度変化に変換
        # action: 0=下げる, 1=維持, 2=上げる
        difficulty_change = action - 1
        new_difficulty = current_difficulty + difficulty_change
        new_difficulty = np.clip(new_difficulty, 1, self.difficulty_levels)

        # 報酬を計算（訓練用）
        reward = DifficultyReward.compute(
            performance,
            profile,
            current_difficulty,
            new_difficulty,
            self.target_win_rate,
            self.target_completion_time
        )

        # 履歴に記録
        self.performance_history.append((state, action, reward))

        # RLエージェントのバッファに追加
        if self.use_rl and self.agent is not None:
            # TODO: オンライン学習の実装
            pass

        # 調整履歴に記録
        self.adjustment_history.append({
            'player_id': player_id,
            'timestamp': time.time(),
            'old_difficulty': current_difficulty,
            'new_difficulty': new_difficulty,
            'action': action,
            'reward': reward,
            'performance': performance.to_dict(),
            'skill_level': profile.skill_level,
            'win_rate': profile.recent_win_rate,
        })

        # 難易度を更新
        self.current_difficulties[player_id] = new_difficulty

        return new_difficulty

    def _select_action_rl(self, state: np.ndarray) -> int:
        """RLエージェントで行動を選択."""
        if self.agent is None:
            raise ValueError("RL agent not initialized")

        action, _, _ = self.agent.select_action(state)
        return int(action)

    def _select_action_heuristic(
        self,
        performance: PlayerPerformance,
        profile: PlayerProfile,
        current_difficulty: int
    ) -> int:
        """
        ヒューリスティックで行動を選択.

        ルールベースの難易度調整:
        - 勝率が高すぎる → 難易度を上げる
        - 勝率が低すぎる → 難易度を下げる
        - リトライが多い → 難易度を下げる
        """
        action = 1  # デフォルトは維持

        # 勝率ベース
        if profile.recent_win_rate > self.target_win_rate + 0.15:
            # 勝率が高すぎる → 難易度を上げる
            if current_difficulty < self.difficulty_levels:
                action = 2
        elif profile.recent_win_rate < self.target_win_rate - 0.15:
            # 勝率が低すぎる → 難易度を下げる
            if current_difficulty > 1:
                action = 0

        # リトライ数ベース
        if performance.retry_count > 3:
            action = 0  # 難易度を下げる

        # 完璧クリア
        if performance.success and performance.deaths == 0 and performance.mistakes == 0:
            if current_difficulty < self.difficulty_levels:
                action = 2  # 難易度を上げる

        return action

    def train_from_data(
        self,
        training_data: List[Tuple[PlayerPerformance, str]],
        n_epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ):
        """
        収集したデータからオフライン学習.

        Args:
            training_data: (パフォーマンス, プレイヤーID) のリスト
            n_epochs: エポック数
            batch_size: バッチサイズ
            verbose: 詳細ログ
        """
        if not self.use_rl or self.agent is None:
            raise ValueError("RL agent not initialized")

        if verbose:
            print(f"\nTraining difficulty tuner from {len(training_data)} samples...")

        # データを処理してエピソードを作成
        for epoch in range(n_epochs):
            # データをシャッフル
            np.random.shuffle(training_data)

            # プロファイルをリセット
            temp_profiles = {}
            temp_difficulties = {}

            epoch_rewards = []

            for performance, player_id in training_data:
                # プロファイル取得/作成
                if player_id not in temp_profiles:
                    temp_profiles[player_id] = PlayerProfile()
                    temp_difficulties[player_id] = self.difficulty_levels // 2

                profile = temp_profiles[player_id]
                current_diff = temp_difficulties[player_id]

                # 状態作成
                state = DifficultyState.from_performance_and_profile(
                    performance,
                    profile,
                    current_diff,
                    self.difficulty_levels
                )

                # 行動選択
                action, log_prob, value = self.agent.select_action(state)

                # 新しい難易度
                difficulty_change = int(action) - 1
                new_diff = np.clip(current_diff + difficulty_change, 1, self.difficulty_levels)

                # 報酬計算
                reward = DifficultyReward.compute(
                    performance,
                    profile,
                    current_diff,
                    new_diff,
                    self.target_win_rate,
                    self.target_completion_time
                )

                epoch_rewards.append(reward)

                # バッファに追加
                done = False  # 難易度調整は継続的なタスク
                self.agent.buffer.add(state, action, reward, value, log_prob, done)

                # プロファイル更新
                profile.update(performance)
                temp_difficulties[player_id] = new_diff

                # バッファがいっぱいになったら更新
                if len(self.agent.buffer) >= self.agent.n_steps:
                    self.agent.update()

            if verbose and (epoch + 1) % 10 == 0:
                avg_reward = np.mean(epoch_rewards)
                print(f"Epoch {epoch + 1}/{n_epochs} - Avg Reward: {avg_reward:.2f}")

        if verbose:
            print("✓ Training completed!")

    def get_difficulty_explanation(
        self,
        player_id: str = "default"
    ) -> Dict[str, Any]:
        """
        現在の難易度設定の説明を取得.

        デバッグやUI表示用.

        Args:
            player_id: プレイヤーID

        Returns:
            explanation: 説明情報
        """
        profile = self.get_or_create_profile(player_id)
        current_diff = self.current_difficulties[player_id]

        explanation = {
            'current_difficulty': current_diff,
            'difficulty_name': self._get_difficulty_name(current_diff),
            'skill_level': f"{profile.skill_level:.2f}",
            'win_rate': f"{profile.recent_win_rate:.1%}",
            'avg_deaths': f"{profile.avg_deaths:.1f}",
            'total_attempts': profile.total_attempts,
            'recommendation': self._get_recommendation(profile, current_diff)
        }

        return explanation

    def _get_difficulty_name(self, difficulty: int) -> str:
        """難易度レベルを名前に変換."""
        names = {
            1: "Very Easy",
            2: "Easy",
            3: "Normal",
            4: "Hard",
            5: "Very Hard"
        }
        # 5段階以上の場合は動的に生成
        if difficulty <= len(names):
            return names.get(difficulty, "Normal")
        else:
            percentage = (difficulty / self.difficulty_levels) * 100
            return f"Level {difficulty} ({percentage:.0f}%)"

    def _get_recommendation(self, profile: PlayerProfile, current_diff: int) -> str:
        """推奨メッセージを生成."""
        if profile.recent_win_rate > self.target_win_rate + 0.2:
            return "You're doing great! Consider increasing difficulty for more challenge."
        elif profile.recent_win_rate < self.target_win_rate - 0.2:
            return "Having trouble? We can lower the difficulty to help."
        else:
            return "Difficulty is well-balanced for your skill level."

    def save(self, filepath: str):
        """モデルとプロファイルを保存."""
        save_data = {
            'difficulty_levels': self.difficulty_levels,
            'target_win_rate': self.target_win_rate,
            'target_completion_time': self.target_completion_time,
            'algorithm': self.algorithm,
            'use_rl': self.use_rl,
            'profiles': {
                player_id: {
                    'skill_level': profile.skill_level,
                    'total_playtime': profile.total_playtime,
                    'total_attempts': profile.total_attempts,
                    'total_successes': profile.total_successes,
                    'recent_win_rate': profile.recent_win_rate,
                }
                for player_id, profile in self.profiles.items()
            },
            'current_difficulties': self.current_difficulties,
        }

        # メタデータを保存
        meta_filepath = filepath.replace('.pth', '_meta.json')
        with open(meta_filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

        # RLエージェントを保存
        if self.use_rl and self.agent is not None:
            self.agent.save(filepath)

        print(f"✓ Difficulty tuner saved to {filepath}")
        print(f"✓ Metadata saved to {meta_filepath}")

    def load(self, filepath: str):
        """モデルとプロファイルを読み込み."""
        # メタデータを読み込み
        meta_filepath = filepath.replace('.pth', '_meta.json')
        with open(meta_filepath, 'r') as f:
            save_data = json.load(f)

        self.difficulty_levels = save_data['difficulty_levels']
        self.target_win_rate = save_data['target_win_rate']
        self.target_completion_time = save_data['target_completion_time']
        self.algorithm = save_data['algorithm']
        self.use_rl = save_data['use_rl']

        # プロファイルを復元
        for player_id, profile_data in save_data['profiles'].items():
            profile = PlayerProfile()
            profile.skill_level = profile_data['skill_level']
            profile.total_playtime = profile_data['total_playtime']
            profile.total_attempts = profile_data['total_attempts']
            profile.total_successes = profile_data['total_successes']
            profile.recent_win_rate = profile_data['recent_win_rate']
            self.profiles[player_id] = profile

        self.current_difficulties = save_data['current_difficulties']

        # RLエージェントを読み込み
        if self.use_rl and self.agent is not None:
            self.agent.load(filepath)

        print(f"✓ Difficulty tuner loaded from {filepath}")

    def export_analytics(self, filepath: str):
        """分析用データをエクスポート."""
        analytics = {
            'adjustment_history': self.adjustment_history,
            'player_profiles': {
                player_id: {
                    'skill_level': profile.skill_level,
                    'total_attempts': profile.total_attempts,
                    'win_rate': profile.recent_win_rate,
                    'avg_deaths': profile.avg_deaths,
                }
                for player_id, profile in self.profiles.items()
            },
            'summary': {
                'total_adjustments': len(self.adjustment_history),
                'total_players': len(self.profiles),
                'avg_skill_level': np.mean([p.skill_level for p in self.profiles.values()]) if self.profiles else 0,
                'avg_win_rate': np.mean([p.recent_win_rate for p in self.profiles.values()]) if self.profiles else 0,
            }
        }

        with open(filepath, 'w') as f:
            json.dump(analytics, f, indent=2)

        print(f"✓ Analytics exported to {filepath}")


# ============================================================================
# デモ: ゲームシミュレーションで難易度調整
# ============================================================================

def simulate_game_session(skill_level: float, difficulty: int) -> PlayerPerformance:
    """
    ゲームセッションをシミュレート.

    Args:
        skill_level: プレイヤースキル（0-1）
        difficulty: 難易度（1-5）

    Returns:
        performance: シミュレートされたパフォーマンス
    """
    # スキルと難易度の差でパフォーマンスを決定
    difficulty_normalized = difficulty / 5.0
    skill_difficulty_gap = skill_level - difficulty_normalized

    # 成功確率
    success_prob = 0.5 + skill_difficulty_gap * 0.5
    success_prob = np.clip(success_prob, 0.1, 0.9)
    success = np.random.random() < success_prob

    # 完了時間（スキルが高いほど速い）
    base_time = 180.0
    time_factor = 1.0 + (difficulty_normalized - skill_level) * 0.5
    completion_time = base_time * time_factor + np.random.normal(0, 30)
    completion_time = max(30, completion_time)

    # 死亡回数
    death_prob = (1.0 - skill_level) * difficulty_normalized
    deaths = int(np.random.poisson(death_prob * 5))

    # スコア
    base_score = 1000
    score_factor = success_prob * (1.0 + skill_level - difficulty_normalized * 0.5)
    score = base_score * score_factor + np.random.normal(0, 100)
    score = max(0, score)

    return PlayerPerformance(
        success=success,
        completion_time=completion_time,
        deaths=deaths,
        score=score,
        retry_count=deaths
    )


if __name__ == "__main__":
    print("Dynamic Difficulty Adjustment Demo")
    print("=" * 60)

    # チューナー作成
    tuner = DifficultyTuner(
        difficulty_levels=5,
        target_win_rate=0.6,
        target_completion_time=180.0,
        use_rl=False,  # まずヒューリスティックでテスト
    )

    # プレイヤー3人をシミュレート
    players = {
        'beginner': 0.3,    # 初心者
        'intermediate': 0.6, # 中級者
        'expert': 0.9,      # 上級者
    }

    print("\nSimulating game sessions...")
    print("-" * 60)

    # 各プレイヤーで10セッションをシミュレート
    for player_id, skill in players.items():
        print(f"\n{player_id.capitalize()} (Skill: {skill:.1f})")

        for session in range(10):
            # 現在の難易度を取得
            current_diff = tuner.current_difficulties.get(player_id, 3)

            # ゲームセッションをシミュレート
            performance = simulate_game_session(skill, current_diff)

            # 難易度を調整
            new_diff = tuner.adjust_difficulty(performance, player_id)

            # 結果を表示
            status = "✓" if performance.success else "✗"
            print(f"  Session {session + 1}: {status} Difficulty {current_diff} → {new_diff} "
                  f"(Time: {performance.completion_time:.0f}s, Deaths: {performance.deaths})")

        # プレイヤーの最終状態を表示
        explanation = tuner.get_difficulty_explanation(player_id)
        print(f"\n  Final Stats:")
        print(f"    Difficulty: {explanation['difficulty_name']}")
        print(f"    Skill Level: {explanation['skill_level']}")
        print(f"    Win Rate: {explanation['win_rate']}")
        print(f"    Recommendation: {explanation['recommendation']}")

    # 保存
    tuner.save('difficulty_tuner_demo.pth')

    # 分析をエクスポート
    tuner.export_analytics('difficulty_analytics.json')

    print("\n" + "=" * 60)
    print("✓ Demo completed!")
    print("\nNext steps:")
    print("  1. Integrate with your game engine")
    print("  2. Collect real player data")
    print("  3. Train RL model with collected data: tuner.train_from_data()")
    print("  4. Use trained model for dynamic adjustment")
