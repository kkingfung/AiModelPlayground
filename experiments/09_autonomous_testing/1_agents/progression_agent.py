"""
Progression Agent

レベル進行テストエージェント.
ゲームのレベルが完了可能か、難易度が適切かを検証します.

使い方:
    from progression_agent import ProgressionAgent

    agent = ProgressionAgent(
        goal_state={'level': 'Level_2'},  # ゴール条件
        max_attempts=100,
        difficulty_threshold=0.3  # 30%未満の達成率で「難しすぎ」判定
    )

    results = agent.run(
        get_state_fn=get_game_state,
        take_action_fn=execute_action,
        is_terminal_fn=check_game_over,
        episodes=100
    )

    # レポート
    print(f"Completion rate: {results['completion_rate']:.1%}")
    print(f"Average attempts: {results['avg_attempts']:.1f}")
    print(f"Difficulty rating: {results['difficulty']}/10")

戦略:
    - ゴール指向の探索（A*、Dijkstra）
    - 複数の異なるアプローチを試行
    - 失敗箇所の特定
    - 難易度の客観的測定
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict, deque
from pathlib import Path
import json
import heapq

from base_agent import (
    BaseTestingAgent,
    GameState,
    Bug,
    create_bug
)


class ProgressionAgent(BaseTestingAgent):
    """
    レベル進行テスト特化型エージェント.

    目標:
        - レベルが完了可能かを検証
        - 難易度を客観的に測定
        - ソフトロック（進行不可状態）を検出
        - 最適ルートを発見

    戦略:
        - ゴール指向探索（A*）
        - 複数のアプローチ（探索/記憶/ランダム）
        - 失敗分析
        - 完了率と時間の計測
    """

    def __init__(
        self,
        name: str = "ProgressionAgent",
        goal_state: Optional[Dict[str, Any]] = None,
        max_attempts_per_episode: int = 3,
        difficulty_threshold: float = 0.3,
        search_strategy: str = 'mixed',  # 'astar', 'greedy', 'mixed', 'random'
        adaptive_learning: bool = True,
        verbose: bool = True,
        save_screenshots: bool = True,
        screenshot_dir: str = "screenshots/progression"
    ):
        """
        Args:
            name: エージェント名
            goal_state: ゴール条件（辞書）
            max_attempts_per_episode: エピソードあたりの試行回数
            difficulty_threshold: 難易度判定閾値
            search_strategy: 探索戦略
            adaptive_learning: 適応学習有効化
            verbose: 詳細ログ
            save_screenshots: スクリーンショット保存
            screenshot_dir: 保存先
        """
        super().__init__(name, verbose, save_screenshots, screenshot_dir)

        # ゴール設定
        self.goal_state = goal_state or {}
        self.max_attempts_per_episode = max_attempts_per_episode
        self.difficulty_threshold = difficulty_threshold
        self.search_strategy = search_strategy
        self.adaptive_learning = adaptive_learning

        # 行動リスト
        self.available_actions = []

        # 進行記録
        self.completions = []
        self.attempts_history = []
        self.failure_points = defaultdict(int)
        self.success_paths = []

        # 学習データ
        self.state_values = defaultdict(float)  # 各状態の価値
        self.action_success_rate = defaultdict(lambda: defaultdict(lambda: {'success': 0, 'total': 0}))

        # 現在の試行
        self.current_attempt = 0
        self.current_episode_attempts = 0

        # 統計
        self.progression_stats = {
            'levels_completed': 0,
            'total_attempts': 0,
            'soft_locks_detected': 0,
            'impossible_sections_found': 0,
            'optimal_path_length': float('inf')
        }

    def set_available_actions(self, actions: List[Any]):
        """利用可能な行動リストを設定."""
        self.available_actions = actions
        if self.verbose:
            print(f"[{self.name}] Available actions set: {actions}")

    def set_goal_state(self, goal_state: Dict[str, Any]):
        """ゴール条件を設定."""
        self.goal_state = goal_state
        if self.verbose:
            print(f"[{self.name}] Goal state set: {goal_state}")

    def select_action(self, state: GameState) -> Any:
        """
        ゴール指向の行動選択.

        戦略:
            - astar: A*探索（ヒューリスティック使用）
            - greedy: 貪欲法（常に最良の選択）
            - mixed: 探索とランダムのミックス
            - random: ランダム
        """
        if not self.available_actions:
            self.available_actions = list(range(4))

        if self.search_strategy == 'astar':
            return self._astar_action(state)
        elif self.search_strategy == 'greedy':
            return self._greedy_action(state)
        elif self.search_strategy == 'mixed':
            # 80%でゴール指向、20%でランダム探索
            if np.random.random() < 0.8:
                return self._greedy_action(state)
            else:
                return self._random_action()
        else:
            return self._random_action()

    def _random_action(self) -> Any:
        """ランダムな行動選択."""
        return np.random.choice(self.available_actions)

    def _greedy_action(self, state: GameState) -> Any:
        """
        貪欲な行動選択.

        学習した行動成功率に基づいて最良の行動を選択.
        """
        state_hash = state.hash()

        # 各行動のスコアを計算
        action_scores = []
        for action in self.available_actions:
            stats = self.action_success_rate[state_hash][str(action)]

            if stats['total'] == 0:
                # 未試行の行動には探索ボーナス
                score = 1.0
            else:
                # 成功率をスコアとする
                success_rate = stats['success'] / stats['total']
                score = success_rate

            # ゴールへの距離を考慮（ヒューリスティック）
            heuristic = self._heuristic(state)
            score += 0.1 / (heuristic + 1)  # 距離が近いほど高スコア

            action_scores.append(score)

        # 最高スコアの行動を選択
        best_idx = np.argmax(action_scores)
        return self.available_actions[best_idx]

    def _astar_action(self, state: GameState) -> Any:
        """
        A*探索に基づく行動選択.

        実際のA*実装は複雑なので、簡易版としてヒューリスティックを使用.
        """
        state_hash = state.hash()

        # 各行動を試した場合のヒューリスティックを計算
        action_scores = []
        for action in self.available_actions:
            # この行動を選んだときの推定コスト
            cost = 1.0  # 行動のコスト
            h = self._heuristic(state)  # ゴールまでの推定距離

            # f(n) = g(n) + h(n)
            f_score = cost + h

            action_scores.append(-f_score)  # スコアとして使うため反転

        # 最良のf値を持つ行動を選択
        best_idx = np.argmax(action_scores)
        return self.available_actions[best_idx]

    def _heuristic(self, state: GameState) -> float:
        """
        ゴールまでの推定距離（ヒューリスティック）.

        位置ベースのゲームの場合、ユークリッド距離を使用.
        """
        # ゴール位置が設定されている場合
        if 'goal_position' in self.goal_state:
            current_pos = state.get('player_position')
            goal_pos = self.goal_state['goal_position']

            if current_pos and goal_pos:
                distance = np.linalg.norm(
                    np.array(current_pos) - np.array(goal_pos)
                )
                return distance

        # レベル名ベースのゴール
        if 'level' in self.goal_state:
            current_level = state.get('level')
            if current_level == self.goal_state['level']:
                return 0.0
            else:
                return 100.0  # 未達成

        # デフォルト
        return 10.0

    def detect_bugs(self, state: GameState, next_state: GameState) -> List[Bug]:
        """
        状態遷移からバグを検出.

        検出するバグ:
            - ソフトロック（進行不可状態）
            - 不可能なジャンプ/セクション
            - ゴール到達不可
        """
        bugs = []

        # 1. ソフトロック検出
        softlock_bug = self._detect_softlock(state, next_state)
        if softlock_bug:
            bugs.append(softlock_bug)
            self.progression_stats['soft_locks_detected'] += 1

        # 2. 同じ場所で複数回失敗（難しすぎる可能性）
        failure_bug = self._detect_repeated_failure(state)
        if failure_bug:
            bugs.append(failure_bug)

        # 3. ゴール到達の可能性チェック
        reachability_bug = self._check_goal_reachability(state, next_state)
        if reachability_bug:
            bugs.append(reachability_bug)

        return bugs

    def _detect_softlock(self, state: GameState, next_state: GameState) -> Optional[Bug]:
        """
        ソフトロック（進行不可状態）を検出.

        同じ状態から抜け出せない場合にソフトロック.
        """
        state_hash = state.hash()
        next_hash = next_state.hash()

        # すべての行動を試しても状態が変わらない
        if state_hash == next_hash:
            # この状態から何回試行したか
            attempts_from_state = sum(
                stats['total']
                for stats in self.action_success_rate[state_hash].values()
            )

            if attempts_from_state > len(self.available_actions) * 3:
                # すべての行動を3回以上試しても抜け出せない
                return create_bug(
                    severity='critical',
                    title='Soft-lock detected',
                    description=f'Player cannot escape from state {state_hash}. '
                                f'Tried {attempts_from_state} times.',
                    state=state,
                    steps=self.action_history[-10:]
                )

        return None

    def _detect_repeated_failure(self, state: GameState) -> Optional[Bug]:
        """
        同じ場所での繰り返し失敗を検出（難しすぎる可能性）.
        """
        state_hash = state.hash()
        failure_count = self.failure_points[state_hash]

        if failure_count > 10:
            return create_bug(
                severity='medium',
                title='Difficult section detected',
                description=f'Failed {failure_count} times at this location. '
                            'May be too difficult or require specific strategy.',
                state=state
            )

        return None

    def _check_goal_reachability(self, state: GameState, next_state: GameState) -> Optional[Bug]:
        """
        ゴール到達可能性をチェック.

        多くの試行を経てもゴールに近づかない場合、到達不可の可能性.
        """
        if self.current_step > 500:
            # 500ステップ経過
            current_h = self._heuristic(next_state)

            # 初期状態からのヒューリスティックと比較
            if self.state_history:
                initial_h = self._heuristic(self.state_history[0])

                # ゴールに全く近づいていない
                if current_h >= initial_h * 0.9:
                    return create_bug(
                        severity='high',
                        title='Goal may be unreachable',
                        description=f'After {self.current_step} steps, distance to goal: '
                                    f'{current_h:.2f} (initial: {initial_h:.2f})',
                        state=state
                    )

        return None

    def _check_goal_reached(self, state: GameState) -> bool:
        """
        ゴール到達判定.

        goal_stateの条件がすべて満たされているかチェック.
        """
        for key, value in self.goal_state.items():
            if state.get(key) != value:
                return False
        return True

    def run_episode(
        self,
        get_state_fn,
        take_action_fn,
        is_terminal_fn,
        max_steps: int = 1000
    ) -> Dict:
        """
        1エピソードを実行（複数試行含む）.
        """
        self.current_episode_attempts = 0
        episode_results = []

        for attempt in range(self.max_attempts_per_episode):
            self.current_episode_attempts += 1
            self.progression_stats['total_attempts'] += 1

            # 1回の試行
            result = super().run_episode(
                get_state_fn,
                take_action_fn,
                is_terminal_fn,
                max_steps
            )

            # ゴール到達チェック
            if self.state_history:
                final_state = self.state_history[-1]
                if self._check_goal_reached(final_state):
                    result['goal_reached'] = True
                    self.progression_stats['levels_completed'] += 1
                    self.completions.append(self.current_step)
                    self.success_paths.append(self.action_history.copy())

                    # 最適パス更新
                    if self.current_step < self.progression_stats['optimal_path_length']:
                        self.progression_stats['optimal_path_length'] = self.current_step

                    if self.verbose:
                        print(f"✓ [{self.name}] Goal reached in {self.current_step} steps!")

                    # 成功したら学習
                    if self.adaptive_learning:
                        self._learn_from_success()

                    break  # 成功したので次のエピソードへ
                else:
                    result['goal_reached'] = False
                    # 失敗地点を記録
                    failure_hash = final_state.hash()
                    self.failure_points[failure_hash] += 1

                    if self.verbose:
                        print(f"✗ [{self.name}] Failed attempt {attempt + 1}/{self.max_attempts_per_episode}")

            episode_results.append(result)

        self.attempts_history.append(self.current_episode_attempts)

        # エピソード結果をまとめる
        return {
            'episode': self.current_episode,
            'attempts': self.current_episode_attempts,
            'goal_reached': any(r.get('goal_reached', False) for r in episode_results),
            'episode_results': episode_results
        }

    def _learn_from_success(self):
        """
        成功した経路から学習.

        成功につながった行動の成功率を更新.
        """
        for i, state in enumerate(self.state_history):
            if i < len(self.action_history):
                action = self.action_history[i]
                state_hash = state.hash()

                stats = self.action_success_rate[state_hash][str(action)]
                stats['success'] += 1
                stats['total'] += 1

    def get_progression_report(self) -> Dict:
        """
        進行レポートを生成.

        Returns:
            完了率、平均試行回数、難易度評価など
        """
        total_episodes = len(self.attempts_history)
        if total_episodes == 0:
            return {
                'completion_rate': 0.0,
                'avg_attempts': 0.0,
                'difficulty': 10
            }

        completion_rate = self.progression_stats['levels_completed'] / total_episodes
        avg_attempts = np.mean(self.attempts_history)

        # 難易度評価（0-10）
        # 完了率が高い & 試行回数が少ない = 易しい
        # 完了率が低い & 試行回数が多い = 難しい
        if completion_rate > 0.8:
            difficulty = 1 + (avg_attempts / self.max_attempts_per_episode) * 2
        elif completion_rate > 0.5:
            difficulty = 4 + (avg_attempts / self.max_attempts_per_episode) * 2
        elif completion_rate > 0.2:
            difficulty = 7 + (avg_attempts / self.max_attempts_per_episode)
        else:
            difficulty = 9 + min(1, (1 - completion_rate) * 10)

        difficulty = min(10, max(1, difficulty))

        # 診断メッセージ
        diagnosis = []
        if completion_rate < self.difficulty_threshold:
            diagnosis.append("⚠️ Level may be too difficult or impossible")
        if self.progression_stats['soft_locks_detected'] > 0:
            diagnosis.append(f"⚠️ {self.progression_stats['soft_locks_detected']} soft-lock(s) detected")
        if completion_rate > 0.95 and avg_attempts < 1.5:
            diagnosis.append("✓ Level may be too easy")

        # 失敗の多い場所
        top_failure_points = sorted(
            self.failure_points.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            'completion_rate': completion_rate,
            'avg_attempts': avg_attempts,
            'difficulty': difficulty,
            'total_episodes': total_episodes,
            'completions': self.progression_stats['levels_completed'],
            'avg_completion_time': np.mean(self.completions) if self.completions else 0,
            'optimal_path_length': self.progression_stats['optimal_path_length'],
            'progression_stats': self.progression_stats,
            'diagnosis': diagnosis,
            'top_failure_points': [
                {'state_hash': h[:16], 'failures': count}
                for h, count in top_failure_points
            ]
        }

    def save_results(self, filepath: str):
        """テスト結果を保存（進行レポート含む）."""
        results = {
            'agent_name': self.name,
            'goal_state': self.goal_state,
            'search_strategy': self.search_strategy,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': self.metrics.get_summary(),
            'progression_report': self.get_progression_report(),
            'bugs': [bug.to_dict() for bug in self.metrics.bugs_found]
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"[{self.name}] Results saved to {filepath}")


# ヘルパー関数

def run_progression_test(
    get_state_fn: Callable,
    take_action_fn: Callable,
    is_terminal_fn: Callable,
    available_actions: List[Any],
    goal_state: Dict[str, Any],
    episodes: int = 100,
    max_steps: int = 1000,
    output_dir: str = 'results/progression'
) -> Dict:
    """
    進行テストを実行.
    """
    agent = ProgressionAgent(
        goal_state=goal_state,
        search_strategy='mixed',
        verbose=True
    )

    agent.set_available_actions(available_actions)

    results = agent.run(
        get_state_fn=get_state_fn,
        take_action_fn=take_action_fn,
        is_terminal_fn=is_terminal_fn,
        episodes=episodes,
        max_steps=max_steps
    )

    # 結果保存
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    agent.save_results(f"{output_dir}/progression_results.json")

    return results


if __name__ == "__main__":
    # デモ: ゴール到達テスト
    print("Progression Agent Demo")
    print("=" * 60)

    # ゴール付きグリッドワールド
    class GoalGridWorld:
        def __init__(self, size=10):
            self.size = size
            self.start = [0, 0]
            self.goal = [size - 1, size - 1]
            self.position = self.start.copy()
            self.obstacles = {(3, 3), (4, 4), (5, 5)}

        def get_state(self):
            return {
                'player_position': self.position.copy(),
                'level': 'Level_1',
                'health': 100
            }

        def take_action(self, action):
            # 0=up, 1=right, 2=down, 3=left
            moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
            move = moves[action]

            new_pos = [
                self.position[0] + move[0],
                self.position[1] + move[1]
            ]

            # 境界・障害物チェック
            if (0 <= new_pos[0] < self.size and
                0 <= new_pos[1] < self.size and
                tuple(new_pos) not in self.obstacles):
                self.position = new_pos

        def is_terminal(self, state):
            return state['player_position'] == self.goal

        def reset(self):
            self.position = self.start.copy()

    # 環境作成
    env = GoalGridWorld(size=8)

    # エージェント作成
    agent = ProgressionAgent(
        goal_state={'player_position': [7, 7]},
        search_strategy='mixed',
        max_attempts_per_episode=3,
        verbose=True
    )

    agent.set_available_actions([0, 1, 2, 3])

    # テスト実行
    results = agent.run(
        get_state_fn=env.get_state,
        take_action_fn=env.take_action,
        is_terminal_fn=env.is_terminal,
        episodes=20,
        max_steps=100
    )

    # 結果表示
    print("\n" + "=" * 60)
    print("Progression Report:")
    print("=" * 60)
    report = agent.get_progression_report()
    print(f"Completion rate: {report['completion_rate']:.1%}")
    print(f"Average attempts: {report['avg_attempts']:.2f}")
    print(f"Difficulty rating: {report['difficulty']:.1f}/10")
    print(f"Optimal path length: {report['optimal_path_length']} steps")

    if report['diagnosis']:
        print(f"\nDiagnosis:")
        for msg in report['diagnosis']:
            print(f"  {msg}")

    # 結果保存
    agent.save_results("demo_progression_results.json")

    print("\n✓ Demo completed!")
