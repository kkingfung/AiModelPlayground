"""
Exploration Agent

カバレッジ駆動の探索エージェント.
新しい状態を発見し、ゲームの状態空間を網羅的にテストします.

使い方:
    from exploration_agent import ExplorationAgent

    agent = ExplorationAgent(
        exploration_strategy='curiosity',  # 'random', 'curiosity', 'epsilon_greedy'
        novelty_threshold=0.8
    )

    results = agent.run(
        get_state_fn=get_game_state,
        take_action_fn=execute_action,
        is_terminal_fn=check_game_over,
        episodes=1000
    )

戦略:
    - Random Walk: ランダムな行動選択
    - Curiosity-Driven: 新規状態を優先的に探索
    - Epsilon-Greedy: 探索と活用のバランス
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from collections import defaultdict
from pathlib import Path
import json

from base_agent import (
    BaseTestingAgent,
    GameState,
    Bug,
    create_bug
)


class ExplorationAgent(BaseTestingAgent):
    """
    探索特化型テストエージェント.

    目標:
        - ゲームの状態空間を最大限カバー
        - 未訪問エリアを発見
        - 隠れたバグやエクスプロイトを検出

    戦略:
        - 新規状態の優先探索（好奇心駆動）
        - 訪問回数の少ない状態を重視
        - エントロピー最大化
    """

    def __init__(
        self,
        name: str = "ExplorationAgent",
        exploration_strategy: str = "curiosity",
        epsilon: float = 0.3,
        novelty_threshold: float = 0.8,
        curiosity_bonus: float = 10.0,
        verbose: bool = True,
        save_screenshots: bool = True,
        screenshot_dir: str = "screenshots/exploration"
    ):
        """
        Args:
            name: エージェント名
            exploration_strategy: 探索戦略 ('random', 'curiosity', 'epsilon_greedy')
            epsilon: ε-greedy戦略のε値
            novelty_threshold: 新規状態と判定する閾値
            curiosity_bonus: 好奇心ボーナスの重み
            verbose: 詳細ログ
            save_screenshots: スクリーンショット保存
            screenshot_dir: 保存先ディレクトリ
        """
        super().__init__(name, verbose, save_screenshots, screenshot_dir)

        # 探索戦略
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.novelty_threshold = novelty_threshold
        self.curiosity_bonus = curiosity_bonus

        # 探索記録
        self.state_graph = defaultdict(set)  # 状態遷移グラフ
        self.action_outcomes = defaultdict(lambda: defaultdict(int))  # 行動結果
        self.boundary_states = set()  # 境界状態（探索フロンティア）

        # 利用可能な行動リスト（外部から設定可能）
        self.available_actions = []

        # 統計
        self.exploration_stats = {
            'new_states_found': 0,
            'dead_ends_found': 0,
            'loops_detected': 0,
            'max_depth_reached': 0
        }

    def set_available_actions(self, actions: List[Any]):
        """
        利用可能な行動リストを設定.

        Args:
            actions: 行動のリスト (例: ['UP', 'DOWN', 'LEFT', 'RIGHT'])
        """
        self.available_actions = actions
        if self.verbose:
            print(f"[{self.name}] Available actions set: {actions}")

    def select_action(self, state: GameState) -> Any:
        """
        状態から行動を選択.

        戦略:
            - random: 完全ランダム
            - curiosity: 新規状態につながりそうな行動を優先
            - epsilon_greedy: ε確率でランダム、それ以外は最良行動
        """
        if not self.available_actions:
            # デフォルト: 数値行動 (0-3)
            self.available_actions = list(range(4))

        if self.exploration_strategy == 'random':
            return self._random_action()

        elif self.exploration_strategy == 'curiosity':
            return self._curiosity_driven_action(state)

        elif self.exploration_strategy == 'epsilon_greedy':
            if np.random.random() < self.epsilon:
                return self._random_action()
            else:
                return self._greedy_action(state)

        else:
            # デフォルトはランダム
            return self._random_action()

    def _random_action(self) -> Any:
        """完全ランダムな行動選択."""
        return np.random.choice(self.available_actions)

    def _curiosity_driven_action(self, state: GameState) -> Any:
        """
        好奇心駆動の行動選択.

        未訪問状態につながりそうな行動を優先的に選択.
        """
        state_hash = state.hash()

        # 各行動の「新規性スコア」を計算
        action_scores = []
        for action in self.available_actions:
            # この状態・行動ペアを試した回数
            tried_count = self.action_outcomes[state_hash][str(action)]

            # 試していない行動ほど高スコア
            novelty_score = 1.0 / (tried_count + 1)

            action_scores.append(novelty_score)

        # ソフトマックスで確率化
        action_scores = np.array(action_scores)
        exp_scores = np.exp(action_scores - np.max(action_scores))  # 数値安定化
        probabilities = exp_scores / exp_scores.sum()

        # 確率に基づいて選択
        action_idx = np.random.choice(len(self.available_actions), p=probabilities)
        return self.available_actions[action_idx]

    def _greedy_action(self, state: GameState) -> Any:
        """
        貪欲な行動選択.

        最も訪問回数の少ない行動を選択.
        """
        state_hash = state.hash()

        min_count = float('inf')
        best_actions = []

        for action in self.available_actions:
            count = self.action_outcomes[state_hash][str(action)]
            if count < min_count:
                min_count = count
                best_actions = [action]
            elif count == min_count:
                best_actions.append(action)

        return np.random.choice(best_actions)

    def detect_bugs(self, state: GameState, next_state: GameState) -> List[Bug]:
        """
        状態遷移からバグを検出.

        検出するバグ:
            - 到達不可能エリア（デッドエンド）
            - 無限ループ
            - 異常な状態遷移
            - 境界外アクセス
        """
        bugs = []

        # 状態ハッシュ
        state_hash = state.hash()
        next_hash = next_state.hash()

        # 状態グラフ更新
        self.state_graph[state_hash].add(next_hash)

        # 1. 新規状態の検出
        if next_hash not in self.metrics.unique_states:
            self.exploration_stats['new_states_found'] += 1

        # 2. デッドエンド検出（同じ状態に戻る）
        if state_hash == next_hash:
            self.exploration_stats['dead_ends_found'] += 1

            bugs.append(create_bug(
                severity='low',
                title='Possible dead-end or stuck state',
                description=f'Action did not change state. State hash: {state_hash}',
                state=state,
                steps=self.action_history[-5:] if len(self.action_history) >= 5 else self.action_history
            ))

        # 3. 無限ループ検出
        if self._detect_loop(state_hash, next_hash):
            self.exploration_stats['loops_detected'] += 1

            bugs.append(create_bug(
                severity='medium',
                title='Infinite loop detected',
                description=f'Agent is stuck in a loop: {state_hash} -> {next_hash}',
                state=state,
                steps=self.action_history[-10:]
            ))

        # 4. 物理異常検出（位置が大きく変化）
        position_bug = self._check_position_anomaly(state, next_state)
        if position_bug:
            bugs.append(position_bug)

        # 5. 境界外アクセス
        bounds_bug = self._check_out_of_bounds(state, next_state)
        if bounds_bug:
            bugs.append(bounds_bug)

        # 6. 健康値異常
        health_bug = self._check_health_anomaly(state, next_state)
        if health_bug:
            bugs.append(health_bug)

        return bugs

    def _detect_loop(self, state_hash: str, next_hash: str, lookback: int = 5) -> bool:
        """
        ループを検出.

        最近の状態履歴から繰り返しパターンを検出.
        """
        if len(self.state_history) < lookback * 2:
            return False

        # 最近の状態ハッシュを取得
        recent_hashes = [s.hash() for s in self.state_history[-lookback * 2:]]

        # パターン検出
        pattern = recent_hashes[:lookback]
        next_pattern = recent_hashes[lookback:]

        return pattern == next_pattern

    def _check_position_anomaly(
        self,
        state: GameState,
        next_state: GameState
    ) -> Optional[Bug]:
        """
        位置の異常な変化を検出（テレポート、壁抜けなど）.
        """
        pos = state.get('player_position')
        next_pos = next_state.get('player_position')

        if pos is None or next_pos is None:
            return None

        # 距離計算
        distance = np.linalg.norm(np.array(next_pos) - np.array(pos))

        # 異常な移動距離（閾値は調整可能）
        if distance > 50.0:
            return create_bug(
                severity='high',
                title='Abnormal position change detected',
                description=f'Player teleported or clipped through wall. '
                            f'Distance: {distance:.2f} units. '
                            f'From {pos} to {next_pos}',
                state=state,
                steps=self.action_history[-3:]
            )

        return None

    def _check_out_of_bounds(
        self,
        state: GameState,
        next_state: GameState
    ) -> Optional[Bug]:
        """
        境界外アクセスを検出.
        """
        pos = next_state.get('player_position')

        if pos is None:
            return None

        # Y座標が極端に低い（落下）
        if pos[1] < -100:
            return create_bug(
                severity='critical',
                title='Player fell out of world',
                description=f'Player position Y: {pos[1]} (below -100)',
                state=state,
                steps=self.action_history[-5:]
            )

        # 極端に大きな座標
        max_coord = max(abs(p) for p in pos)
        if max_coord > 10000:
            return create_bug(
                severity='high',
                title='Player position out of bounds',
                description=f'Player position: {pos} (coordinate > 10000)',
                state=state,
                steps=self.action_history[-5:]
            )

        return None

    def _check_health_anomaly(
        self,
        state: GameState,
        next_state: GameState
    ) -> Optional[Bug]:
        """
        健康値の異常を検出.
        """
        health = state.get('health')
        next_health = next_state.get('health')

        if health is None or next_health is None:
            return None

        # 負の健康値
        if next_health < 0:
            return create_bug(
                severity='high',
                title='Negative health value',
                description=f'Health became negative: {next_health}',
                state=state
            )

        # 極端な回復
        if next_health - health > 500:
            return create_bug(
                severity='medium',
                title='Abnormal health increase',
                description=f'Health increased by {next_health - health} in one step',
                state=state
            )

        return None

    def _calculate_reward(
        self,
        state: GameState,
        next_state: GameState,
        action: Any
    ) -> float:
        """
        報酬計算（好奇心ベース）.

        新しい状態を訪れるほど高報酬.
        """
        base_reward = super()._calculate_reward(state, next_state, action)

        # 新規状態ボーナス
        next_hash = next_state.hash()
        if next_hash not in self.metrics.unique_states:
            base_reward += self.curiosity_bonus

        # 訪問回数が少ない状態ほど高報酬
        visit_count = self.metrics.state_visitation_count.get(next_hash, 0)
        novelty_reward = 1.0 / (visit_count + 1)

        return base_reward + novelty_reward

    def get_coverage_report(self) -> Dict:
        """
        カバレッジレポートを生成.

        Returns:
            カバレッジ統計の辞書
        """
        total_states = len(self.metrics.unique_states)
        total_transitions = sum(len(neighbors) for neighbors in self.state_graph.values())

        # 孤立状態（出口のない状態）
        dead_end_states = [
            state for state, neighbors in self.state_graph.items()
            if len(neighbors) == 0
        ]

        # 密度（平均遷移数）
        avg_transitions = total_transitions / total_states if total_states > 0 else 0

        return {
            'total_states_discovered': total_states,
            'total_transitions': total_transitions,
            'avg_transitions_per_state': avg_transitions,
            'dead_end_states': len(dead_end_states),
            'exploration_stats': self.exploration_stats,
            'state_graph_size': len(self.state_graph)
        }

    def save_results(self, filepath: str):
        """
        テスト結果を保存（カバレッジレポート含む）.
        """
        results = {
            'agent_name': self.name,
            'exploration_strategy': self.exploration_strategy,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': self.metrics.get_summary(),
            'coverage_report': self.get_coverage_report(),
            'bugs': [bug.to_dict() for bug in self.metrics.bugs_found]
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"[{self.name}] Results saved to {filepath}")

    def visualize_state_graph(self, output_path: str = "state_graph.png"):
        """
        状態遷移グラフを可視化（オプション）.

        Requires: networkx, matplotlib
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ networkx and matplotlib required for visualization")
            print("Install with: pip install networkx matplotlib")
            return

        # グラフ構築
        G = nx.DiGraph()

        for state, neighbors in self.state_graph.items():
            for neighbor in neighbors:
                G.add_edge(state[:8], neighbor[:8])  # ハッシュの最初の8文字のみ

        # 描画
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        nx.draw(
            G, pos,
            node_size=100,
            node_color='lightblue',
            edge_color='gray',
            with_labels=False,
            arrows=True,
            arrowsize=10,
            alpha=0.7
        )

        plt.title(f"State Graph - {len(G.nodes)} states, {len(G.edges)} transitions")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"✓ State graph saved to {output_path}")


# ヘルパー関数

def run_exploration_test(
    get_state_fn: Callable,
    take_action_fn: Callable,
    is_terminal_fn: Callable,
    available_actions: List[Any],
    episodes: int = 100,
    max_steps: int = 1000,
    strategy: str = 'curiosity',
    output_dir: str = 'results/exploration'
) -> Dict:
    """
    探索テストを実行するヘルパー関数.

    Args:
        get_state_fn: 状態取得関数
        take_action_fn: 行動実行関数
        is_terminal_fn: 終了判定関数
        available_actions: 利用可能な行動リスト
        episodes: エピソード数
        max_steps: 最大ステップ数
        strategy: 探索戦略
        output_dir: 結果出力ディレクトリ

    Returns:
        テスト結果
    """
    agent = ExplorationAgent(
        exploration_strategy=strategy,
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
    agent.save_results(f"{output_dir}/exploration_results.json")

    # グラフ可視化
    agent.visualize_state_graph(f"{output_dir}/state_graph.png")

    return results


if __name__ == "__main__":
    # デモ: GridWorldで探索テスト
    print("Exploration Agent Demo")
    print("=" * 60)

    # ダミーのゲーム環境
    class DummyGridWorld:
        def __init__(self, size=10):
            self.size = size
            self.position = [0, 0]
            self.goal = [size - 1, size - 1]

        def get_state(self):
            return {
                'player_position': self.position.copy(),
                'level': 'demo_level',
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

            # 境界チェック
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                self.position = new_pos

        def is_terminal(self, state):
            return state['player_position'] == self.goal

        def reset(self):
            self.position = [0, 0]

    # 環境作成
    env = DummyGridWorld(size=5)

    # エージェント作成
    agent = ExplorationAgent(
        exploration_strategy='curiosity',
        verbose=True
    )

    agent.set_available_actions([0, 1, 2, 3])  # UP, RIGHT, DOWN, LEFT

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
    print("Coverage Report:")
    print("=" * 60)
    coverage = agent.get_coverage_report()
    print(f"States discovered: {coverage['total_states_discovered']}")
    print(f"Transitions: {coverage['total_transitions']}")
    print(f"Avg transitions per state: {coverage['avg_transitions_per_state']:.2f}")
    print(f"Dead-end states: {coverage['dead_end_states']}")
    print(f"\nExploration Stats:")
    for key, value in coverage['exploration_stats'].items():
        print(f"  {key}: {value}")

    # 結果保存
    agent.save_results("demo_exploration_results.json")

    print("\n✓ Demo completed!")
