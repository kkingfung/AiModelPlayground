"""
Base Testing Agent

基底クラスとなるテストエージェント.
すべてのテストエージェントはこのクラスを継承.

使い方:
    from base_agent import BaseTestingAgent

    class MyAgent(BaseTestingAgent):
        def select_action(self, state):
            # カスタムロジック
            return action

        def detect_bugs(self, state, next_state):
            # バグ検出ロジック
            return bugs
"""

import time
import json
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict
import hashlib


class GameState:
    """
    ゲーム状態のラッパークラス.

    統一されたインターフェースでゲーム状態を管理.
    """

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.timestamp = time.time()

    def __getitem__(self, key):
        return self.data.get(key)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def to_dict(self) -> Dict:
        return self.data

    def hash(self) -> str:
        """状態のハッシュ値を計算（訪問済み判定用）."""
        # 重要な状態要素のみをハッシュ化
        key_elements = []

        # プレイヤー位置（丸め処理）
        if 'player_position' in self.data:
            pos = self.data['player_position']
            rounded_pos = tuple(round(p, 1) for p in pos)
            key_elements.append(str(rounded_pos))

        # レベル名
        if 'level' in self.data:
            key_elements.append(self.data['level'])

        # その他の重要な状態
        for key in ['health', 'score', 'stage']:
            if key in self.data:
                key_elements.append(f"{key}:{self.data[key]}")

        state_str = '|'.join(key_elements)
        return hashlib.md5(state_str.encode()).hexdigest()


class Bug:
    """
    バグ情報を保持するクラス.
    """

    def __init__(
        self,
        bug_id: str,
        severity: str,
        title: str,
        description: str,
        state: GameState,
        steps_to_reproduce: Optional[List[str]] = None
    ):
        self.bug_id = bug_id
        self.severity = severity  # critical, high, medium, low
        self.title = title
        self.description = description
        self.state = state
        self.steps_to_reproduce = steps_to_reproduce or []
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """JSON出力用の辞書に変換."""
        return {
            'bug_id': self.bug_id,
            'severity': self.severity,
            'title': self.title,
            'description': self.description,
            'steps_to_reproduce': self.steps_to_reproduce,
            'game_state': self.state.to_dict(),
            'timestamp': self.timestamp
        }


class TestingMetrics:
    """
    テスト実行のメトリクスを記録.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """メトリクスをリセット."""
        self.episodes_completed = 0
        self.total_steps = 0
        self.total_time = 0.0
        self.bugs_found = []
        self.crashes = 0
        self.unique_states = set()
        self.state_visitation_count = defaultdict(int)
        self.action_counts = defaultdict(int)
        self.rewards = []
        self.episode_lengths = []

    def record_episode(self, steps: int, reward: float, duration: float):
        """エピソード完了時に記録."""
        self.episodes_completed += 1
        self.total_steps += steps
        self.total_time += duration
        self.rewards.append(reward)
        self.episode_lengths.append(steps)

    def record_state(self, state: GameState):
        """状態訪問を記録."""
        state_hash = state.hash()
        self.unique_states.add(state_hash)
        self.state_visitation_count[state_hash] += 1

    def record_action(self, action: Any):
        """行動を記録."""
        self.action_counts[str(action)] += 1

    def record_bug(self, bug: Bug):
        """バグを記録."""
        self.bugs_found.append(bug)

    def get_summary(self) -> Dict:
        """サマリーを取得."""
        return {
            'episodes_completed': self.episodes_completed,
            'total_steps': self.total_steps,
            'total_time': self.total_time,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'avg_reward': np.mean(self.rewards) if self.rewards else 0,
            'unique_states_visited': len(self.unique_states),
            'bugs_found': len(self.bugs_found),
            'crashes': self.crashes,
            'state_coverage': self._calculate_coverage(),
        }

    def _calculate_coverage(self) -> float:
        """状態空間のカバレッジを推定."""
        # 簡易的なカバレッジ計算（訪問回数の分布から推定）
        if not self.state_visitation_count:
            return 0.0

        # よく訪れる状態が多い = カバレッジが低い
        # 訪問が均等に分散 = カバレッジが高い
        visit_counts = list(self.state_visitation_count.values())
        mean_visits = np.mean(visit_counts)
        std_visits = np.std(visit_counts)

        # 正規化されたカバレッジスコア
        if mean_visits == 0:
            return 0.0

        coverage = 1.0 - min(1.0, std_visits / mean_visits)
        return coverage * 100


class BaseTestingAgent(ABC):
    """
    テストエージェントの基底クラス.

    すべてのテストエージェントはこのクラスを継承し、
    以下のメソッドを実装する必要がある:
    - select_action(state): 状態から行動を選択
    - detect_bugs(state, next_state): バグを検出
    """

    def __init__(
        self,
        name: str = "TestAgent",
        verbose: bool = True,
        save_screenshots: bool = True,
        screenshot_dir: str = "screenshots"
    ):
        """
        Args:
            name: エージェント名
            verbose: 詳細ログを表示
            save_screenshots: スクリーンショットを保存
            screenshot_dir: スクリーンショット保存先
        """
        self.name = name
        self.verbose = verbose
        self.save_screenshots = save_screenshots
        self.screenshot_dir = Path(screenshot_dir)

        if self.save_screenshots:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        # メトリクス
        self.metrics = TestingMetrics()

        # 実行履歴
        self.action_history = []
        self.state_history = []

        # 現在のエピソード情報
        self.current_episode = 0
        self.current_step = 0

    @abstractmethod
    def select_action(self, state: GameState) -> Any:
        """
        状態から行動を選択.

        Args:
            state: 現在のゲーム状態

        Returns:
            選択された行動
        """
        pass

    @abstractmethod
    def detect_bugs(self, state: GameState, next_state: GameState) -> List[Bug]:
        """
        状態遷移からバグを検出.

        Args:
            state: 現在の状態
            next_state: 次の状態

        Returns:
            検出されたバグのリスト
        """
        pass

    def reset(self):
        """エピソード開始時の初期化."""
        self.action_history = []
        self.state_history = []
        self.current_step = 0

    def run_episode(
        self,
        get_state_fn,
        take_action_fn,
        is_terminal_fn,
        max_steps: int = 1000
    ) -> Dict:
        """
        1エピソードを実行.

        Args:
            get_state_fn: 状態を取得する関数
            take_action_fn: 行動を実行する関数
            is_terminal_fn: 終了判定関数
            max_steps: 最大ステップ数

        Returns:
            エピソード結果の辞書
        """
        self.reset()
        episode_start = time.time()

        state = GameState(get_state_fn())
        total_reward = 0
        bugs_in_episode = []

        while self.current_step < max_steps:
            # 状態記録
            self.state_history.append(state)
            self.metrics.record_state(state)

            # 行動選択
            action = self.select_action(state)
            self.action_history.append(action)
            self.metrics.record_action(action)

            if self.verbose and self.current_step % 100 == 0:
                print(f"[{self.name}] Episode {self.current_episode}, "
                      f"Step {self.current_step}, Action: {action}")

            # 行動実行
            take_action_fn(action)
            time.sleep(0.01)  # ゲーム処理を待つ

            # 次の状態取得
            next_state = GameState(get_state_fn())

            # バグ検出
            bugs = self.detect_bugs(state, next_state)
            if bugs:
                for bug in bugs:
                    self.metrics.record_bug(bug)
                    bugs_in_episode.append(bug)

                    if self.verbose:
                        print(f"🐛 [{bug.severity.upper()}] {bug.title}")

            # 報酬計算（オプション）
            reward = self._calculate_reward(state, next_state, action)
            total_reward += reward

            # 終了判定
            if is_terminal_fn(next_state.to_dict()):
                if self.verbose:
                    print(f"✓ Episode {self.current_episode} completed in {self.current_step} steps")
                break

            state = next_state
            self.current_step += 1

        # エピソード終了
        duration = time.time() - episode_start
        self.metrics.record_episode(self.current_step, total_reward, duration)
        self.current_episode += 1

        return {
            'episode': self.current_episode - 1,
            'steps': self.current_step,
            'reward': total_reward,
            'duration': duration,
            'bugs_found': len(bugs_in_episode),
            'bugs': bugs_in_episode
        }

    def run(
        self,
        get_state_fn,
        take_action_fn,
        is_terminal_fn,
        episodes: int = 100,
        max_steps: int = 1000,
        save_every: int = 10
    ) -> Dict:
        """
        複数エピソードを実行.

        Args:
            get_state_fn: 状態取得関数
            take_action_fn: 行動実行関数
            is_terminal_fn: 終了判定関数
            episodes: エピソード数
            max_steps: 最大ステップ数
            save_every: 保存間隔

        Returns:
            テスト結果の辞書
        """
        print(f"=" * 60)
        print(f"Starting {self.name} Testing")
        print(f"Episodes: {episodes}, Max steps: {max_steps}")
        print(f"=" * 60)

        for ep in range(episodes):
            episode_result = self.run_episode(
                get_state_fn,
                take_action_fn,
                is_terminal_fn,
                max_steps
            )

            # 定期的に保存
            if (ep + 1) % save_every == 0:
                self.save_results(f"results_ep{ep+1}.json")

        # 最終結果
        print(f"\n" + "=" * 60)
        print(f"Testing Completed!")
        print(f"=" * 60)

        summary = self.metrics.get_summary()
        self._print_summary(summary)

        return {
            'summary': summary,
            'bugs': [bug.to_dict() for bug in self.metrics.bugs_found],
            'metrics': self.metrics
        }

    def _calculate_reward(
        self,
        state: GameState,
        next_state: GameState,
        action: Any
    ) -> float:
        """
        報酬を計算（オプション、サブクラスでオーバーライド可能）.

        デフォルトでは新規状態の探索に報酬を与える.
        """
        # 新しい状態を訪れた場合にボーナス
        if next_state.hash() not in self.metrics.unique_states:
            return 1.0
        return 0.0

    def _print_summary(self, summary: Dict):
        """サマリーを表示."""
        print(f"\nTest Summary:")
        print(f"  Episodes: {summary['episodes_completed']}")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Total time: {summary['total_time']:.1f}s")
        print(f"  Avg episode length: {summary['avg_episode_length']:.1f}")
        print(f"  Unique states: {summary['unique_states_visited']}")
        print(f"  State coverage: {summary['state_coverage']:.1f}%")
        print(f"  Bugs found: {summary['bugs_found']}")

        if summary['bugs_found'] > 0:
            print(f"\n🐛 Bug Breakdown:")
            severity_counts = defaultdict(int)
            for bug in self.metrics.bugs_found:
                severity_counts[bug.severity] += 1

            for severity in ['critical', 'high', 'medium', 'low']:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    print(f"  {severity.capitalize()}: {count}")

    def save_results(self, filepath: str):
        """テスト結果を保存."""
        results = {
            'agent_name': self.name,
            'timestamp': datetime.now().isoformat(),
            'summary': self.metrics.get_summary(),
            'bugs': [bug.to_dict() for bug in self.metrics.bugs_found]
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"Results saved to {filepath}")

    def load_results(self, filepath: str) -> Dict:
        """保存された結果を読み込み."""
        with open(filepath, 'r') as f:
            return json.load(f)


# ヘルパー関数

def create_bug(
    severity: str,
    title: str,
    description: str,
    state: GameState,
    steps: Optional[List[str]] = None
) -> Bug:
    """バグオブジェクトを作成するヘルパー."""
    bug_id = f"BUG-{int(time.time() * 1000)}"
    return Bug(bug_id, severity, title, description, state, steps)


if __name__ == "__main__":
    # デモ: 簡易テスト
    print("Base Testing Agent Demo")
    print("=" * 60)

    # ダミーの実装
    class DemoAgent(BaseTestingAgent):
        def select_action(self, state):
            return np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])

        def detect_bugs(self, state, next_state):
            # ダミー: 低確率でバグを"検出"
            if np.random.random() < 0.05:
                return [create_bug(
                    severity='medium',
                    title='Random bug found',
                    description='This is a demo bug',
                    state=state
                )]
            return []

    agent = DemoAgent(name="DemoAgent")

    # ダミーのゲーム関数
    dummy_state = {'position': [0, 0], 'health': 100}

    def get_state():
        dummy_state['position'][0] += np.random.randint(-1, 2)
        dummy_state['position'][1] += np.random.randint(-1, 2)
        return dummy_state.copy()

    def take_action(action):
        pass  # ダミー

    def is_terminal(state):
        return state['position'][0] > 10 or state['position'][1] > 10

    # テスト実行
    results = agent.run(
        get_state_fn=get_state,
        take_action_fn=take_action,
        is_terminal_fn=is_terminal,
        episodes=10,
        max_steps=50
    )

    # 結果保存
    agent.save_results("demo_results.json")

    print("\n✓ Demo completed!")
