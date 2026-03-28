"""
Bug Hunter Agent

エッジケースを見つけてクラッシュさせる敵対的テストエージェント.
意図的に異常な行動を取り、バグを誘発します.

使い方:
    from bug_hunter_agent import BugHunterAgent

    agent = BugHunterAgent(
        aggression_level=0.8,  # 0.0-1.0: 敵対性の強さ
        boundary_testing=True,  # 境界値テスト有効化
        rapid_input=True        # 高速入力テスト有効化
    )

    results = agent.run(
        get_state_fn=get_game_state,
        take_action_fn=execute_action,
        is_terminal_fn=check_game_over,
        episodes=500
    )

戦略:
    - 入力スパム（同じ行動を連続実行）
    - 極端な値のテスト（最大/最小値）
    - 高速な状態遷移
    - リソース枯渇攻撃
    - 無効な組み合わせの試行
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from pathlib import Path
import json

from base_agent import (
    BaseTestingAgent,
    GameState,
    Bug,
    create_bug
)


class BugHunterAgent(BaseTestingAgent):
    """
    バグハンティング特化型テストエージェント.

    目標:
        - クラッシュを誘発
        - エッジケースを発見
        - 無効な状態遷移を検出
        - リソースリークを発見

    戦略:
        - 敵対的行動（spam, rapid input）
        - 境界値テスト
        - 無効な入力の組み合わせ
        - 状態の異常監視
    """

    def __init__(
        self,
        name: str = "BugHunterAgent",
        aggression_level: float = 0.7,
        boundary_testing: bool = True,
        rapid_input: bool = True,
        spam_probability: float = 0.3,
        invalid_combo_testing: bool = True,
        resource_monitoring: bool = True,
        crash_detection: bool = True,
        verbose: bool = True,
        save_screenshots: bool = True,
        screenshot_dir: str = "screenshots/bug_hunter"
    ):
        """
        Args:
            name: エージェント名
            aggression_level: 敵対性レベル（0.0-1.0）
            boundary_testing: 境界値テスト有効化
            rapid_input: 高速入力テスト
            spam_probability: 入力スパムの確率
            invalid_combo_testing: 無効な組み合わせテスト
            resource_monitoring: リソース監視
            crash_detection: クラッシュ検出
            verbose: 詳細ログ
            save_screenshots: スクリーンショット保存
            screenshot_dir: 保存先
        """
        super().__init__(name, verbose, save_screenshots, screenshot_dir)

        # 戦略設定
        self.aggression_level = aggression_level
        self.boundary_testing = boundary_testing
        self.rapid_input = rapid_input
        self.spam_probability = spam_probability
        self.invalid_combo_testing = invalid_combo_testing
        self.resource_monitoring = resource_monitoring
        self.crash_detection = crash_detection

        # 行動リスト
        self.available_actions = []

        # スパムモード
        self.spam_mode = False
        self.spam_action = None
        self.spam_count = 0
        self.spam_duration = 0

        # 境界値テストキュー
        self.boundary_test_queue = deque()

        # 異常検出履歴
        self.crash_history = []
        self.freeze_detection_window = deque(maxlen=10)
        self.last_state_change_time = time.time()

        # リソース監視
        self.resource_history = defaultdict(list)

        # 統計
        self.bug_hunter_stats = {
            'crashes_triggered': 0,
            'freezes_detected': 0,
            'invalid_states_found': 0,
            'resource_leaks_detected': 0,
            'spam_sequences_tested': 0
        }

    def set_available_actions(self, actions: List[Any]):
        """利用可能な行動リストを設定."""
        self.available_actions = actions
        if self.verbose:
            print(f"[{self.name}] Available actions set: {actions}")

    def select_action(self, state: GameState) -> Any:
        """
        敵対的な行動を選択.

        戦略:
            1. スパムモード（同じ行動を連続実行）
            2. 境界値テスト
            3. 無効な組み合わせ
            4. ランダム攻撃的行動
        """
        if not self.available_actions:
            self.available_actions = list(range(4))

        # スパムモード実行中
        if self.spam_mode:
            return self._continue_spam()

        # 新しいスパムシーケンス開始
        if np.random.random() < self.spam_probability * self.aggression_level:
            return self._start_spam()

        # 境界値テスト
        if self.boundary_testing and self.boundary_test_queue:
            return self.boundary_test_queue.popleft()

        # 無効な組み合わせテスト
        if self.invalid_combo_testing and np.random.random() < 0.2:
            return self._select_invalid_combo(state)

        # 通常の敵対的行動
        return self._adversarial_action(state)

    def _start_spam(self) -> Any:
        """スパムシーケンスを開始."""
        self.spam_mode = True
        self.spam_action = np.random.choice(self.available_actions)
        self.spam_duration = np.random.randint(5, 20)  # 5-20回連続
        self.spam_count = 0
        self.bug_hunter_stats['spam_sequences_tested'] += 1

        if self.verbose:
            print(f"🔥 [{self.name}] Starting spam: action={self.spam_action}, duration={self.spam_duration}")

        return self.spam_action

    def _continue_spam(self) -> Any:
        """スパムシーケンスを継続."""
        self.spam_count += 1

        if self.spam_count >= self.spam_duration:
            self.spam_mode = False
            if self.verbose:
                print(f"✓ [{self.name}] Spam sequence completed")

        return self.spam_action

    def _adversarial_action(self, state: GameState) -> Any:
        """
        敵対的な行動選択.

        直前の行動とは異なる行動を選ぶ（高速切り替え）.
        """
        if self.rapid_input and self.action_history:
            last_action = self.action_history[-1]
            # 前回と違う行動を選択
            candidates = [a for a in self.available_actions if a != last_action]
            if candidates:
                return np.random.choice(candidates)

        return np.random.choice(self.available_actions)

    def _select_invalid_combo(self, state: GameState) -> Any:
        """
        無効な行動の組み合わせをテスト.

        例: ジャンプ中にさらにジャンプ、移動不可能な場所への移動など.
        """
        # ここでは単純にランダムだが、実際のゲームでは
        # 状態に応じて「してはいけない」行動を選ぶ
        return np.random.choice(self.available_actions)

    def detect_bugs(self, state: GameState, next_state: GameState) -> List[Bug]:
        """
        状態遷移からバグを検出.

        検出するバグ:
            - クラッシュ・フリーズ
            - 無限ループ
            - メモリリーク
            - 異常な状態値
            - 入力バリデーション不足
        """
        bugs = []

        # 1. クラッシュ検出
        crash_bug = self._detect_crash(state, next_state)
        if crash_bug:
            bugs.append(crash_bug)
            self.bug_hunter_stats['crashes_triggered'] += 1

        # 2. フリーズ検出
        freeze_bug = self._detect_freeze(state, next_state)
        if freeze_bug:
            bugs.append(freeze_bug)
            self.bug_hunter_stats['freezes_detected'] += 1

        # 3. 無効な状態
        invalid_bug = self._detect_invalid_state(state, next_state)
        if invalid_bug:
            bugs.append(invalid_bug)
            self.bug_hunter_stats['invalid_states_found'] += 1

        # 4. リソースリーク
        if self.resource_monitoring:
            leak_bug = self._detect_resource_leak(state, next_state)
            if leak_bug:
                bugs.append(leak_bug)
                self.bug_hunter_stats['resource_leaks_detected'] += 1

        # 5. 境界外アクセス
        bounds_bug = self._check_boundary_violation(state, next_state)
        if bounds_bug:
            bugs.append(bounds_bug)

        # 6. 数値オーバーフロー
        overflow_bug = self._check_numeric_overflow(state, next_state)
        if overflow_bug:
            bugs.append(overflow_bug)

        # 7. NaN/Inf検出
        nan_bug = self._check_nan_or_inf(state, next_state)
        if nan_bug:
            bugs.append(nan_bug)

        return bugs

    def _detect_crash(self, state: GameState, next_state: GameState) -> Optional[Bug]:
        """
        クラッシュを検出.

        次の状態が空、または重要なフィールドが欠落している場合.
        """
        if not next_state.data:
            return create_bug(
                severity='critical',
                title='Game crashed or returned empty state',
                description='State returned empty after action. Possible crash.',
                state=state,
                steps=self.action_history[-10:]
            )

        # 重要なフィールドの欠落
        required_fields = ['player_position', 'health']
        missing = [f for f in required_fields if f not in next_state.data]

        if missing and all(f in state.data for f in missing):
            return create_bug(
                severity='critical',
                title='Critical state fields missing',
                description=f'Required fields disappeared: {missing}',
                state=state,
                steps=self.action_history[-5:]
            )

        return None

    def _detect_freeze(self, state: GameState, next_state: GameState) -> Optional[Bug]:
        """
        フリーズ（状態が変化しない）を検出.
        """
        self.freeze_detection_window.append(next_state.hash())

        # ウィンドウが満杯で、すべて同じ状態
        if len(self.freeze_detection_window) == 10:
            if len(set(self.freeze_detection_window)) == 1:
                return create_bug(
                    severity='high',
                    title='Game appears frozen',
                    description='State has not changed for 10 consecutive steps',
                    state=state,
                    steps=self.action_history[-10:]
                )

        return None

    def _detect_invalid_state(self, state: GameState, next_state: GameState) -> Optional[Bug]:
        """
        無効な状態値を検出.
        """
        # 健康値が範囲外
        health = next_state.get('health')
        if health is not None:
            if health < 0:
                return create_bug(
                    severity='high',
                    title='Negative health value',
                    description=f'Health: {health} (should be >= 0)',
                    state=state
                )
            elif health > 10000:
                return create_bug(
                    severity='medium',
                    title='Abnormally high health',
                    description=f'Health: {health} (possible overflow)',
                    state=state
                )

        # スコアが負
        score = next_state.get('score')
        if score is not None and score < 0:
            return create_bug(
                severity='medium',
                title='Negative score value',
                description=f'Score: {score}',
                state=state
            )

        return None

    def _detect_resource_leak(self, state: GameState, next_state: GameState) -> Optional[Bug]:
        """
        リソースリーク（メモリ、敵の数など）を検出.
        """
        # 敵の数が異常に増加
        enemies = next_state.get('enemies')
        if enemies is not None:
            self.resource_history['enemies'].append(enemies)

            if len(self.resource_history['enemies']) > 100:
                recent = self.resource_history['enemies'][-100:]
                if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
                    # 100ステップ連続で敵が増え続けている
                    return create_bug(
                        severity='high',
                        title='Possible enemy spawn leak',
                        description=f'Enemy count continuously increasing: {recent[0]} -> {recent[-1]}',
                        state=state
                    )

        return None

    def _check_boundary_violation(self, state: GameState, next_state: GameState) -> Optional[Bug]:
        """境界違反を検出."""
        pos = next_state.get('player_position')
        if pos is None:
            return None

        # 極端な座標
        for i, coord in enumerate(pos):
            if abs(coord) > 100000:
                return create_bug(
                    severity='critical',
                    title='Coordinate overflow detected',
                    description=f'Position[{i}] = {coord} (overflow or teleport bug)',
                    state=state,
                    steps=self.action_history[-5:]
                )

        return None

    def _check_numeric_overflow(self, state: GameState, next_state: GameState) -> Optional[Bug]:
        """数値オーバーフローを検出."""
        # スコアが突然負になった（オーバーフロー）
        prev_score = state.get('score', 0)
        next_score = next_state.get('score', 0)

        if prev_score > 0 and next_score < 0 and abs(next_score - prev_score) > 1000000:
            return create_bug(
                severity='high',
                title='Score overflow detected',
                description=f'Score: {prev_score} -> {next_score} (likely integer overflow)',
                state=state
            )

        return None

    def _check_nan_or_inf(self, state: GameState, next_state: GameState) -> Optional[Bug]:
        """NaN/Infiniteを検出."""
        def check_value(val, name):
            if isinstance(val, (int, float)):
                if np.isnan(val):
                    return f'{name} is NaN'
                elif np.isinf(val):
                    return f'{name} is Infinite'
            elif isinstance(val, (list, tuple)):
                for i, v in enumerate(val):
                    if isinstance(v, (int, float)):
                        if np.isnan(v):
                            return f'{name}[{i}] is NaN'
                        elif np.isinf(v):
                            return f'{name}[{i}] is Infinite'
            return None

        # すべてのフィールドをチェック
        for key, value in next_state.data.items():
            error = check_value(value, key)
            if error:
                return create_bug(
                    severity='critical',
                    title='NaN or Infinite value detected',
                    description=error,
                    state=state,
                    steps=self.action_history[-3:]
                )

        return None

    def _calculate_reward(self, state: GameState, next_state: GameState, action: Any) -> float:
        """
        報酬計算（バグ発見にボーナス）.
        """
        bugs = self.detect_bugs(state, next_state)

        # バグを見つけたら高報酬
        if bugs:
            severity_rewards = {
                'critical': 100.0,
                'high': 50.0,
                'medium': 20.0,
                'low': 5.0
            }
            return sum(severity_rewards.get(bug.severity, 0) for bug in bugs)

        # 新しい状態へのボーナス
        if next_state.hash() not in self.metrics.unique_states:
            return 1.0

        return 0.0

    def get_bug_hunter_report(self) -> Dict:
        """バグハンターレポートを生成."""
        return {
            'total_bugs_found': len(self.metrics.bugs_found),
            'bug_breakdown': self._get_bug_breakdown(),
            'bug_hunter_stats': self.bug_hunter_stats,
            'most_severe_bugs': self._get_top_bugs(n=10)
        }

    def _get_bug_breakdown(self) -> Dict:
        """バグを重要度別に集計."""
        breakdown = defaultdict(int)
        for bug in self.metrics.bugs_found:
            breakdown[bug.severity] += 1
        return dict(breakdown)

    def _get_top_bugs(self, n: int = 10) -> List[Dict]:
        """重要度の高いバグを取得."""
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}

        sorted_bugs = sorted(
            self.metrics.bugs_found,
            key=lambda b: severity_order.get(b.severity, 999)
        )

        return [bug.to_dict() for bug in sorted_bugs[:n]]

    def save_results(self, filepath: str):
        """テスト結果を保存（バグハンターレポート含む）."""
        results = {
            'agent_name': self.name,
            'aggression_level': self.aggression_level,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': self.metrics.get_summary(),
            'bug_hunter_report': self.get_bug_hunter_report(),
            'bugs': [bug.to_dict() for bug in self.metrics.bugs_found]
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"[{self.name}] Results saved to {filepath}")


# ヘルパー関数

def run_bug_hunter_test(
    get_state_fn: Callable,
    take_action_fn: Callable,
    is_terminal_fn: Callable,
    available_actions: List[Any],
    episodes: int = 500,
    max_steps: int = 1000,
    aggression: float = 0.8,
    output_dir: str = 'results/bug_hunter'
) -> Dict:
    """
    バグハンターテストを実行.

    Args:
        get_state_fn: 状態取得関数
        take_action_fn: 行動実行関数
        is_terminal_fn: 終了判定関数
        available_actions: 利用可能な行動リスト
        episodes: エピソード数
        max_steps: 最大ステップ数
        aggression: 敵対性レベル
        output_dir: 結果出力ディレクトリ

    Returns:
        テスト結果
    """
    agent = BugHunterAgent(
        aggression_level=aggression,
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
    agent.save_results(f"{output_dir}/bug_hunter_results.json")

    return results


if __name__ == "__main__":
    # デモ: バグを仕込んだゲーム環境でテスト
    print("Bug Hunter Agent Demo")
    print("=" * 60)

    # バグのあるダミーゲーム
    class BuggyGame:
        def __init__(self):
            self.position = [0, 0]
            self.health = 100
            self.score = 0
            self.enemies = 0
            self.step_count = 0

        def get_state(self):
            return {
                'player_position': self.position.copy(),
                'health': self.health,
                'score': self.score,
                'enemies': self.enemies
            }

        def take_action(self, action):
            self.step_count += 1

            # BUG 1: 高速入力でクラッシュ（スパム検出用）
            if hasattr(self, '_last_action') and self._last_action == action:
                self._spam_count = getattr(self, '_spam_count', 0) + 1
                if self._spam_count > 15:
                    # クラッシュをシミュレート（健康値を負に）
                    self.health = -999
            else:
                self._spam_count = 0
            self._last_action = action

            # BUG 2: スコアオーバーフロー
            self.score += 1000
            if self.score > 2000000000:  # INT_MAX付近
                self.score = -2147483648  # オーバーフロー

            # BUG 3: 敵スポーンリーク
            if self.step_count % 5 == 0:
                self.enemies += 1  # 減らない

            # 通常の移動
            moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            if action < len(moves):
                move = moves[action]
                self.position[0] += move[0]
                self.position[1] += move[1]

                # BUG 4: 境界チェック不足
                # 特定位置で座標が爆発
                if self.position == [5, 5]:
                    self.position[0] = 999999

        def is_terminal(self, state):
            return state['health'] <= 0 or self.step_count > 100

        def reset(self):
            self.position = [0, 0]
            self.health = 100
            self.score = 0
            self.enemies = 0
            self.step_count = 0

    # 環境作成
    env = BuggyGame()

    # エージェント作成
    agent = BugHunterAgent(
        aggression_level=0.9,
        spam_probability=0.5,
        verbose=True
    )

    agent.set_available_actions([0, 1, 2, 3])

    # テスト実行
    results = agent.run(
        get_state_fn=env.get_state,
        take_action_fn=env.take_action,
        is_terminal_fn=env.is_terminal,
        episodes=10,
        max_steps=50
    )

    # 結果表示
    print("\n" + "=" * 60)
    print("Bug Hunter Report:")
    print("=" * 60)
    report = agent.get_bug_hunter_report()
    print(f"Total bugs found: {report['total_bugs_found']}")
    print(f"\nBug Breakdown:")
    for severity, count in report['bug_breakdown'].items():
        print(f"  {severity.capitalize()}: {count}")

    print(f"\nBug Hunter Stats:")
    for key, value in report['bug_hunter_stats'].items():
        print(f"  {key}: {value}")

    if report['most_severe_bugs']:
        print(f"\nTop Bugs:")
        for i, bug in enumerate(report['most_severe_bugs'][:3], 1):
            print(f"  {i}. [{bug['severity'].upper()}] {bug['title']}")

    # 結果保存
    agent.save_results("demo_bug_hunter_results.json")

    print("\n✓ Demo completed!")
