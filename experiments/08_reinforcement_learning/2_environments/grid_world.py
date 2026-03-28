"""
Grid World Environment

シンプルなグリッドベースの強化学習環境.
OpenAI Gym互換インターフェース.

特徴:
- カスタマイズ可能なサイズ、障害物、ゴール
- 複数の報酬設計パターン
- 可視化機能
- レベルエディタ

使い方:
    # 基本的な使い方
    python grid_world.py --size 10 --obstacles 5 --episodes 100

    # カスタムレベルから読み込み
    python grid_world.py --load levels/maze.json --episodes 100

    # レベルエディタ
    python grid_world.py --edit --size 10 --save levels/custom.json
"""

import argparse
import json
import random
from pathlib import Path
from typing import Tuple, Set, Optional, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GridWorld:
    """
    グリッドワールド環境（OpenAI Gym互換）.

    エージェントが開始地点からゴールを目指す.
    - 障害物: 通過不可
    - 落とし穴: 踏むとペナルティ & ゲームオーバー
    - 宝箱: 踏むとボーナス
    """

    def __init__(
        self,
        size: int = 5,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
        obstacles: Optional[Set[Tuple[int, int]]] = None,
        pits: Optional[Set[Tuple[int, int]]] = None,
        treasures: Optional[Set[Tuple[int, int]]] = None,
        reward_goal: float = 100.0,
        reward_step: float = -1.0,
        reward_pit: float = -50.0,
        reward_treasure: float = 10.0,
        max_steps: int = 100
    ):
        """
        Args:
            size: グリッドサイズ（size x size）
            start: 開始位置（デフォルト: (0, 0)）
            goal: ゴール位置（デフォルト: (size-1, size-1)）
            obstacles: 障害物の座標セット
            pits: 落とし穴の座標セット
            treasures: 宝箱の座標セット
            reward_goal: ゴール到達報酬
            reward_step: ステップペナルティ
            reward_pit: 落とし穴ペナルティ
            reward_treasure: 宝箱報酬
            max_steps: 最大ステップ数（超えたら終了）
        """
        self.size = size
        self.start = start or (0, 0)
        self.goal = goal or (size - 1, size - 1)
        self.obstacles = obstacles or set()
        self.pits = pits or set()
        self.treasures = treasures or set()

        self.reward_goal = reward_goal
        self.reward_step = reward_step
        self.reward_pit = reward_pit
        self.reward_treasure = reward_treasure
        self.max_steps = max_steps

        # 現在の状態
        self.agent_pos = self.start
        self.collected_treasures = set()
        self.steps = 0

        # Gym互換属性
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

    def _get_observation_space(self) -> Dict:
        """観測空間（状態の範囲）."""
        return {
            'shape': (2,),  # (row, col)
            'low': np.array([0, 0]),
            'high': np.array([self.size - 1, self.size - 1])
        }

    def _get_action_space(self) -> Dict:
        """行動空間."""
        return {
            'n': 4,  # 0=上, 1=右, 2=下, 3=左
            'names': ['UP', 'RIGHT', 'DOWN', 'LEFT']
        }

    def reset(self) -> Tuple[int, int]:
        """環境をリセット."""
        self.agent_pos = self.start
        self.collected_treasures = set()
        self.steps = 0
        return self.agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, bool, Dict]:
        """
        行動を実行.

        Args:
            action: 0=上, 1=右, 2=下, 3=左

        Returns:
            (next_state, reward, terminated, truncated, info):
                - next_state: 次の状態
                - reward: 報酬
                - terminated: ゴール到達/落とし穴でエピソード終了
                - truncated: 最大ステップ超過で打ち切り
                - info: 追加情報
        """
        self.steps += 1

        # 移動方向
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上, 右, 下, 左
        move = moves[action]

        # 新しい位置
        new_pos = (
            self.agent_pos[0] + move[0],
            self.agent_pos[1] + move[1]
        )

        # 境界チェック & 障害物チェック
        if (0 <= new_pos[0] < self.size and
            0 <= new_pos[1] < self.size and
            new_pos not in self.obstacles):
            self.agent_pos = new_pos

        # 報酬計算
        reward = self.reward_step  # 基本的なステップペナルティ
        terminated = False
        info = {'event': 'step'}

        # ゴール到達
        if self.agent_pos == self.goal:
            reward = self.reward_goal
            terminated = True
            info['event'] = 'goal'

        # 落とし穴
        elif self.agent_pos in self.pits:
            reward = self.reward_pit
            terminated = True
            info['event'] = 'pit'

        # 宝箱
        elif self.agent_pos in self.treasures and self.agent_pos not in self.collected_treasures:
            reward += self.reward_treasure
            self.collected_treasures.add(self.agent_pos)
            info['event'] = 'treasure'

        # 最大ステップ超過
        truncated = self.steps >= self.max_steps

        return self.agent_pos, reward, terminated, truncated, info

    def render(self, mode: str = 'console'):
        """
        環境を可視化.

        Args:
            mode: 'console' or 'matplotlib'
        """
        if mode == 'console':
            self._render_console()
        elif mode == 'matplotlib':
            self._render_matplotlib()

    def _render_console(self):
        """コンソールに表示."""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]

        # 障害物
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'

        # 落とし穴
        for pit in self.pits:
            grid[pit[0]][pit[1]] = 'P'

        # 宝箱
        for treasure in self.treasures:
            if treasure not in self.collected_treasures:
                grid[treasure[0]][treasure[1]] = 'T'

        # スタートとゴール
        grid[self.start[0]][self.start[1]] = 'S'
        grid[self.goal[0]][self.goal[1]] = 'G'

        # エージェント
        if self.agent_pos != self.start and self.agent_pos != self.goal:
            grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'

        # 表示
        print(f"\nStep: {self.steps}")
        for row in grid:
            print(' '.join(row))
        print()

    def _render_matplotlib(self):
        """Matplotlibで可視化."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # グリッド描画
        for i in range(self.size + 1):
            ax.plot([0, self.size], [i, i], 'k-', linewidth=0.5)
            ax.plot([i, i], [0, self.size], 'k-', linewidth=0.5)

        # 色設定
        colors = {
            'obstacle': 'black',
            'pit': 'red',
            'treasure': 'gold',
            'start': 'lightblue',
            'goal': 'lightgreen',
            'agent': 'blue'
        }

        # 障害物
        for obs in self.obstacles:
            rect = patches.Rectangle((obs[1], self.size - obs[0] - 1), 1, 1,
                                     facecolor=colors['obstacle'])
            ax.add_patch(rect)

        # 落とし穴
        for pit in self.pits:
            rect = patches.Rectangle((pit[1], self.size - pit[0] - 1), 1, 1,
                                     facecolor=colors['pit'], alpha=0.7)
            ax.add_patch(rect)

        # 宝箱
        for treasure in self.treasures:
            if treasure not in self.collected_treasures:
                rect = patches.Rectangle((treasure[1], self.size - treasure[0] - 1), 1, 1,
                                         facecolor=colors['treasure'], alpha=0.7)
                ax.add_patch(rect)

        # スタート
        rect = patches.Rectangle((self.start[1], self.size - self.start[0] - 1), 1, 1,
                                 facecolor=colors['start'], alpha=0.5)
        ax.add_patch(rect)

        # ゴール
        rect = patches.Rectangle((self.goal[1], self.size - self.goal[0] - 1), 1, 1,
                                 facecolor=colors['goal'], alpha=0.5)
        ax.add_patch(rect)

        # エージェント
        circle = patches.Circle((self.agent_pos[1] + 0.5, self.size - self.agent_pos[0] - 0.5),
                               0.3, facecolor=colors['agent'])
        ax.add_patch(circle)

        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.set_title(f'Grid World (Step: {self.steps})')

        plt.show()

    def save(self, filepath: str):
        """レベルをJSONに保存."""
        level_data = {
            'size': self.size,
            'start': list(self.start),
            'goal': list(self.goal),
            'obstacles': [list(obs) for obs in self.obstacles],
            'pits': [list(pit) for pit in self.pits],
            'treasures': [list(t) for t in self.treasures],
            'rewards': {
                'goal': self.reward_goal,
                'step': self.reward_step,
                'pit': self.reward_pit,
                'treasure': self.reward_treasure
            },
            'max_steps': self.max_steps
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(level_data, f, indent=2)

        print(f"Level saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'GridWorld':
        """JSONからレベルを読み込み."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        obstacles = {tuple(obs) for obs in data.get('obstacles', [])}
        pits = {tuple(pit) for pit in data.get('pits', [])}
        treasures = {tuple(t) for t in data.get('treasures', [])}

        rewards = data.get('rewards', {})

        env = cls(
            size=data['size'],
            start=tuple(data['start']),
            goal=tuple(data['goal']),
            obstacles=obstacles,
            pits=pits,
            treasures=treasures,
            reward_goal=rewards.get('goal', 100.0),
            reward_step=rewards.get('step', -1.0),
            reward_pit=rewards.get('pit', -50.0),
            reward_treasure=rewards.get('treasure', 10.0),
            max_steps=data.get('max_steps', 100)
        )

        print(f"Level loaded from {filepath}")
        return env

    @classmethod
    def generate_random(
        cls,
        size: int = 10,
        obstacle_ratio: float = 0.2,
        pit_ratio: float = 0.05,
        treasure_ratio: float = 0.1
    ) -> 'GridWorld':
        """
        ランダムレベル生成.

        Args:
            size: グリッドサイズ
            obstacle_ratio: 障害物の割合
            pit_ratio: 落とし穴の割合
            treasure_ratio: 宝箱の割合

        Returns:
            GridWorld環境
        """
        start = (0, 0)
        goal = (size - 1, size - 1)

        # 利用可能な座標（スタート・ゴール除外）
        available_cells = [
            (i, j) for i in range(size) for j in range(size)
            if (i, j) != start and (i, j) != goal
        ]

        # ランダムに配置
        num_obstacles = int(len(available_cells) * obstacle_ratio)
        num_pits = int(len(available_cells) * pit_ratio)
        num_treasures = int(len(available_cells) * treasure_ratio)

        random.shuffle(available_cells)

        obstacles = set(available_cells[:num_obstacles])
        pits = set(available_cells[num_obstacles:num_obstacles + num_pits])
        treasures = set(available_cells[num_obstacles + num_pits:num_obstacles + num_pits + num_treasures])

        return cls(
            size=size,
            start=start,
            goal=goal,
            obstacles=obstacles,
            pits=pits,
            treasures=treasures
        )


def demo_gridworld():
    """グリッドワールドのデモ."""
    print("=" * 60)
    print("GRID WORLD DEMO")
    print("=" * 60)

    # ランダムレベル生成
    env = GridWorld.generate_random(size=10, obstacle_ratio=0.15, pit_ratio=0.05, treasure_ratio=0.1)

    print("\n1. Random level generated")
    env.render(mode='console')

    # ランダムプレイ
    print("\n2. Random agent playing...")
    state = env.reset()
    total_reward = 0
    done = False

    for step in range(50):
        action = random.randint(0, 3)  # ランダム行動
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward

        if step % 10 == 0:
            print(f"Step {step}: Action={env.action_space['names'][action]}, "
                  f"Reward={reward:.1f}, Event={info['event']}")

        if done:
            print(f"\nEpisode finished! Total reward: {total_reward:.1f}")
            break

    # 最終状態表示
    env.render(mode='console')

    # レベル保存
    save_path = "levels/demo_level.json"
    env.save(save_path)

    # レベル読み込み
    print("\n3. Loading saved level...")
    loaded_env = GridWorld.load(save_path)
    loaded_env.render(mode='console')

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


def interactive_editor():
    """インタラクティブなレベルエディタ."""
    print("=" * 60)
    print("GRID WORLD LEVEL EDITOR")
    print("=" * 60)

    size = int(input("Grid size (e.g., 10): ") or "10")

    env = GridWorld(size=size)

    print("\nCommands:")
    print("  add obstacle <row> <col>")
    print("  add pit <row> <col>")
    print("  add treasure <row> <col>")
    print("  set start <row> <col>")
    print("  set goal <row> <col>")
    print("  show")
    print("  save <filepath>")
    print("  quit")

    while True:
        env.render(mode='console')

        cmd = input("\nCommand: ").strip().lower().split()

        if not cmd:
            continue

        if cmd[0] == 'quit':
            break

        elif cmd[0] == 'add' and len(cmd) == 4:
            obj_type, row, col = cmd[1], int(cmd[2]), int(cmd[3])
            pos = (row, col)

            if obj_type == 'obstacle':
                env.obstacles.add(pos)
            elif obj_type == 'pit':
                env.pits.add(pos)
            elif obj_type == 'treasure':
                env.treasures.add(pos)

        elif cmd[0] == 'set' and len(cmd) == 4:
            target, row, col = cmd[1], int(cmd[2]), int(cmd[3])
            pos = (row, col)

            if target == 'start':
                env.start = pos
                env.agent_pos = pos
            elif target == 'goal':
                env.goal = pos

        elif cmd[0] == 'save' and len(cmd) == 2:
            env.save(cmd[1])

        elif cmd[0] == 'show':
            env.render(mode='console')

    print("\nEditor closed.")


def main():
    parser = argparse.ArgumentParser(description="Grid World Environment")
    parser.add_argument("--size", type=int, default=5, help="Grid size")
    parser.add_argument("--obstacles", type=int, help="Number of random obstacles")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--load", type=str, help="Load level from JSON")
    parser.add_argument("--save", type=str, help="Save level to JSON")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--edit", action="store_true", help="Open level editor")
    parser.add_argument("--render", action="store_true", help="Render environment")

    args = parser.parse_args()

    # デモモード
    if args.demo:
        demo_gridworld()
        return

    # エディタモード
    if args.edit:
        interactive_editor()
        return

    # 環境作成
    if args.load:
        env = GridWorld.load(args.load)
    else:
        if args.obstacles:
            env = GridWorld.generate_random(
                size=args.size,
                obstacle_ratio=args.obstacles / (args.size * args.size)
            )
        else:
            env = GridWorld(size=args.size)

    # 保存
    if args.save:
        env.save(args.save)

    # ランダムプレイ
    print(f"Running {args.episodes} episodes with random agent...")

    episode_rewards = []

    for episode in range(args.episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if args.render:
                env.render(mode='console')

            action = random.randint(0, 3)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward

        episode_rewards.append(total_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {total_reward:.1f}")

    print(f"\nAverage reward: {np.mean(episode_rewards):.2f}")


if __name__ == "__main__":
    main()
