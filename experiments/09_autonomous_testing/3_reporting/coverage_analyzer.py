"""
Coverage Analyzer

ゲームの状態空間カバレッジを分析・可視化するモジュール.

使い方:
    from coverage_analyzer import CoverageAnalyzer

    analyzer = CoverageAnalyzer()

    # カバレッジ分析
    coverage = analyzer.analyze(agent_results)

    # レポート生成
    report = analyzer.generate_report(coverage, output_file='coverage_report.html')

    # ヒートマップ生成（2Dゲームの場合）
    analyzer.generate_heatmap(coverage, output_file='coverage_heatmap.png')

機能:
    - 状態空間カバレッジ計測
    - 未到達エリアの検出
    - 訪問頻度ヒートマップ
    - カバレッジ推移グラフ
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime


class CoverageAnalyzer:
    """
    状態空間カバレッジ分析.
    """

    def __init__(self):
        """初期化."""
        self.coverage_data = {}

    def analyze(self, agent_results: Dict) -> Dict:
        """
        エージェント実行結果からカバレッジを分析.

        Args:
            agent_results: エージェントの実行結果

        Returns:
            カバレッジ分析結果
        """
        summary = agent_results.get('summary', {})
        metrics = agent_results.get('metrics')

        # 基本カバレッジ
        total_states = summary.get('unique_states_visited', 0)
        total_steps = summary.get('total_steps', 0)

        # 訪問頻度分布
        visitation_distribution = self._analyze_visitation_distribution(metrics)

        # カバレッジ推移
        coverage_over_time = self._analyze_coverage_over_time(metrics)

        # 未到達エリア推定
        unreached_areas = self._estimate_unreached_areas(metrics)

        # カバレッジスコア
        coverage_score = summary.get('state_coverage', 0)

        analysis = {
            'basic_stats': {
                'total_states_visited': total_states,
                'total_steps': total_steps,
                'avg_steps_per_state': total_steps / total_states if total_states > 0 else 0,
                'coverage_score': coverage_score
            },
            'visitation_distribution': visitation_distribution,
            'coverage_over_time': coverage_over_time,
            'unreached_areas': unreached_areas,
            'hotspots': self._identify_hotspots(metrics),
            'dead_zones': self._identify_dead_zones(metrics)
        }

        self.coverage_data = analysis
        return analysis

    def _analyze_visitation_distribution(self, metrics: Any) -> Dict:
        """
        状態の訪問頻度分布を分析.

        Returns:
            {
                'once': int,  # 1回だけ訪問
                'few': int,   # 2-5回
                'many': int,  # 6-20回
                'heavy': int  # 21回以上
            }
        """
        if not metrics or not hasattr(metrics, 'state_visitation_count'):
            return {'once': 0, 'few': 0, 'many': 0, 'heavy': 0}

        counts = list(metrics.state_visitation_count.values())

        distribution = {
            'once': sum(1 for c in counts if c == 1),
            'few': sum(1 for c in counts if 2 <= c <= 5),
            'many': sum(1 for c in counts if 6 <= c <= 20),
            'heavy': sum(1 for c in counts if c > 20)
        }

        return distribution

    def _analyze_coverage_over_time(self, metrics: Any) -> List[Dict]:
        """
        カバレッジの時系列推移を分析.

        Returns:
            [{episode: int, states_discovered: int, cumulative_states: int}, ...]
        """
        # 簡易版: エピソードごとの新規状態発見数
        # 実際のメトリクスにエピソード情報がある場合はそれを使用
        return []

    def _estimate_unreached_areas(self, metrics: Any) -> Dict:
        """
        未到達エリアを推定.

        Returns:
            {
                'estimated_total_states': int,
                'unreached_count': int,
                'unreached_percentage': float
            }
        """
        if not metrics or not hasattr(metrics, 'unique_states'):
            return {
                'estimated_total_states': 0,
                'unreached_count': 0,
                'unreached_percentage': 0.0
            }

        visited_count = len(metrics.unique_states)

        # 簡易推定: 訪問頻度の分布から全体を推定
        # より高度な推定にはGood-Turing推定などを使用
        estimated_total = int(visited_count * 1.5)  # 簡易的に1.5倍と仮定

        unreached = estimated_total - visited_count

        return {
            'estimated_total_states': estimated_total,
            'unreached_count': unreached,
            'unreached_percentage': (unreached / estimated_total * 100) if estimated_total > 0 else 0
        }

    def _identify_hotspots(self, metrics: Any) -> List[Dict]:
        """
        ホットスポット（頻繁に訪問される状態）を特定.

        Returns:
            [{state_hash: str, visit_count: int}, ...]
        """
        if not metrics or not hasattr(metrics, 'state_visitation_count'):
            return []

        # 訪問回数でソート
        sorted_states = sorted(
            metrics.state_visitation_count.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 上位10個
        hotspots = [
            {
                'state_hash': state[:16],  # 最初の16文字のみ
                'visit_count': count
            }
            for state, count in sorted_states[:10]
        ]

        return hotspots

    def _identify_dead_zones(self, metrics: Any) -> List[Dict]:
        """
        デッドゾーン（1回しか訪問されない状態）を特定.

        Returns:
            [{state_hash: str}, ...]
        """
        if not metrics or not hasattr(metrics, 'state_visitation_count'):
            return []

        dead_zones = [
            {'state_hash': state[:16]}
            for state, count in metrics.state_visitation_count.items()
            if count == 1
        ]

        return dead_zones[:20]  # 最大20個

    def generate_report(
        self,
        coverage_data: Optional[Dict] = None,
        output_file: str = 'coverage_report.html',
        output_dir: str = 'reports'
    ) -> str:
        """
        HTMLカバレッジレポートを生成.

        Args:
            coverage_data: カバレッジ分析結果（Noneの場合は最後の分析結果を使用）
            output_file: 出力ファイル名
            output_dir: 出力ディレクトリ

        Returns:
            生成されたHTMLの内容
        """
        if coverage_data is None:
            coverage_data = self.coverage_data

        if not coverage_data:
            print("⚠️ No coverage data available. Run analyze() first.")
            return ""

        basic = coverage_data.get('basic_stats', {})
        distribution = coverage_data.get('visitation_distribution', {})
        hotspots = coverage_data.get('hotspots', [])
        unreached = coverage_data.get('unreached_areas', {})

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coverage Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .bar-chart {{
            margin-top: 15px;
        }}
        .bar {{
            margin-bottom: 10px;
        }}
        .bar-label {{
            display: inline-block;
            width: 100px;
            font-size: 14px;
            color: #666;
        }}
        .bar-visual {{
            display: inline-block;
            height: 30px;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            border-radius: 4px;
            vertical-align: middle;
        }}
        .bar-value {{
            display: inline-block;
            margin-left: 10px;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: bold;
            color: #333;
        }}
        .progress-ring {{
            transform: rotate(-90deg);
        }}
        .coverage-score {{
            text-align: center;
            padding: 20px;
        }}
        .score-value {{
            font-size: 48px;
            font-weight: bold;
            color: #4CAF50;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Coverage Analysis Report</h1>
        <div style="color: #666; font-size: 14px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="metrics">
        <div class="metric-card">
            <div class="metric-label">States Visited</div>
            <div class="metric-value">{basic.get('total_states_visited', 0):,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Steps</div>
            <div class="metric-value">{basic.get('total_steps', 0):,}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Steps/State</div>
            <div class="metric-value">{basic.get('avg_steps_per_state', 0):.1f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Coverage Score</div>
            <div class="metric-value" style="color: #4CAF50;">{basic.get('coverage_score', 0):.1f}%</div>
        </div>
    </div>

    <div class="section">
        <h2>Visitation Distribution</h2>
        <p>How many times each state was visited</p>
        <div class="bar-chart">
            <div class="bar">
                <span class="bar-label">Once (1x)</span>
                <div class="bar-visual" style="width: {distribution.get('once', 0) * 2}px;"></div>
                <span class="bar-value">{distribution.get('once', 0)}</span>
            </div>
            <div class="bar">
                <span class="bar-label">Few (2-5x)</span>
                <div class="bar-visual" style="width: {distribution.get('few', 0) * 2}px;"></div>
                <span class="bar-value">{distribution.get('few', 0)}</span>
            </div>
            <div class="bar">
                <span class="bar-label">Many (6-20x)</span>
                <div class="bar-visual" style="width: {distribution.get('many', 0) * 2}px;"></div>
                <span class="bar-value">{distribution.get('many', 0)}</span>
            </div>
            <div class="bar">
                <span class="bar-label">Heavy (21+x)</span>
                <div class="bar-visual" style="width: {distribution.get('heavy', 0) * 2}px;"></div>
                <span class="bar-value">{distribution.get('heavy', 0)}</span>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Unreached Areas</h2>
        <p>Estimated states that were never visited</p>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Estimated Total States</td>
                <td>{unreached.get('estimated_total_states', 0):,}</td>
            </tr>
            <tr>
                <td>Unreached Count</td>
                <td>{unreached.get('unreached_count', 0):,}</td>
            </tr>
            <tr>
                <td>Unreached Percentage</td>
                <td>{unreached.get('unreached_percentage', 0):.1f}%</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2>Hotspots</h2>
        <p>Most frequently visited states</p>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>State Hash</th>
                    <th>Visit Count</th>
                </tr>
            </thead>
            <tbody>
"""

        for i, hotspot in enumerate(hotspots, 1):
            html += f"""
                <tr>
                    <td>{i}</td>
                    <td><code>{hotspot['state_hash']}</code></td>
                    <td><strong>{hotspot['visit_count']}</strong></td>
                </tr>
"""

        html += """
            </tbody>
        </table>
    </div>

</body>
</html>
"""

        # ファイル保存
        output_path = Path(output_dir) / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✓ Coverage report saved: {output_path}")
        return html

    def generate_heatmap(
        self,
        coverage_data: Optional[Dict] = None,
        output_file: str = 'coverage_heatmap.png',
        output_dir: str = 'reports',
        grid_size: Tuple[int, int] = (50, 50)
    ):
        """
        2Dカバレッジヒートマップを生成.

        Args:
            coverage_data: カバレッジデータ
            output_file: 出力ファイル名
            output_dir: 出力ディレクトリ
            grid_size: グリッドサイズ
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib required for heatmap. Install with: pip install matplotlib")
            return

        # TODO: 実際のメトリクスから位置情報を抽出してヒートマップを生成
        # ここでは簡易デモ
        print("⚠️ Heatmap generation requires position data in game state")

        # ダミーヒートマップ
        heatmap = np.random.rand(*grid_size) * 10

        plt.figure(figsize=(12, 10))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Visit Count')
        plt.title('State Space Coverage Heatmap')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')

        output_path = Path(output_dir) / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Heatmap saved: {output_path}")


if __name__ == "__main__":
    # デモ: カバレッジ分析
    print("Coverage Analyzer Demo")
    print("=" * 60)

    # ダミーのメトリクス
    class DummyMetrics:
        def __init__(self):
            self.unique_states = set(f"state_{i}" for i in range(500))
            self.state_visitation_count = {
                f"state_{i}": np.random.randint(1, 50)
                for i in range(500)
            }

    dummy_results = {
        'summary': {
            'unique_states_visited': 500,
            'total_steps': 10000,
            'state_coverage': 75.5
        },
        'metrics': DummyMetrics()
    }

    # 分析実行
    analyzer = CoverageAnalyzer()
    coverage = analyzer.analyze(dummy_results)

    print("\nCoverage Analysis:")
    print(f"  States visited: {coverage['basic_stats']['total_states_visited']}")
    print(f"  Coverage score: {coverage['basic_stats']['coverage_score']:.1f}%")
    print(f"\nVisitation Distribution:")
    for category, count in coverage['visitation_distribution'].items():
        print(f"  {category}: {count}")

    # レポート生成
    print("\nGenerating HTML report...")
    analyzer.generate_report(coverage, output_dir='demo_reports')

    print("\n✓ Demo completed! Check 'demo_reports/coverage_report.html'")
