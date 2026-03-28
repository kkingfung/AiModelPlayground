"""
Bug Reporter

テスト結果からバグレポートを生成し、外部システムに統合するモジュール.

使い方:
    from bug_reporter import BugReporter

    reporter = BugReporter(
        output_format='markdown',  # 'json', 'markdown', 'html', 'jira'
        jira_url='https://jira.company.com',
        jira_project='GAME'
    )

    # バグレポート生成
    report = reporter.generate_report(agent_results)

    # JIRA課題作成
    reporter.create_jira_issues(agent_results)

    # メール送信
    reporter.send_email_report(agent_results, recipients=['qa@company.com'])

機能:
    - Markdown/HTML/JSONレポート生成
    - JIRA統合（課題自動作成）
    - Slack通知
    - メール送信
    - スクリーンショット添付
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class BugReporter:
    """
    バグレポート生成・外部システム統合.
    """

    def __init__(
        self,
        output_format: str = 'markdown',
        output_dir: str = 'reports',
        jira_url: Optional[str] = None,
        jira_project: Optional[str] = None,
        jira_username: Optional[str] = None,
        jira_api_token: Optional[str] = None,
        slack_webhook: Optional[str] = None,
        email_config: Optional[Dict] = None
    ):
        """
        Args:
            output_format: 出力形式 ('json', 'markdown', 'html')
            output_dir: レポート出力ディレクトリ
            jira_url: JIRA URL
            jira_project: JIRAプロジェクトキー
            jira_username: JIRAユーザー名
            jira_api_token: JIRA APIトークン
            slack_webhook: Slack Webhook URL
            email_config: メール設定
        """
        self.output_format = output_format
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # JIRA設定
        self.jira_url = jira_url
        self.jira_project = jira_project
        self.jira_username = jira_username
        self.jira_api_token = jira_api_token
        self.jira_client = None

        # Slack設定
        self.slack_webhook = slack_webhook

        # メール設定
        self.email_config = email_config

    def generate_report(
        self,
        agent_results: Dict,
        output_file: Optional[str] = None
    ) -> str:
        """
        テスト結果からレポートを生成.

        Args:
            agent_results: エージェントの実行結果
            output_file: 出力ファイル名（Noneの場合は自動生成）

        Returns:
            レポートの内容（文字列）
        """
        if self.output_format == 'json':
            return self._generate_json_report(agent_results, output_file)
        elif self.output_format == 'markdown':
            return self._generate_markdown_report(agent_results, output_file)
        elif self.output_format == 'html':
            return self._generate_html_report(agent_results, output_file)
        else:
            raise ValueError(f"Unknown format: {self.output_format}")

    def _generate_json_report(
        self,
        agent_results: Dict,
        output_file: Optional[str] = None
    ) -> str:
        """JSON形式のレポート生成."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'agent_name': agent_results.get('summary', {}).get('agent_name', 'Unknown'),
            'summary': agent_results.get('summary', {}),
            'bugs': agent_results.get('bugs', [])
        }

        report_json = json.dumps(report, indent=2)

        if output_file:
            filepath = self.output_dir / output_file
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f"bug_report_{timestamp}.json"

        with open(filepath, 'w') as f:
            f.write(report_json)

        print(f"✓ JSON report saved: {filepath}")
        return report_json

    def _generate_markdown_report(
        self,
        agent_results: Dict,
        output_file: Optional[str] = None
    ) -> str:
        """Markdown形式のレポート生成."""
        summary = agent_results.get('summary', {})
        bugs = agent_results.get('bugs', [])

        # ヘッダー
        lines = [
            f"# Bug Report - {summary.get('agent_name', 'Unknown Agent')}",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"- **Total Episodes**: {summary.get('episodes_completed', 0)}",
            f"- **Total Steps**: {summary.get('total_steps', 0)}",
            f"- **Unique States Visited**: {summary.get('unique_states_visited', 0)}",
            f"- **Bugs Found**: {summary.get('bugs_found', 0)}",
            f"- **State Coverage**: {summary.get('state_coverage', 0):.1f}%",
            "",
            "---",
            "",
        ]

        # バグがない場合
        if not bugs:
            lines.append("## No Bugs Found")
            lines.append("")
            lines.append("✓ All tests passed successfully!")
            lines.append("")
        else:
            # バグを重要度別に分類
            bugs_by_severity = self._categorize_bugs_by_severity(bugs)

            lines.append("## Bugs Found")
            lines.append("")

            for severity in ['critical', 'high', 'medium', 'low']:
                if severity not in bugs_by_severity:
                    continue

                severity_bugs = bugs_by_severity[severity]
                emoji = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '🟢'
                }

                lines.append(f"### {emoji[severity]} {severity.upper()} ({len(severity_bugs)})")
                lines.append("")

                for i, bug in enumerate(severity_bugs, 1):
                    lines.append(f"#### {i}. {bug['title']}")
                    lines.append("")
                    lines.append(f"**Bug ID**: `{bug['bug_id']}`")
                    lines.append(f"**Timestamp**: {bug['timestamp']}")
                    lines.append("")
                    lines.append(f"**Description**:")
                    lines.append(f"{bug['description']}")
                    lines.append("")

                    if bug.get('steps_to_reproduce'):
                        lines.append(f"**Steps to Reproduce**:")
                        for step_num, step in enumerate(bug['steps_to_reproduce'], 1):
                            lines.append(f"{step_num}. {step}")
                        lines.append("")

                    lines.append(f"**Game State**:")
                    lines.append("```json")
                    lines.append(json.dumps(bug['game_state'], indent=2))
                    lines.append("```")
                    lines.append("")
                    lines.append("---")
                    lines.append("")

        report_md = "\n".join(lines)

        if output_file:
            filepath = self.output_dir / output_file
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f"bug_report_{timestamp}.md"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_md)

        print(f"✓ Markdown report saved: {filepath}")
        return report_md

    def _generate_html_report(
        self,
        agent_results: Dict,
        output_file: Optional[str] = None
    ) -> str:
        """HTML形式のレポート生成."""
        summary = agent_results.get('summary', {})
        bugs = agent_results.get('bugs', [])

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bug Report - {summary.get('agent_name', 'Unknown')}</title>
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
        .timestamp {{
            color: #666;
            font-size: 14px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .summary-item {{
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .summary-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .summary-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        .bug {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            border-left: 4px solid #ccc;
        }}
        .bug.critical {{ border-left-color: #dc3545; }}
        .bug.high {{ border-left-color: #fd7e14; }}
        .bug.medium {{ border-left-color: #ffc107; }}
        .bug.low {{ border-left-color: #28a745; }}
        .bug-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .bug-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}
        .severity-badge {{
            padding: 5px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .severity-badge.critical {{ background: #dc3545; color: white; }}
        .severity-badge.high {{ background: #fd7e14; color: white; }}
        .severity-badge.medium {{ background: #ffc107; color: black; }}
        .severity-badge.low {{ background: #28a745; color: white; }}
        .bug-description {{
            color: #555;
            line-height: 1.6;
            margin-bottom: 15px;
        }}
        .bug-meta {{
            font-size: 12px;
            color: #666;
            margin-bottom: 10px;
        }}
        pre {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 12px;
        }}
        .no-bugs {{
            background: white;
            padding: 40px;
            text-align: center;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .no-bugs-icon {{
            font-size: 64px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Bug Report - {summary.get('agent_name', 'Unknown Agent')}</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="summary">
        <h2>Test Summary</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-label">Episodes</div>
                <div class="summary-value">{summary.get('episodes_completed', 0)}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Total Steps</div>
                <div class="summary-value">{summary.get('total_steps', 0):,}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">States Visited</div>
                <div class="summary-value">{summary.get('unique_states_visited', 0)}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Bugs Found</div>
                <div class="summary-value">{summary.get('bugs_found', 0)}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Coverage</div>
                <div class="summary-value">{summary.get('state_coverage', 0):.1f}%</div>
            </div>
        </div>
    </div>
"""

        if not bugs:
            html += """
    <div class="no-bugs">
        <div class="no-bugs-icon">✓</div>
        <h2>No Bugs Found</h2>
        <p>All tests passed successfully!</p>
    </div>
"""
        else:
            html += "<h2>Bugs Found</h2>\n"

            for bug in bugs:
                severity = bug.get('severity', 'low')
                html += f"""
    <div class="bug {severity}">
        <div class="bug-header">
            <div class="bug-title">{bug['title']}</div>
            <div class="severity-badge {severity}">{severity}</div>
        </div>
        <div class="bug-meta">
            <strong>ID:</strong> {bug['bug_id']} |
            <strong>Time:</strong> {bug['timestamp']}
        </div>
        <div class="bug-description">{bug['description']}</div>
"""

                if bug.get('steps_to_reproduce'):
                    html += "<h4>Steps to Reproduce:</h4>\n<ol>\n"
                    for step in bug['steps_to_reproduce']:
                        html += f"<li>{step}</li>\n"
                    html += "</ol>\n"

                html += f"""
        <h4>Game State:</h4>
        <pre>{json.dumps(bug['game_state'], indent=2)}</pre>
    </div>
"""

        html += """
</body>
</html>
"""

        if output_file:
            filepath = self.output_dir / output_file
        else:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filepath = self.output_dir / f"bug_report_{timestamp}.html"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"✓ HTML report saved: {filepath}")
        return html

    def _categorize_bugs_by_severity(self, bugs: List[Dict]) -> Dict[str, List[Dict]]:
        """バグを重要度別に分類."""
        categorized = defaultdict(list)
        for bug in bugs:
            severity = bug.get('severity', 'low')
            categorized[severity].append(bug)
        return dict(categorized)

    def create_jira_issues(
        self,
        agent_results: Dict,
        severity_filter: Optional[List[str]] = None
    ) -> List[str]:
        """
        JIRA課題を自動作成.

        Args:
            agent_results: エージェント実行結果
            severity_filter: 作成する重要度（例: ['critical', 'high']）

        Returns:
            作成された課題キーのリスト
        """
        if not self.jira_url or not self.jira_project:
            print("⚠️ JIRA configuration not set")
            return []

        try:
            from jira import JIRA
        except ImportError:
            print("⚠️ jira library not installed. Install with: pip install jira")
            return []

        if not self.jira_client:
            self.jira_client = JIRA(
                server=self.jira_url,
                basic_auth=(self.jira_username, self.jira_api_token)
            )

        bugs = agent_results.get('bugs', [])
        if severity_filter:
            bugs = [b for b in bugs if b['severity'] in severity_filter]

        created_issues = []

        for bug in bugs:
            issue_dict = {
                'project': {'key': self.jira_project},
                'summary': bug['title'],
                'description': self._format_jira_description(bug),
                'issuetype': {'name': 'Bug'},
                'priority': {'name': self._severity_to_jira_priority(bug['severity'])}
            }

            try:
                new_issue = self.jira_client.create_issue(fields=issue_dict)
                created_issues.append(new_issue.key)
                print(f"✓ Created JIRA issue: {new_issue.key}")
            except Exception as e:
                print(f"✗ Failed to create JIRA issue: {e}")

        return created_issues

    def _format_jira_description(self, bug: Dict) -> str:
        """JIRA用の説明文を生成."""
        desc = f"{bug['description']}\n\n"

        if bug.get('steps_to_reproduce'):
            desc += "*Steps to Reproduce:*\n"
            for i, step in enumerate(bug['steps_to_reproduce'], 1):
                desc += f"# {step}\n"
            desc += "\n"

        desc += "*Game State:*\n{code:json}\n"
        desc += json.dumps(bug['game_state'], indent=2)
        desc += "\n{code}\n"

        desc += f"\n*Bug ID:* {bug['bug_id']}\n"
        desc += f"*Detected:* {bug['timestamp']}\n"

        return desc

    def _severity_to_jira_priority(self, severity: str) -> str:
        """重要度をJIRA優先度にマッピング."""
        mapping = {
            'critical': 'Highest',
            'high': 'High',
            'medium': 'Medium',
            'low': 'Low'
        }
        return mapping.get(severity, 'Medium')

    def send_slack_notification(
        self,
        agent_results: Dict,
        channel: Optional[str] = None
    ) -> bool:
        """
        Slack通知を送信.

        Args:
            agent_results: エージェント実行結果
            channel: 送信先チャンネル（オプション）

        Returns:
            成功したかどうか
        """
        if not self.slack_webhook:
            print("⚠️ Slack webhook not configured")
            return False

        try:
            import requests
        except ImportError:
            print("⚠️ requests library not installed")
            return False

        summary = agent_results.get('summary', {})
        bugs = agent_results.get('bugs', [])

        # メッセージ構築
        text = f"*Automated Test Results - {summary.get('agent_name', 'Unknown')}*\n\n"
        text += f"Episodes: {summary.get('episodes_completed', 0)} | "
        text += f"Bugs Found: {summary.get('bugs_found', 0)}\n\n"

        if bugs:
            bugs_by_severity = self._categorize_bugs_by_severity(bugs)
            text += "*Bug Breakdown:*\n"
            for severity in ['critical', 'high', 'medium', 'low']:
                if severity in bugs_by_severity:
                    count = len(bugs_by_severity[severity])
                    emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}
                    text += f"{emoji[severity]} {severity.capitalize()}: {count}\n"
        else:
            text += "✓ No bugs found!"

        payload = {'text': text}
        if channel:
            payload['channel'] = channel

        response = requests.post(self.slack_webhook, json=payload)

        if response.status_code == 200:
            print("✓ Slack notification sent")
            return True
        else:
            print(f"✗ Failed to send Slack notification: {response.status_code}")
            return False


if __name__ == "__main__":
    # デモ: レポート生成
    print("Bug Reporter Demo")
    print("=" * 60)

    # ダミーのテスト結果
    dummy_results = {
        'summary': {
            'agent_name': 'DemoAgent',
            'episodes_completed': 100,
            'total_steps': 5000,
            'unique_states_visited': 500,
            'bugs_found': 3,
            'state_coverage': 75.5
        },
        'bugs': [
            {
                'bug_id': 'BUG-001',
                'severity': 'critical',
                'title': 'Player falls through floor',
                'description': 'Player clips through floor when jumping at (125, 45)',
                'timestamp': '2025-01-15 14:32:10',
                'steps_to_reproduce': [
                    'Load Level 3',
                    'Move to coordinates (125, 45)',
                    'Jump while moving right'
                ],
                'game_state': {
                    'level': 'Level_3',
                    'player_position': [125, 45, 0],
                    'health': 100
                }
            },
            {
                'bug_id': 'BUG-002',
                'severity': 'high',
                'title': 'Negative health value',
                'description': 'Health became -50 after taking damage',
                'timestamp': '2025-01-15 14:35:20',
                'steps_to_reproduce': [],
                'game_state': {
                    'level': 'Level_1',
                    'health': -50
                }
            },
            {
                'bug_id': 'BUG-003',
                'severity': 'medium',
                'title': 'Enemy spawn leak',
                'description': 'Enemy count continuously increasing',
                'timestamp': '2025-01-15 14:40:15',
                'steps_to_reproduce': [],
                'game_state': {
                    'enemies': 150
                }
            }
        ]
    }

    # レポーター作成
    reporter = BugReporter(output_dir='demo_reports')

    # 1. Markdownレポート
    print("\nGenerating Markdown report...")
    reporter.output_format = 'markdown'
    reporter.generate_report(dummy_results, 'demo_report.md')

    # 2. HTMLレポート
    print("\nGenerating HTML report...")
    reporter.output_format = 'html'
    reporter.generate_report(dummy_results, 'demo_report.html')

    # 3. JSONレポート
    print("\nGenerating JSON report...")
    reporter.output_format = 'json'
    reporter.generate_report(dummy_results, 'demo_report.json')

    print("\n✓ Demo completed! Check the 'demo_reports' directory.")
