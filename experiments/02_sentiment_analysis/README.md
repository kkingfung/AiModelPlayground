# Experiment 02: Advanced Game Review Sentiment Analysis

ゲームレビューに特化した高度な感情分析システム

---

## 概要

このモジュールは、基本的な感情分析（肯定/否定）を超えて、ゲームレビューの深い洞察を提供します。

### 主要機能

1. **Aspect-Based Sentiment Analysis (アスペクトベース分析)**
   - ゲームプレイ、グラフィック、ストーリーなど個別の要素を分析
   - 各アスペクトの感情スコアを算出

2. **Intent Classification (意図分類)**
   - バグレポート、機能要望、賞賛、不満などを自動分類
   - アクション可能なレビューを特定

3. **Toxicity Detection (有害性検出)**
   - 攻撃的、有害なコメントを検出
   - コミュニティ管理のサポート

4. **Emotion Analysis (感情分析)**
   - 怒り、喜び、失望などの感情を検出

5. **Priority Scoring (優先度スコアリング)**
   - レビューの重要度を自動計算
   - 緊急対応が必要な問題を特定

---

## クイックスタート

### インストール

```bash
cd experiments/02_sentiment_analysis
pip install -r requirements.txt
```

### 基本的な使用

```python
from game_review_analyzer import GameReviewAnalyzer

# アナライザー作成
analyzer = GameReviewAnalyzer()

# レビューを分析
review = "Great graphics but terrible gameplay. Keeps crashing!"
result = analyzer.analyze_review(review)

print(f"Sentiment: {result.overall_sentiment}")
print(f"Aspects: {result.aspects}")
print(f"Intent: {result.intent}")
print(f"Priority: {result.priority}")
```

### バッチ分析

```python
# 複数レビューを分析
reviews = [
    "Amazing game! Love it!",
    "Terrible optimization. Always crashing.",
    "Good concept but needs more content."
]

analyses = analyzer.analyze_batch(reviews)

# 洞察を集約
insights = analyzer.aggregate_insights(analyses)

print(f"Positive ratio: {insights['sentiment_ratio']['positive']:.1%}")
print(f"Top aspects: {insights['aspect_scores']}")
```

---

## アーキテクチャ

```
GameReviewAnalyzer
├── GameAspectExtractor      # アスペクト抽出
├── IntentClassifier          # 意図分類
├── ToxicityDetector         # 有害性検出
└── Sentiment Pipeline       # 全体感情分析（HuggingFace）
```

### アスペクトカテゴリ

| カテゴリ | 説明 | キーワード例 |
|---------|------|------------|
| gameplay | ゲームプレイ | mechanics, controls, combat, difficulty |
| graphics | グラフィック | visuals, art, animation, beautiful |
| story | ストーリー | plot, narrative, characters, dialogue |
| audio | 音響 | music, sound, soundtrack, voice |
| performance | パフォーマンス | fps, lag, crash, bug, optimization |
| ui | UI/UX | interface, menu, hud, usability |
| multiplayer | マルチプレイヤー | online, coop, pvp, matchmaking |
| monetization | 収益化 | price, dlc, microtransaction, value |

### 意図カテゴリ

- **bug_report**: バグの報告
- **feature_request**: 新機能の要望
- **praise**: ゲームの称賛
- **complaint**: 不満・苦情
- **question**: 質問
- **feedback**: 一般的なフィードバック

---

## 使用例

### 1. Steam レビュー分析

```python
import requests

# Steam APIからレビューを取得（例）
def fetch_steam_reviews(app_id):
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {'json': 1, 'filter': 'recent', 'language': 'english'}
    response = requests.get(url, params=params)
    return response.json()

# レビュー分析
analyzer = GameReviewAnalyzer()
reviews = fetch_steam_reviews('123456')

for review in reviews['reviews']:
    analysis = analyzer.analyze_review(review['review'])

    if analysis.priority == 'high':
        print(f"HIGH PRIORITY: {review['review'][:100]}...")
        print(f"  Intent: {analysis.intent}")
        print(f"  Aspects: {analysis.aspects.keys()}")
```

### 2. ダッシュボード用データ生成

```python
from collections import defaultdict
import datetime

def generate_daily_report(reviews_by_date):
    """日次レポート生成."""
    analyzer = GameReviewAnalyzer()

    report = {}
    for date, reviews in reviews_by_date.items():
        analyses = analyzer.analyze_batch(reviews)
        insights = analyzer.aggregate_insights(analyses)

        report[date] = {
            'total': len(reviews),
            'sentiment': insights['sentiment_ratio'],
            'top_aspects': insights['aspect_scores'],
            'actionable_count': insights['actionable']['actionable_count'],
            'toxic_ratio': insights['toxicity']['toxic_ratio']
        }

    return report
```

### 3. アラートシステム

```python
def check_for_alerts(analysis):
    """アラートが必要かチェック."""
    alerts = []

    # バグレポート
    if analysis.intent == 'bug_report' and analysis.priority == 'high':
        alerts.append({
            'type': 'bug',
            'severity': 'high',
            'message': 'Critical bug reported'
        })

    # 有害コメント
    if analysis.toxicity_score > 0.8:
        alerts.append({
            'type': 'toxicity',
            'severity': 'high',
            'message': 'Highly toxic comment detected'
        })

    # パフォーマンス問題
    if 'performance' in analysis.aspects:
        perf_score = analysis.aspects['performance']['score']
        if perf_score < -0.7:
            alerts.append({
                'type': 'performance',
                'severity': 'medium',
                'message': 'Performance complaints increasing'
            })

    return alerts

# 使用例
analyzer = GameReviewAnalyzer()
for review in new_reviews:
    analysis = analyzer.analyze_review(review)
    alerts = check_for_alerts(analysis)

    for alert in alerts:
        send_notification(alert)  # Slack, email, etc.
```

---

## カスタマイズ

### アスペクトキーワードの追加

```python
analyzer = GameReviewAnalyzer()

# カスタムアスペクトを追加
analyzer.aspect_extractor.aspect_keywords['accessibility'] = [
    'accessibility', 'colorblind', 'subtitles', 'difficulty options'
]

# 分析実行
result = analyzer.analyze_review(review_text)
```

### カスタムモデルの使用

```python
# より高度なモデルを使用
analyzer = GameReviewAnalyzer(
    sentiment_model="bert-base-uncased",
    use_gpu=True
)
```

---

## パフォーマンス

### ベンチマーク

| タスク | 速度 (CPU) | 速度 (GPU) |
|-------|-----------|-----------|
| 単一レビュー分析 | 200-300ms | 50-100ms |
| バッチ分析 (100件) | 15-20s | 4-6s |
| 大規模分析 (10,000件) | 25-35分 | 7-10分 |

### 最適化のヒント

```python
# バッチ処理で高速化
reviews = [...] # 大量のレビュー
analyses = analyzer.analyze_batch(reviews, batch_size=64)

# GPU使用
analyzer = GameReviewAnalyzer(use_gpu=True)

# 軽量モデル
analyzer = GameReviewAnalyzer(
    sentiment_model="distilbert-base-uncased-finetuned-sst-2-english"
)
```

---

## 出力形式

### ReviewAnalysis オブジェクト

```python
ReviewAnalysis(
    overall_sentiment='negative',
    sentiment_score=-0.85,
    aspects={
        'gameplay': {'score': -0.9, 'sentiment': 'negative', 'mentions': 2},
        'graphics': {'score': 0.8, 'sentiment': 'positive', 'mentions': 1}
    },
    emotions={
        'anger': 0.8,
        'joy': 0.1,
        'sadness': 0.3,
        'fear': 0.0,
        'surprise': 0.2
    },
    intent='bug_report',
    toxicity_score=0.3,
    key_phrases=['terrible gameplay', 'keeps crashing'],
    actionable=True,
    priority='high'
)
```

### Insights (集約データ)

```python
{
    'total_reviews': 100,
    'sentiment_distribution': {
        'positive': 60,
        'negative': 35,
        'neutral': 5
    },
    'sentiment_ratio': {
        'positive': 0.60,
        'negative': 0.35,
        'neutral': 0.05
    },
    'aspect_scores': {
        'gameplay': 0.45,
        'graphics': 0.82,
        'story': 0.15,
        'performance': -0.65
    },
    'intent_distribution': {
        'praise': 45,
        'bug_report': 20,
        'feedback': 20,
        'complaint': 15
    },
    'toxicity': {
        'toxic_count': 5,
        'toxic_ratio': 0.05
    },
    'actionable': {
        'actionable_count': 35,
        'high_priority_count': 8
    }
}
```

---

## 実践的な応用

### 1. リアルタイムモニタリング

```python
import time

def monitor_reviews_realtime(api_endpoint, check_interval=300):
    """5分ごとに新しいレビューをチェック."""
    analyzer = GameReviewAnalyzer()
    last_check = time.time()

    while True:
        new_reviews = fetch_reviews_since(api_endpoint, last_check)

        for review in new_reviews:
            analysis = analyzer.analyze_review(review['text'])

            # 高優先度のレビューをアラート
            if analysis.priority == 'high':
                send_slack_alert({
                    'text': review['text'][:200],
                    'intent': analysis.intent,
                    'aspects': list(analysis.aspects.keys())
                })

        last_check = time.time()
        time.sleep(check_interval)
```

### 2. A/Bテスト評価

```python
def compare_versions(reviews_v1, reviews_v2):
    """2つのバージョンのレビューを比較."""
    analyzer = GameReviewAnalyzer()

    analyses_v1 = analyzer.analyze_batch(reviews_v1)
    analyses_v2 = analyzer.analyze_batch(reviews_v2)

    insights_v1 = analyzer.aggregate_insights(analyses_v1)
    insights_v2 = analyzer.aggregate_insights(analyses_v2)

    print("Version 1 vs Version 2:")
    print(f"Positive ratio: {insights_v1['sentiment_ratio']['positive']:.1%} vs {insights_v2['sentiment_ratio']['positive']:.1%}")

    for aspect in insights_v1['aspect_scores']:
        if aspect in insights_v2['aspect_scores']:
            v1_score = insights_v1['aspect_scores'][aspect]
            v2_score = insights_v2['aspect_scores'][aspect]
            change = v2_score - v1_score
            print(f"{aspect}: {v1_score:.2f} → {v2_score:.2f} ({change:+.2f})")
```

---

## トラブルシューティング

### メモリ不足

```python
# 小さいバッチサイズを使用
analyses = analyzer.analyze_batch(reviews, batch_size=16)
```

### 遅い分析

```python
# GPUを使用
analyzer = GameReviewAnalyzer(use_gpu=True)

# または軽量モデル
analyzer = GameReviewAnalyzer(
    sentiment_model="distilbert-base-uncased-finetuned-sst-2-english"
)
```

---

## 今後の改善

- [ ] カスタムトレーニング機能
- [ ] 多言語サポート
- [ ] より高度なアスペクト抽出（BERT-based）
- [ ] リアルタイムストリーミング分析
- [ ] ダッシュボードUI

---

## 参考文献

- Aspect-Based Sentiment Analysis: https://arxiv.org/abs/1804.07821
- BERT for Sentiment Analysis: https://arxiv.org/abs/1810.04805
- Toxicity Detection: Perspective API by Jigsaw

---

**Built for game developers, by game developers 🎮**
