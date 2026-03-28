"""
Advanced Game Review Sentiment Analysis

ゲームレビューに特化した高度な感情分析システム。
基本的な肯定/否定だけでなく、アスペクトベース分析、毒性検出、
プレイヤーの意図分類などを行います。

特徴:
    - Aspect-based sentiment analysis (ゲームプレイ、グラフィックなど)
    - Toxicity detection (有害なコメント検出)
    - Player intent classification (バグレポート、要望など)
    - Emotion analysis (怒り、喜び、失望など)
    - Multi-language support

使い方:
    from game_review_analyzer import GameReviewAnalyzer

    analyzer = GameReviewAnalyzer()

    review = "Great graphics but terrible gameplay. Keeps crashing!"
    result = analyzer.analyze_review(review)

    print(f"Overall sentiment: {result['sentiment']}")
    print(f"Aspects: {result['aspects']}")
    print(f"Intent: {result['intent']}")
    print(f"Toxicity: {result['toxicity_score']}")

参考:
    - ABSA (Aspect-Based Sentiment Analysis)
    - Perspective API (Google Jigsaw)
    - Multi-task learning for review analysis
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
from collections import Counter
from dataclasses import dataclass
import json


@dataclass
class ReviewAnalysis:
    """レビュー分析結果."""
    overall_sentiment: str          # positive, negative, neutral
    sentiment_score: float          # -1.0 to 1.0
    aspects: Dict[str, Dict]        # アスペクトごとの感情
    emotions: Dict[str, float]      # 感情スコア
    intent: str                     # bug_report, feature_request, praise, complaint
    toxicity_score: float           # 0.0 to 1.0
    key_phrases: List[str]          # 重要なフレーズ
    actionable: bool                # アクションが必要か
    priority: str                   # high, medium, low

    def to_dict(self) -> Dict:
        """辞書に変換."""
        return {
            'overall_sentiment': self.overall_sentiment,
            'sentiment_score': self.sentiment_score,
            'aspects': self.aspects,
            'emotions': self.emotions,
            'intent': self.intent,
            'toxicity_score': self.toxicity_score,
            'key_phrases': self.key_phrases,
            'actionable': self.actionable,
            'priority': self.priority
        }


class GameAspectExtractor:
    """
    ゲームレビューからアスペクトを抽出.

    アスペクト例:
        - gameplay (ゲームプレイ)
        - graphics (グラフィック)
        - story (ストーリー)
        - audio (音響)
        - performance (パフォーマンス)
        - ui (UI/UX)
        - multiplayer (マルチプレイヤー)
        - monetization (収益化)
    """

    def __init__(self):
        # アスペクトキーワード辞書
        self.aspect_keywords = {
            'gameplay': [
                'gameplay', 'mechanics', 'controls', 'combat', 'difficulty',
                'balance', 'fun', 'boring', 'engaging', 'addictive', 'repetitive'
            ],
            'graphics': [
                'graphics', 'visuals', 'art', 'graphics', 'animation', 'style',
                'beautiful', 'ugly', 'realistic', 'textures', 'models'
            ],
            'story': [
                'story', 'plot', 'narrative', 'characters', 'dialogue',
                'writing', 'lore', 'campaign', 'quest'
            ],
            'audio': [
                'music', 'sound', 'audio', 'soundtrack', 'voice',
                'sfx', 'effects', 'ambient'
            ],
            'performance': [
                'performance', 'fps', 'lag', 'crash', 'bug', 'glitch',
                'optimization', 'loading', 'framerate', 'stutter'
            ],
            'ui': [
                'ui', 'interface', 'menu', 'hud', 'inventory', 'map',
                'navigation', 'usability'
            ],
            'multiplayer': [
                'multiplayer', 'online', 'coop', 'pvp', 'matchmaking',
                'server', 'netcode', 'ping', 'latency'
            ],
            'monetization': [
                'price', 'dlc', 'microtransaction', 'mtx', 'pay2win',
                'expensive', 'cheap', 'value', 'worth'
            ]
        }

        # 感情表現パターン
        self.sentiment_patterns = {
            'positive': [
                'amazing', 'great', 'awesome', 'love', 'perfect',
                'excellent', 'fantastic', 'incredible', 'best'
            ],
            'negative': [
                'terrible', 'awful', 'hate', 'worst', 'bad',
                'horrible', 'disappointing', 'broken', 'trash'
            ]
        }

    def extract_aspects(self, text: str) -> Dict[str, List[str]]:
        """
        テキストからアスペクトを抽出.

        Args:
            text: レビューテキスト

        Returns:
            aspects: {aspect_name: [relevant_sentences]}
        """
        text_lower = text.lower()
        sentences = self._split_sentences(text)

        aspects = {}

        for aspect, keywords in self.aspect_keywords.items():
            relevant_sentences = []

            for sentence in sentences:
                sentence_lower = sentence.lower()

                # キーワードマッチング
                if any(keyword in sentence_lower for keyword in keywords):
                    relevant_sentences.append(sentence)

            if relevant_sentences:
                aspects[aspect] = relevant_sentences

        return aspects

    def _split_sentences(self, text: str) -> List[str]:
        """文に分割."""
        # 簡易的な文分割
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


class IntentClassifier:
    """
    プレイヤーの意図を分類.

    意図カテゴリ:
        - bug_report: バグレポート
        - feature_request: 機能要望
        - praise: 賞賛
        - complaint: 不満
        - question: 質問
        - feedback: フィードバック
    """

    def __init__(self):
        # 意図パターン
        self.intent_patterns = {
            'bug_report': [
                r'crash', r'bug', r'glitch', r'broken', r'not working',
                r'error', r'freeze', r'stuck', r'can\'t', r'won\'t'
            ],
            'feature_request': [
                r'should add', r'would like', r'hope', r'wish',
                r'please add', r'need', r'want', r'suggestion'
            ],
            'praise': [
                r'amazing', r'love', r'great', r'best', r'perfect',
                r'thank you', r'awesome', r'fantastic'
            ],
            'complaint': [
                r'terrible', r'awful', r'hate', r'worst', r'disappointed',
                r'waste', r'regret', r'refund'
            ],
            'question': [
                r'\?', r'how to', r'is there', r'can i', r'does it',
                r'wondering', r'anyone know'
            ]
        }

    def classify(self, text: str) -> Tuple[str, float]:
        """
        意図を分類.

        Args:
            text: レビューテキスト

        Returns:
            (intent, confidence)
        """
        text_lower = text.lower()
        scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            scores[intent] = score

        if not scores or max(scores.values()) == 0:
            return 'feedback', 0.5

        # 最も高いスコアの意図
        max_intent = max(scores, key=scores.get)
        max_score = scores[max_intent]

        # 信頼度計算（正規化）
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5

        return max_intent, confidence


class ToxicityDetector:
    """
    有害性検出.

    検出項目:
        - Profanity (罵倒語)
        - Personal attacks (個人攻撃)
        - Hate speech (ヘイトスピーチ)
        - Threats (脅迫)
    """

    def __init__(self):
        # 有害語リスト（簡易版）
        self.toxic_keywords = [
            'stupid', 'idiot', 'moron', 'dumb', 'trash', 'garbage',
            'kill yourself', 'kys', 'die', 'hate you'
        ]

        # 強調パターン（大文字、連続文字）
        self.emphasis_pattern = r'([A-Z]{3,}|(.)\2{2,})'

    def detect(self, text: str) -> Tuple[float, List[str]]:
        """
        有害性を検出.

        Args:
            text: テキスト

        Returns:
            (toxicity_score, detected_terms)
        """
        text_lower = text.lower()
        detected = []
        score = 0.0

        # キーワードチェック
        for keyword in self.toxic_keywords:
            if keyword in text_lower:
                detected.append(keyword)
                score += 0.3

        # 大文字の連続（怒りの表現）
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.5:
            score += 0.2
            detected.append('EXCESSIVE_CAPS')

        # 連続文字（強調）
        if re.search(self.emphasis_pattern, text):
            score += 0.1

        # スコアを0-1に制限
        score = min(score, 1.0)

        return score, detected


class GameReviewAnalyzer:
    """
    ゲームレビューの総合分析システム.

    複数のコンポーネントを統合して詳細な分析を提供します.
    """

    def __init__(
        self,
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        use_gpu: bool = True
    ):
        """
        Args:
            sentiment_model: 感情分析モデル名
            use_gpu: GPU使用
        """
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1

        print(f"Loading sentiment model: {sentiment_model}...")

        # 感情分析パイプライン
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=sentiment_model,
            device=self.device
        )

        # コンポーネント
        self.aspect_extractor = GameAspectExtractor()
        self.intent_classifier = IntentClassifier()
        self.toxicity_detector = ToxicityDetector()

        print("✓ GameReviewAnalyzer initialized")

    def analyze_review(self, text: str) -> ReviewAnalysis:
        """
        レビューを総合的に分析.

        Args:
            text: レビューテキスト

        Returns:
            analysis: 分析結果
        """
        # 1. 全体的な感情分析
        overall_sentiment = self.sentiment_pipeline(text[:512])[0]
        sentiment_label = overall_sentiment['label'].lower()
        sentiment_score = overall_sentiment['score']

        # スコアを-1から1に変換
        if sentiment_label == 'negative':
            sentiment_score = -sentiment_score

        # 2. アスペクトベース分析
        aspects = self.aspect_extractor.extract_aspects(text)
        aspect_sentiments = {}

        for aspect, sentences in aspects.items():
            # 各文の感情を分析
            sentiments = []
            for sentence in sentences:
                if sentence.strip():
                    sent = self.sentiment_pipeline(sentence[:512])[0]
                    score = sent['score'] if sent['label'] == 'POSITIVE' else -sent['score']
                    sentiments.append(score)

            if sentiments:
                avg_score = np.mean(sentiments)
                aspect_sentiments[aspect] = {
                    'score': float(avg_score),
                    'sentiment': 'positive' if avg_score > 0 else 'negative',
                    'mentions': len(sentences)
                }

        # 3. 感情分析（簡易版）
        emotions = self._analyze_emotions(text)

        # 4. 意図分類
        intent, intent_confidence = self.intent_classifier.classify(text)

        # 5. 有害性検出
        toxicity_score, toxic_terms = self.toxicity_detector.detect(text)

        # 6. キーフレーズ抽出
        key_phrases = self._extract_key_phrases(text, aspects)

        # 7. アクション可能性とプライオリティ
        actionable = self._is_actionable(intent, toxicity_score)
        priority = self._calculate_priority(intent, toxicity_score, sentiment_score)

        return ReviewAnalysis(
            overall_sentiment=sentiment_label,
            sentiment_score=sentiment_score,
            aspects=aspect_sentiments,
            emotions=emotions,
            intent=intent,
            toxicity_score=toxicity_score,
            key_phrases=key_phrases,
            actionable=actionable,
            priority=priority
        )

    def analyze_batch(
        self,
        reviews: List[str],
        batch_size: int = 32
    ) -> List[ReviewAnalysis]:
        """
        複数のレビューをバッチ分析.

        Args:
            reviews: レビューリスト
            batch_size: バッチサイズ

        Returns:
            results: 分析結果リスト
        """
        results = []

        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]

            for review in batch:
                analysis = self.analyze_review(review)
                results.append(analysis)

        return results

    def aggregate_insights(
        self,
        analyses: List[ReviewAnalysis]
    ) -> Dict:
        """
        複数の分析結果から洞察を集約.

        Args:
            analyses: 分析結果リスト

        Returns:
            insights: 集約された洞察
        """
        if not analyses:
            return {}

        # 感情分布
        sentiment_counts = Counter(a.overall_sentiment for a in analyses)

        # アスペクトスコア集計
        aspect_scores = {}
        for analysis in analyses:
            for aspect, data in analysis.aspects.items():
                if aspect not in aspect_scores:
                    aspect_scores[aspect] = []
                aspect_scores[aspect].append(data['score'])

        aspect_averages = {
            aspect: np.mean(scores)
            for aspect, scores in aspect_scores.items()
        }

        # 意図分布
        intent_counts = Counter(a.intent for a in analyses)

        # 有害性統計
        toxic_reviews = sum(1 for a in analyses if a.toxicity_score > 0.5)

        # アクション可能なレビュー
        actionable_reviews = sum(1 for a in analyses if a.actionable)
        high_priority = sum(1 for a in analyses if a.priority == 'high')

        # 共通キーフレーズ
        all_phrases = []
        for analysis in analyses:
            all_phrases.extend(analysis.key_phrases)
        common_phrases = Counter(all_phrases).most_common(10)

        insights = {
            'total_reviews': len(analyses),
            'sentiment_distribution': dict(sentiment_counts),
            'sentiment_ratio': {
                'positive': sentiment_counts.get('positive', 0) / len(analyses),
                'negative': sentiment_counts.get('negative', 0) / len(analyses),
                'neutral': sentiment_counts.get('neutral', 0) / len(analyses)
            },
            'aspect_scores': aspect_averages,
            'intent_distribution': dict(intent_counts),
            'toxicity': {
                'toxic_count': toxic_reviews,
                'toxic_ratio': toxic_reviews / len(analyses)
            },
            'actionable': {
                'actionable_count': actionable_reviews,
                'high_priority_count': high_priority
            },
            'common_phrases': [phrase for phrase, count in common_phrases]
        }

        return insights

    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """簡易的な感情分析."""
        text_lower = text.lower()

        emotion_keywords = {
            'anger': ['angry', 'furious', 'mad', 'annoyed', 'frustrated'],
            'joy': ['happy', 'excited', 'love', 'enjoy', 'fun'],
            'sadness': ['sad', 'disappointed', 'depressed', 'upset'],
            'fear': ['scared', 'afraid', 'worried', 'anxious'],
            'surprise': ['surprised', 'shocked', 'unexpected', 'wow']
        }

        emotions = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotions[emotion] = min(score / 3.0, 1.0)  # 正規化

        return emotions

    def _extract_key_phrases(self, text: str, aspects: Dict) -> List[str]:
        """重要なフレーズを抽出."""
        phrases = []

        # アスペクトから抽出
        for aspect, sentences in aspects.items():
            for sentence in sentences[:2]:  # 最初の2文
                # 短いフレーズを抽出（10-50文字）
                if 10 < len(sentence) < 50:
                    phrases.append(sentence.strip())

        return phrases[:5]  # 上位5つ

    def _is_actionable(self, intent: str, toxicity: float) -> bool:
        """アクション可能かどうか."""
        actionable_intents = ['bug_report', 'feature_request', 'complaint']
        return intent in actionable_intents or toxicity > 0.7

    def _calculate_priority(
        self,
        intent: str,
        toxicity: float,
        sentiment: float
    ) -> str:
        """プライオリティを計算."""
        score = 0

        # 意図ベース
        if intent == 'bug_report':
            score += 3
        elif intent == 'complaint':
            score += 2
        elif intent == 'feature_request':
            score += 1

        # 有害性
        if toxicity > 0.7:
            score += 2

        # 感情
        if sentiment < -0.7:
            score += 1

        if score >= 4:
            return 'high'
        elif score >= 2:
            return 'medium'
        else:
            return 'low'

    def export_report(
        self,
        analyses: List[ReviewAnalysis],
        filepath: str,
        format: str = 'json'
    ):
        """分析結果をエクスポート."""
        insights = self.aggregate_insights(analyses)

        report = {
            'summary': insights,
            'reviews': [a.to_dict() for a in analyses]
        }

        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✓ Report exported to {filepath}")


# ============================================================================
# デモ: ゲームレビューの分析
# ============================================================================

if __name__ == "__main__":
    print("Advanced Game Review Analysis Demo")
    print("=" * 60)

    # サンプルレビュー
    sample_reviews = [
        "Amazing graphics and smooth gameplay! Love the story. Best game I've played this year!",
        "Terrible optimization. Game keeps crashing on my PC. Can't even finish the tutorial. Waste of money.",
        "Great concept but needs more content. Would love to see co-op mode added in future updates.",
        "The UI is confusing and the controls feel clunky. Story is okay but gameplay is boring.",
        "WORST GAME EVER! Total cash grab. Developers are idiots. Refunded immediately.",
        "Decent game overall. Graphics are beautiful but performance could be better. Story is engaging.",
        "Bug report: Can't progress past level 5 due to a crash. Please fix!",
        "Pretty good for the price. Some minor bugs but nothing game-breaking. Recommended."
    ]

    # アナライザー作成
    analyzer = GameReviewAnalyzer(use_gpu=False)  # CPUモード

    print("\n" + "=" * 60)
    print("Analyzing Individual Reviews")
    print("=" * 60)

    # 個別分析
    for i, review in enumerate(sample_reviews[:3], 1):
        print(f"\n--- Review {i} ---")
        print(f"Text: {review}")

        analysis = analyzer.analyze_review(review)

        print(f"\nOverall Sentiment: {analysis.overall_sentiment} ({analysis.sentiment_score:.2f})")
        print(f"Intent: {analysis.intent}")
        print(f"Toxicity: {analysis.toxicity_score:.2f}")
        print(f"Priority: {analysis.priority}")

        if analysis.aspects:
            print(f"Aspects mentioned:")
            for aspect, data in analysis.aspects.items():
                print(f"  - {aspect}: {data['sentiment']} ({data['score']:.2f})")

    print("\n" + "=" * 60)
    print("Batch Analysis & Insights")
    print("=" * 60)

    # バッチ分析
    all_analyses = analyzer.analyze_batch(sample_reviews)

    # 洞察を集約
    insights = analyzer.aggregate_insights(all_analyses)

    print(f"\nTotal Reviews: {insights['total_reviews']}")
    print(f"\nSentiment Distribution:")
    for sentiment, ratio in insights['sentiment_ratio'].items():
        print(f"  {sentiment}: {ratio:.1%}")

    print(f"\nIntent Distribution:")
    for intent, count in insights['intent_distribution'].items():
        print(f"  {intent}: {count}")

    print(f"\nAspect Scores:")
    for aspect, score in sorted(insights['aspect_scores'].items(), key=lambda x: x[1], reverse=True):
        sentiment_label = "positive" if score > 0 else "negative"
        print(f"  {aspect}: {score:.2f} ({sentiment_label})")

    print(f"\nToxicity:")
    print(f"  Toxic reviews: {insights['toxicity']['toxic_count']}/{insights['total_reviews']}")
    print(f"  Toxic ratio: {insights['toxicity']['toxic_ratio']:.1%}")

    print(f"\nActionable Items:")
    print(f"  Actionable reviews: {insights['actionable']['actionable_count']}")
    print(f"  High priority: {insights['actionable']['high_priority_count']}")

    # レポートをエクスポート
    analyzer.export_report(all_analyses, 'game_review_analysis.json')

    print("\n✓ Demo completed!")
    print("\nNext steps:")
    print("  1. Integrate with Steam API for real reviews")
    print("  2. Build dashboard for visualization")
    print("  3. Set up alerts for high-priority issues")
    print("  4. Train custom models on game-specific data")
