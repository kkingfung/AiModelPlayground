"""
Player Feedback Sentiment Analyzer

プレイヤーレビュー・フィードバックの感情分析.
ポジティブ/ネガティブ判定、トピック抽出、改善点発見.

使い方:
    # 単一レビュー分析
    python sentiment_analyzer.py \
        --text "This game is amazing! Great graphics and gameplay."

    # バッチ分析（JSONファイル）
    python sentiment_analyzer.py \
        --analyze data/reviews.json \
        --output analysis_results.json

    # ファインチューニング（ゲーム固有）
    python sentiment_analyzer.py \
        --train labeled_reviews.json \
        --epochs 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import numpy as np
from collections import Counter
from tqdm import tqdm


class SentimentAnalyzer:
    """
    感情分析器.

    プレイヤーレビューやフィードバックの感情を分析し、
    ポジティブ/ネガティブ/ニュートラルを判定します.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: ベースモデル名
            device: デバイス（cuda/cpu）
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading sentiment model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)

        # ラベルマッピング
        self.label_map = {0: "NEGATIVE", 1: "POSITIVE"}

    def analyze(
        self,
        text: str
    ) -> Dict:
        """
        テキストの感情分析.

        Args:
            text: 分析するテキスト

        Returns:
            感情分析結果（ラベル、スコア、信頼度）
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]

        # 最も高いスコアのラベルを取得
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

        return {
            "text": text,
            "sentiment": self.label_map.get(predicted_class, "NEUTRAL"),
            "confidence": confidence,
            "scores": {
                "negative": probabilities[0].item(),
                "positive": probabilities[1].item()
            }
        }

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        複数テキストの一括分析.

        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ

        Returns:
            分析結果のリスト
        """
        results = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing"):
            batch_texts = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)

            for text, probs in zip(batch_texts, probabilities):
                predicted_class = torch.argmax(probs).item()
                confidence = probs[predicted_class].item()

                results.append({
                    "text": text,
                    "sentiment": self.label_map.get(predicted_class, "NEUTRAL"),
                    "confidence": confidence,
                    "scores": {
                        "negative": probs[0].item(),
                        "positive": probs[1].item()
                    }
                })

        return results

    def analyze_reviews(
        self,
        reviews: List[Dict]
    ) -> Dict:
        """
        レビューデータセットを包括的に分析.

        Args:
            reviews: レビューのリスト [{"text": "...", "rating": 5}, ...]

        Returns:
            統計とインサイト
        """
        texts = [review["text"] for review in reviews]
        results = self.analyze_batch(texts)

        # 統計計算
        sentiments = [r["sentiment"] for r in results]
        sentiment_counts = Counter(sentiments)

        total = len(sentiments)
        stats = {
            "total_reviews": total,
            "positive_count": sentiment_counts.get("POSITIVE", 0),
            "negative_count": sentiment_counts.get("NEGATIVE", 0),
            "neutral_count": sentiment_counts.get("NEUTRAL", 0),
            "positive_ratio": sentiment_counts.get("POSITIVE", 0) / total * 100,
            "negative_ratio": sentiment_counts.get("NEGATIVE", 0) / total * 100,
            "average_confidence": np.mean([r["confidence"] for r in results])
        }

        # トップポジティブ/ネガティブレビュー抽出
        positive_reviews = sorted(
            [r for r in results if r["sentiment"] == "POSITIVE"],
            key=lambda x: x["confidence"],
            reverse=True
        )[:5]

        negative_reviews = sorted(
            [r for r in results if r["sentiment"] == "NEGATIVE"],
            key=lambda x: x["confidence"],
            reverse=True
        )[:5]

        return {
            "statistics": stats,
            "top_positive": positive_reviews,
            "top_negative": negative_reviews,
            "all_results": results
        }

    def extract_topics(
        self,
        texts: List[str],
        sentiment_filter: Optional[str] = None
    ) -> List[str]:
        """
        頻出トピック（キーワード）を抽出.

        Args:
            texts: テキストのリスト
            sentiment_filter: 感情フィルタ（POSITIVE/NEGATIVE/None）

        Returns:
            頻出キーワードのリスト
        """
        # 感情分析でフィルタリング
        if sentiment_filter:
            results = self.analyze_batch(texts)
            texts = [
                r["text"] for r in results
                if r["sentiment"] == sentiment_filter
            ]

        # 簡易的なキーワード抽出（実際はspaCyやBERTを使うとより良い）
        from collections import Counter
        import re

        # ストップワード（簡易版）
        stop_words = set([
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "this", "that", "these", "those", "i", "you", "he", "she",
            "it", "we", "they", "my", "your", "his", "her", "its", "our", "their"
        ])

        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            all_words.extend([w for w in words if w not in stop_words])

        # 頻出順
        word_counts = Counter(all_words)
        top_keywords = [word for word, count in word_counts.most_common(20)]

        return top_keywords

    def generate_report(
        self,
        reviews: List[Dict],
        output_file: str = "sentiment_report.txt"
    ):
        """
        分析レポート生成.

        Args:
            reviews: レビューのリスト
            output_file: 出力ファイルパス
        """
        analysis = self.analyze_reviews(reviews)
        stats = analysis["statistics"]

        # レポート作成
        report = []
        report.append("=" * 60)
        report.append("SENTIMENT ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Reviews Analyzed: {stats['total_reviews']}")
        report.append(f"\nSentiment Distribution:")
        report.append(f"  Positive: {stats['positive_count']} ({stats['positive_ratio']:.1f}%)")
        report.append(f"  Negative: {stats['negative_count']} ({stats['negative_ratio']:.1f}%)")
        report.append(f"  Neutral:  {stats['neutral_count']}")
        report.append(f"\nAverage Confidence: {stats['average_confidence']:.2%}")

        # トップポジティブレビュー
        report.append("\n" + "-" * 60)
        report.append("TOP POSITIVE REVIEWS:")
        report.append("-" * 60)
        for i, review in enumerate(analysis["top_positive"], 1):
            report.append(f"\n{i}. [{review['confidence']:.2%} confident]")
            report.append(f"   {review['text'][:200]}...")

        # トップネガティブレビュー
        report.append("\n" + "-" * 60)
        report.append("TOP NEGATIVE REVIEWS (Areas for Improvement):")
        report.append("-" * 60)
        for i, review in enumerate(analysis["top_negative"], 1):
            report.append(f"\n{i}. [{review['confidence']:.2%} confident]")
            report.append(f"   {review['text'][:200]}...")

        # トピック分析
        texts = [review["text"] for review in reviews]
        positive_topics = self.extract_topics(texts, sentiment_filter="POSITIVE")
        negative_topics = self.extract_topics(texts, sentiment_filter="NEGATIVE")

        report.append("\n" + "-" * 60)
        report.append("TOPIC ANALYSIS:")
        report.append("-" * 60)
        report.append(f"\nMost Mentioned (Positive Reviews):")
        report.append(f"  {', '.join(positive_topics[:10])}")
        report.append(f"\nMost Mentioned (Negative Reviews):")
        report.append(f"  {', '.join(negative_topics[:10])}")

        report.append("\n" + "=" * 60)

        # ファイル保存
        report_text = "\n".join(report)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"\nReport saved to {output_file}")
        print(report_text)

    def fine_tune(
        self,
        train_data: List[Dict],
        output_dir: str = "fine_tuned_sentiment",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        """
        ゲーム固有データでファインチューニング.

        Args:
            train_data: [{"text": "...", "label": 0/1}, ...]
            output_dir: 保存ディレクトリ
            epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
        """
        print(f"Fine-tuning on {len(train_data)} examples")

        # データセット準備
        texts = [item["text"] for item in train_data]
        labels = [item["label"] for item in train_data]

        dataset = Dataset.from_dict({"text": texts, "label": labels})

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            fp16=torch.cuda.is_available()
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset
        )

        # Train
        trainer.train()

        # Save
        trainer.save_model(output_dir + "/final")
        self.tokenizer.save_pretrained(output_dir + "/final")

        print(f"Model saved to {output_dir}/final")


def load_reviews_json(file_path: str) -> List[Dict]:
    """
    レビューJSONファイル読み込み.

    フォーマット例:
    [
        {"text": "Great game!", "rating": 5},
        {"text": "Terrible bugs", "rating": 1},
        ...
    ]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analyzer")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--analyze", type=str, help="JSON file with reviews")
    parser.add_argument("--train", type=str, help="Training data JSON")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased-finetuned-sst-2-english",
                        help="Model name")
    parser.add_argument("--output", type=str, default="sentiment_report.txt",
                        help="Output file")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    analyzer = SentimentAnalyzer(model_name=args.model)

    # 単一テキスト分析
    if args.text:
        result = analyzer.analyze(args.text)

        print("\nSentiment Analysis:")
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Scores: Positive={result['scores']['positive']:.2%}, "
              f"Negative={result['scores']['negative']:.2%}")

    # レビュー分析
    elif args.analyze:
        print(f"Loading reviews from {args.analyze}")
        reviews = load_reviews_json(args.analyze)

        analyzer.generate_report(reviews, output_file=args.output)

    # ファインチューニング
    elif args.train:
        print(f"Loading training data from {args.train}")
        train_data = load_reviews_json(args.train)

        analyzer.fine_tune(
            train_data=train_data,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
