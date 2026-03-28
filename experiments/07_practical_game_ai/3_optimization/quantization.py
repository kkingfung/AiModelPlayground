"""
Model Quantization

モデルを量子化して軽量化・高速化.
FP32 → FP16/INT8で50-75%のサイズ削減、2-10x高速化.

使い方:
    # FP16量子化
    python quantization.py \
        --model checkpoints/model.pth \
        --quantize fp16 \
        --output model_fp16.pth

    # INT8量子化（Post-Training Quantization）
    python quantization.py \
        --model checkpoints/model.pth \
        --quantize int8 \
        --calibration-data data/calibration \
        --output model_int8.pth

    # ベンチマーク
    python quantization.py \
        --benchmark model.pth model_fp16.pth model_int8.pth
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.ao.quantization import get_default_qconfig_mapping, QConfigMapping
import time
import numpy as np


class ModelQuantizer:
    """
    モデル量子化ツール.

    FP32 → FP16/INT8変換でモバイル・組み込み向けに最適化.
    """

    def __init__(self):
        pass

    def quantize_fp16(
        self,
        model: nn.Module,
        output_path: str
    ):
        """
        FP16（半精度）量子化.

        - サイズ: 50%削減
        - 速度: 1.5-2x高速化（GPU）
        - 精度: ほぼ同等

        Args:
            model: PyTorchモデル
            output_path: 出力パス
        """
        print("Quantizing to FP16...")

        model_fp16 = model.half()  # FP32 → FP16

        # 保存
        torch.save(model_fp16.state_dict(), output_path)

        print(f"✓ FP16 model saved to {output_path}")

        # サイズ比較
        self._print_size_comparison(model, model_fp16)

    def quantize_int8_dynamic(
        self,
        model: nn.Module,
        output_path: str
    ):
        """
        INT8動的量子化.

        - サイズ: 75%削減
        - 速度: 2-4x高速化（CPU）
        - 精度: 1-2%低下

        重み（weights）のみINT8化、活性化（activations）は実行時に量子化.
        最も簡単で、キャリブレーションデータ不要.

        Args:
            model: PyTorchモデル
            output_path: 出力パス
        """
        print("Quantizing to INT8 (Dynamic)...")

        model.eval()

        # 動的量子化
        quantized_model = quant.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # 量子化する層
            dtype=torch.qint8
        )

        # 保存
        torch.save(quantized_model.state_dict(), output_path)

        print(f"✓ INT8 (Dynamic) model saved to {output_path}")

        # サイズ比較
        self._print_size_comparison(model, quantized_model)

    def quantize_int8_static(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        output_path: str
    ):
        """
        INT8静的量子化（Post-Training Quantization）.

        - サイズ: 75%削減
        - 速度: 3-10x高速化（CPU）
        - 精度: 0.5-1%低下

        重みと活性化の両方をINT8化.
        最高速だが、キャリブレーションデータが必要.

        Args:
            model: PyTorchモデル
            calibration_loader: キャリブレーション用DataLoader
            output_path: 出力パス
        """
        print("Quantizing to INT8 (Static)...")

        model.eval()

        # QConfig設定
        model.qconfig = torch.quantization.get_default_qconfig('x86')

        # Fuse layers (Conv + BN + ReLU など)
        model_fused = torch.quantization.fuse_modules(
            model,
            [['conv', 'bn', 'relu']]  # モデル構造に応じて調整
        )

        # Prepare for quantization
        model_prepared = torch.quantization.prepare(model_fused)

        # Calibration（キャリブレーション）
        print("Running calibration...")
        with torch.no_grad():
            for inputs, _ in calibration_loader:
                model_prepared(inputs)

        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)

        # 保存
        torch.save(quantized_model.state_dict(), output_path)

        print(f"✓ INT8 (Static) model saved to {output_path}")

        # サイズ比較
        self._print_size_comparison(model, quantized_model)

    def quantize_qat(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 3,
        lr: float = 0.0001,
        output_path: str = "model_qat.pth"
    ):
        """
        Quantization-Aware Training (QAT).

        - 精度: 最高（ほぼFP32と同等）
        - 速度: INT8と同等
        - 手間: 学習が必要

        量子化を考慮して再学習することで、精度低下を最小化.

        Args:
            model: PyTorchモデル
            train_loader: 学習用DataLoader
            epochs: エポック数
            lr: 学習率
            output_path: 出力パス
        """
        print("Quantization-Aware Training...")

        model.train()

        # QConfig設定
        model.qconfig = torch.quantization.get_default_qat_qconfig('x86')

        # Prepare for QAT
        model_prepared = torch.quantization.prepare_qat(model)

        # 学習
        optimizer = torch.optim.Adam(model_prepared.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model_prepared(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Convert to quantized model
        model_prepared.eval()
        quantized_model = torch.quantization.convert(model_prepared)

        # 保存
        torch.save(quantized_model.state_dict(), output_path)

        print(f"✓ QAT model saved to {output_path}")

    def benchmark(
        self,
        models: dict,
        input_shape: tuple = (1, 3, 224, 224),
        num_iterations: int = 100
    ):
        """
        複数モデルのベンチマーク比較.

        Args:
            models: {"name": model, ...}
            input_shape: 入力shape
            num_iterations: 反復回数
        """
        print("\n" + "=" * 60)
        print("QUANTIZATION BENCHMARK")
        print("=" * 60)

        dummy_input = torch.randn(*input_shape)

        results = {}

        for name, model in models.items():
            model.eval()

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(dummy_input)

            # Benchmark
            start = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = model(dummy_input)

            avg_time = (time.time() - start) / num_iterations * 1000  # ms

            # モデルサイズ
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            size_mb = param_size / (1024 * 1024)

            results[name] = {
                "time_ms": avg_time,
                "fps": 1000 / avg_time,
                "size_mb": size_mb
            }

            print(f"\n{name}:")
            print(f"  Time: {avg_time:.2f} ms")
            print(f"  FPS: {1000/avg_time:.1f}")
            print(f"  Size: {size_mb:.2f} MB")

        # 比較表示
        if "FP32" in results:
            baseline = results["FP32"]

            print(f"\n{'=' * 60}")
            print("COMPARISON (vs FP32):")
            print(f"{'=' * 60}")

            for name, result in results.items():
                if name == "FP32":
                    continue

                speedup = baseline["time_ms"] / result["time_ms"]
                size_reduction = (1 - result["size_mb"] / baseline["size_mb"]) * 100

                print(f"\n{name}:")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Size reduction: {size_reduction:.1f}%")

        print(f"\n{'=' * 60}\n")

        return results

    def _print_size_comparison(self, model_original: nn.Module, model_quantized: nn.Module):
        """モデルサイズ比較を表示."""

        # 元のモデルサイズ
        original_size = sum(p.numel() * p.element_size() for p in model_original.parameters())
        original_mb = original_size / (1024 * 1024)

        # 量子化後サイズ
        quantized_size = sum(p.numel() * p.element_size() for p in model_quantized.parameters())
        quantized_mb = quantized_size / (1024 * 1024)

        reduction = (1 - quantized_size / original_size) * 100

        print(f"\nSize Comparison:")
        print(f"  Original: {original_mb:.2f} MB")
        print(f"  Quantized: {quantized_mb:.2f} MB")
        print(f"  Reduction: {reduction:.1f}%")


def create_sample_model() -> nn.Module:
    """サンプルモデル作成（デモ用）."""
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 10)
    )


def main():
    parser = argparse.ArgumentParser(description="Model Quantization")
    parser.add_argument("--model", type=str, help="PyTorch model path")
    parser.add_argument("--quantize", type=str, choices=["fp16", "int8", "int8-static", "qat"],
                        help="Quantization type")
    parser.add_argument("--calibration-data", type=str, help="Calibration data directory")
    parser.add_argument("--output", type=str, default="model_quantized.pth",
                        help="Output model path")
    parser.add_argument("--benchmark", nargs="+", help="Models to benchmark")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample model")

    args = parser.parse_args()

    quantizer = ModelQuantizer()

    # Demo
    if args.demo:
        print("Running quantization demo...")

        model = create_sample_model()

        # FP16
        quantizer.quantize_fp16(model, "demo_fp16.pth")

        # INT8 Dynamic
        quantizer.quantize_int8_dynamic(model, "demo_int8.pth")

        # Benchmark
        model_fp16 = create_sample_model()
        model_fp16.load_state_dict(torch.load("demo_fp16.pth"))

        model_int8 = create_sample_model()
        # INT8モデルのロードは実装に応じて調整

        quantizer.benchmark({
            "FP32": model,
            "FP16": model_fp16
        })

        return

    # Quantize
    if args.model and args.quantize:
        model = torch.load(args.model, map_location="cpu")

        if args.quantize == "fp16":
            quantizer.quantize_fp16(model, args.output)

        elif args.quantize == "int8":
            quantizer.quantize_int8_dynamic(model, args.output)

        elif args.quantize == "int8-static":
            if not args.calibration_data:
                print("❌ --calibration-data required for static quantization")
                return

            # キャリブレーションデータローダー作成
            # TODO: データセットに応じて実装
            print("⚠ Calibration data loading not implemented. Use demo mode.")

        elif args.quantize == "qat":
            print("⚠ QAT requires training data. Use demo mode or implement custom training.")

    # Benchmark
    elif args.benchmark:
        models = {}
        for model_path in args.benchmark:
            name = Path(model_path).stem
            model = torch.load(model_path, map_location="cpu")
            models[name] = model

        quantizer.benchmark(models)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
