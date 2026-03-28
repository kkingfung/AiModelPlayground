"""
ONNX Model Export & Optimization

PyTorchモデルをONNX形式に変換して高速化・デプロイ最適化.
Unity/Unrealなどのゲームエンジンでも使用可能.

使い方:
    # モデルをONNXに変換
    python onnx_export.py \
        --model checkpoints/model.pth \
        --output model.onnx \
        --input-shape 1,3,224,224

    # ONNX最適化
    python onnx_export.py \
        --model model.onnx \
        --optimize \
        --output model_optimized.onnx

    # ベンチマーク比較
    python onnx_export.py \
        --benchmark model.pth model.onnx \
        --input-shape 1,3,224,224
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from onnxsim import simplify


class ONNXExporter:
    """
    PyTorchモデルをONNX形式にエクスポート.

    ONNX形式にすることで:
    - 推論速度が2-5x向上
    - ゲームエンジン（Unity/Unreal）での使用が可能
    - 異なるフレームワーク間での互換性
    """

    def __init__(self):
        pass

    def export(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        opset_version: int = 14,
        dynamic_axes: Optional[dict] = None,
        simplify_model: bool = True
    ):
        """
        PyTorchモデルをONNXにエクスポート.

        Args:
            model: PyTorchモデル
            output_path: 出力ONNXファイルパス
            input_shape: 入力テンソルshape
            opset_version: ONNXオペレータセットバージョン
            dynamic_axes: 動的な次元の指定
            simplify_model: ONNX簡略化を実行
        """
        model.eval()

        # ダミー入力作成
        dummy_input = torch.randn(*input_shape)

        # デフォルト動的軸（バッチサイズ）
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }

        print(f"Exporting to ONNX: {output_path}")
        print(f"Input shape: {input_shape}")
        print(f"Opset version: {opset_version}")

        # ONNXエクスポート
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )

        print(f"✓ Exported to {output_path}")

        # ONNX検証
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model validated")

        # 簡略化
        if simplify_model:
            print("Simplifying ONNX model...")
            onnx_model_simplified, check = simplify(onnx_model)

            if check:
                onnx.save(onnx_model_simplified, output_path)
                print("✓ Model simplified")
            else:
                print("⚠ Simplification failed, using original model")

        # モデル情報表示
        self._print_model_info(output_path)

    def optimize(
        self,
        onnx_path: str,
        output_path: Optional[str] = None,
        optimization_level: str = "all"
    ):
        """
        ONNX モデルを最適化.

        Args:
            onnx_path: 入力ONNXモデルパス
            output_path: 出力パス（Noneの場合は上書き）
            optimization_level: 最適化レベル（basic/extended/all）
        """
        if output_path is None:
            output_path = onnx_path.replace(".onnx", "_optimized.onnx")

        print(f"Optimizing ONNX model: {onnx_path}")

        # SessionOptionsで最適化
        sess_options = ort.SessionOptions()

        if optimization_level == "basic":
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif optimization_level == "extended":
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        else:  # all
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        sess_options.optimized_model_filepath = output_path

        # 最適化実行（セッション作成時に自動）
        _ = ort.InferenceSession(onnx_path, sess_options)

        print(f"✓ Optimized model saved to {output_path}")

        # サイズ比較
        original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        optimized_size = Path(output_path).stat().st_size / (1024 * 1024)

        print(f"Original size: {original_size:.2f} MB")
        print(f"Optimized size: {optimized_size:.2f} MB")
        print(f"Reduction: {(1 - optimized_size/original_size) * 100:.1f}%")

    def benchmark(
        self,
        pytorch_model: Optional[nn.Module] = None,
        onnx_path: Optional[str] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        num_iterations: int = 100,
        warmup: int = 10
    ) -> dict:
        """
        PyTorchとONNXの推論速度を比較.

        Args:
            pytorch_model: PyTorchモデル
            onnx_path: ONNXモデルパス
            input_shape: 入力shape
            num_iterations: ベンチマーク回数
            warmup: ウォームアップ回数

        Returns:
            ベンチマーク結果
        """
        print("\n" + "=" * 60)
        print("BENCHMARK: PyTorch vs ONNX")
        print("=" * 60)

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        results = {}

        # PyTorchベンチマーク
        if pytorch_model is not None:
            pytorch_model.eval()
            torch_input = torch.from_numpy(dummy_input)

            # Warmup
            for _ in range(warmup):
                with torch.no_grad():
                    _ = pytorch_model(torch_input)

            # Benchmark
            start = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = pytorch_model(torch_input)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            pytorch_time = (time.time() - start) / num_iterations * 1000  # ms

            results["pytorch"] = {
                "avg_time_ms": pytorch_time,
                "fps": 1000 / pytorch_time
            }

            print(f"\nPyTorch:")
            print(f"  Average time: {pytorch_time:.2f} ms")
            print(f"  FPS: {1000/pytorch_time:.1f}")

        # ONNXベンチマーク
        if onnx_path is not None:
            # CPU実行
            ort_session_cpu = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"]
            )

            # Warmup
            for _ in range(warmup):
                _ = ort_session_cpu.run(None, {"input": dummy_input})

            # Benchmark
            start = time.time()
            for _ in range(num_iterations):
                _ = ort_session_cpu.run(None, {"input": dummy_input})

            onnx_cpu_time = (time.time() - start) / num_iterations * 1000  # ms

            results["onnx_cpu"] = {
                "avg_time_ms": onnx_cpu_time,
                "fps": 1000 / onnx_cpu_time
            }

            print(f"\nONNX (CPU):")
            print(f"  Average time: {onnx_cpu_time:.2f} ms")
            print(f"  FPS: {1000/onnx_cpu_time:.1f}")

            # GPU実行（利用可能な場合）
            if "CUDAExecutionProvider" in ort.get_available_providers():
                ort_session_gpu = ort.InferenceSession(
                    onnx_path,
                    providers=["CUDAExecutionProvider"]
                )

                # Warmup
                for _ in range(warmup):
                    _ = ort_session_gpu.run(None, {"input": dummy_input})

                # Benchmark
                start = time.time()
                for _ in range(num_iterations):
                    _ = ort_session_gpu.run(None, {"input": dummy_input})

                onnx_gpu_time = (time.time() - start) / num_iterations * 1000

                results["onnx_gpu"] = {
                    "avg_time_ms": onnx_gpu_time,
                    "fps": 1000 / onnx_gpu_time
                }

                print(f"\nONNX (GPU):")
                print(f"  Average time: {onnx_gpu_time:.2f} ms")
                print(f"  FPS: {1000/onnx_gpu_time:.1f}")

        # 比較
        if "pytorch" in results and "onnx_cpu" in results:
            speedup = results["pytorch"]["avg_time_ms"] / results["onnx_cpu"]["avg_time_ms"]
            print(f"\n{'=' * 60}")
            print(f"ONNX Speedup (CPU): {speedup:.2f}x")
            print(f"{'=' * 60}\n")

        return results

    def _print_model_info(self, onnx_path: str):
        """ONNX モデル情報を表示."""
        model = onnx.load(onnx_path)

        print("\nModel Information:")
        print(f"  IR Version: {model.ir_version}")
        print(f"  Producer: {model.producer_name} {model.producer_version}")

        # 入力情報
        print("\n  Inputs:")
        for inp in model.graph.input:
            shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in inp.type.tensor_type.shape.dim]
            print(f"    {inp.name}: {shape}")

        # 出力情報
        print("\n  Outputs:")
        for out in model.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else "dynamic" for d in out.type.tensor_type.shape.dim]
            print(f"    {out.name}: {shape}")

        # ファイルサイズ
        size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
        print(f"\n  File size: {size_mb:.2f} MB")


class ONNXInference:
    """
    ONNX モデルで高速推論.

    ゲームエンジンやプロダクション環境での推論に使用.
    """

    def __init__(
        self,
        onnx_path: str,
        use_gpu: bool = False
    ):
        """
        Args:
            onnx_path: ONNXモデルパス
            use_gpu: GPU使用（CUDAExecutionProvider）
        """
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # 入出力名取得
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"Loaded ONNX model: {onnx_path}")
        print(f"Provider: {self.session.get_providers()[0]}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        推論実行.

        Args:
            input_data: 入力データ（numpy array）

        Returns:
            推論結果
        """
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        return outputs[0]


def main():
    parser = argparse.ArgumentParser(description="ONNX Export & Optimization")
    parser.add_argument("--model", type=str, help="PyTorch model path (.pth)")
    parser.add_argument("--onnx", type=str, help="ONNX model path for optimization/benchmark")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX path")
    parser.add_argument("--input-shape", type=int, nargs="+", default=[1, 3, 224, 224],
                        help="Input tensor shape")
    parser.add_argument("--optimize", action="store_true", help="Optimize ONNX model")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark PyTorch vs ONNX")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")

    args = parser.parse_args()

    exporter = ONNXExporter()

    # Export
    if args.model and not args.optimize and not args.benchmark:
        # Load PyTorch model
        model = torch.load(args.model, map_location="cpu")

        if isinstance(model, dict):
            # Checkpoint format
            print("⚠ Model is a checkpoint dict. Please provide model architecture.")
            return

        exporter.export(
            model=model,
            output_path=args.output,
            input_shape=tuple(args.input_shape),
            opset_version=args.opset,
            simplify_model=args.simplify
        )

    # Optimize
    elif args.optimize and args.onnx:
        exporter.optimize(
            onnx_path=args.onnx,
            output_path=args.output,
            optimization_level="all"
        )

    # Benchmark
    elif args.benchmark:
        pytorch_model = None
        if args.model:
            pytorch_model = torch.load(args.model, map_location="cpu")

        exporter.benchmark(
            pytorch_model=pytorch_model,
            onnx_path=args.onnx,
            input_shape=tuple(args.input_shape)
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
