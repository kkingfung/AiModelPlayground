"""
UI Element Detector using YOLO

ゲームUIスクリーンショットから UI要素（ボタン、ヘルスバー、メニューなど）を自動検出.
YOLOv8を使用してリアルタイム検出を実現.

使い方:
    # 学習（カスタムデータセット）
    python ui_detector.py --train data/ui_dataset --epochs 50

    # 検出
    python ui_detector.py --detect screenshot.png --model runs/detect/train/weights/best.pt

    # 動画から検出
    python ui_detector.py --detect-video gameplay.mp4 --model best.pt
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


class UIElementDetector:
    """
    YOLOv8を使用したUI要素検出器.

    ゲームのスクリーンショットや動画から、
    ボタン、メニュー、ヘルスバーなどのUI要素を検出します.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Args:
            model_path: YOLOモデルパス（Noneの場合は事前学習済みモデル）
            conf_threshold: 信頼度閾値
            iou_threshold: NMS IoU閾値
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # モデルロード
        if model_path:
            self.model = YOLO(model_path)
        else:
            # 事前学習済みモデル（YOLOv8n）
            self.model = YOLO("yolov8n.pt")

        self.class_names: List[str] = []

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        name: str = "ui_detector"
    ) -> Dict:
        """
        カスタムデータセットで学習.

        Args:
            data_yaml: データセット設定YAMLファイル
            epochs: エポック数
            imgsz: 入力画像サイズ
            batch: バッチサイズ
            name: 実験名

        Returns:
            学習結果
        """
        print(f"Training UI detector on {data_yaml}")

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            patience=10,
            save=True,
            device=0 if torch.cuda.is_available() else "cpu",
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0
        )

        print(f"\nTraining completed!")
        print(f"Best model saved to: runs/detect/{name}/weights/best.pt")

        return results

    def detect(
        self,
        image_path: str,
        save: bool = True,
        show: bool = False,
        output_dir: str = "outputs"
    ) -> Dict:
        """
        画像からUI要素を検出.

        Args:
            image_path: 画像パス
            save: 結果を保存
            show: 結果を表示
            output_dir: 出力ディレクトリ

        Returns:
            検出結果
        """
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            save=save,
            show=show,
            project=output_dir,
            name="detect"
        )

        # 結果を辞書に変換
        detections = []
        for result in results:
            boxes = result.boxes

            for box in boxes:
                detection = {
                    "class": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    "bbox_normalized": box.xywhn[0].tolist(),  # [x_center, y_center, width, height]
                }
                detections.append(detection)

        return {
            "image": str(image_path),
            "detections": detections,
            "count": len(detections)
        }

    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = False
    ):
        """
        動画からUI要素を検出.

        Args:
            video_path: 動画パス
            output_path: 出力動画パス
            show: リアルタイム表示
        """
        cap = cv2.VideoCapture(video_path)

        # 動画プロパティ取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 出力動画設定
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # YOLO検出
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            # 結果を描画
            annotated_frame = results[0].plot()

            if output_path:
                out.write(annotated_frame)

            if show:
                cv2.imshow('UI Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

        print(f"\nVideo processing completed!")
        if output_path:
            print(f"Output saved to: {output_path}")

    def batch_detect(
        self,
        image_dir: str,
        output_dir: str = "outputs/batch"
    ) -> List[Dict]:
        """
        ディレクトリ内の全画像を一括検出.

        Args:
            image_dir: 画像ディレクトリ
            output_dir: 出力ディレクトリ

        Returns:
            全検出結果のリスト
        """
        image_paths = list(Path(image_dir).glob("*.png")) + \
                      list(Path(image_dir).glob("*.jpg")) + \
                      list(Path(image_dir).glob("*.jpeg"))

        all_results = []

        for img_path in image_paths:
            result = self.detect(str(img_path), save=True, output_dir=output_dir)
            all_results.append(result)

        return all_results

    def analyze_ui_layout(
        self,
        image_path: str
    ) -> Dict:
        """
        UI レイアウトを分析.

        検出されたUI要素の位置、サイズ、密度などを分析.

        Args:
            image_path: 画像パス

        Returns:
            レイアウト分析結果
        """
        result = self.detect(image_path, save=False, show=False)

        if result["count"] == 0:
            return {"message": "No UI elements detected"}

        detections = result["detections"]

        # 位置分析（画面の4象限に分類）
        quadrants = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}

        for det in detections:
            x_center, y_center, _, _ = det["bbox_normalized"]

            if x_center < 0.5 and y_center < 0.5:
                quadrants["top_left"] += 1
            elif x_center >= 0.5 and y_center < 0.5:
                quadrants["top_right"] += 1
            elif x_center < 0.5 and y_center >= 0.5:
                quadrants["bottom_left"] += 1
            else:
                quadrants["bottom_right"] += 1

        # サイズ分析
        sizes = [det["bbox_normalized"][2] * det["bbox_normalized"][3] for det in detections]
        avg_size = np.mean(sizes)
        max_size = np.max(sizes)
        min_size = np.min(sizes)

        # クラス分布
        class_counts = {}
        for det in detections:
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        return {
            "total_elements": result["count"],
            "quadrant_distribution": quadrants,
            "size_stats": {
                "average": float(avg_size),
                "max": float(max_size),
                "min": float(min_size)
            },
            "class_distribution": class_counts,
            "layout_density": result["count"] / (1.0)  # 正規化密度
        }

    def export_onnx(
        self,
        output_path: str = "ui_detector.onnx",
        imgsz: int = 640
    ):
        """
        モデルをONNX形式でエクスポート.

        Args:
            output_path: 出力パス
            imgsz: 入力画像サイズ
        """
        self.model.export(format="onnx", imgsz=imgsz, simplify=True)
        print(f"Model exported to ONNX: {output_path}")


def create_sample_dataset_yaml() -> str:
    """
    サンプルデータセット設定YAMLを作成.

    Returns:
        YAML ファイルパス
    """
    yaml_content = """# UI Element Detection Dataset

path: data/ui_dataset  # dataset root dir
train: images/train    # train images (relative to 'path')
val: images/val        # val images (relative to 'path')

# Classes
nc: 6  # number of classes
names:
  0: button
  1: health_bar
  2: menu
  3: icon
  4: text_field
  5: minimap
"""

    yaml_path = Path("data/ui_dataset.yaml")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Created sample dataset config: {yaml_path}")
    return str(yaml_path)


def main():
    parser = argparse.ArgumentParser(description="UI Element Detector")
    parser.add_argument("--train", type=str, help="Training data YAML path")
    parser.add_argument("--detect", type=str, help="Image path to detect")
    parser.add_argument("--detect-video", type=str, help="Video path to detect")
    parser.add_argument("--batch-detect", type=str, help="Directory of images to detect")
    parser.add_argument("--analyze", type=str, help="Image path to analyze layout")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--create-yaml", action="store_true", help="Create sample dataset YAML")

    args = parser.parse_args()

    # サンプルYAML作成
    if args.create_yaml:
        create_sample_dataset_yaml()
        return

    # 検出器初期化
    detector = UIElementDetector(
        model_path=args.model,
        conf_threshold=args.conf
    )

    # 学習
    if args.train:
        detector.train(
            data_yaml=args.train,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch
        )

    # 画像検出
    elif args.detect:
        result = detector.detect(args.detect, save=True, show=False)

        print(f"\nDetection Results:")
        print(f"Total elements detected: {result['count']}")

        for i, det in enumerate(result['detections'], 1):
            print(f"  {i}. {det['class_name']}: {det['confidence']:.2f}")

    # 動画検出
    elif args.detect_video:
        output_path = str(Path(args.detect_video).stem) + "_detected.mp4"
        detector.detect_video(args.detect_video, output_path=output_path)

    # バッチ検出
    elif args.batch_detect:
        results = detector.batch_detect(args.batch_detect)
        print(f"\nProcessed {len(results)} images")

    # レイアウト分析
    elif args.analyze:
        analysis = detector.analyze_ui_layout(args.analyze)

        print("\nUI Layout Analysis:")
        print(f"Total elements: {analysis.get('total_elements', 0)}")
        print(f"Quadrant distribution: {analysis.get('quadrant_distribution', {})}")
        print(f"Class distribution: {analysis.get('class_distribution', {})}")

    else:
        parser.print_help()


if __name__ == "__main__":
    import torch
    main()
