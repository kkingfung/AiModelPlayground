# Computer Vision Use Cases for Game Development

**Practical applications of computer vision in your game pipeline**

---

## 📋 Table of Contents

1. [Auto-Tagging Game Assets](#auto-tagging-game-assets)
2. [UI Element Detection & Testing](#ui-element-detection--testing)
3. [Screenshot Analysis](#screenshot-analysis)
4. [Asset Quality Control](#asset-quality-control)
5. [Player Behavior Analysis](#player-behavior-analysis)
6. [Style Transfer & Effects](#style-transfer--effects)

---

## Auto-Tagging Game Assets

### Problem
You have thousands of game assets (characters, UI elements, items, backgrounds) that need to be:
- Categorized into types
- Tagged with metadata
- Organized in asset management system
- Searchable by team members

**Manual tagging**: 10-30 seconds per asset
**AI tagging**: <1 second per asset

### Solution: Image Classification

**Time Saved**: 80-95%
**Accuracy**: 90-98% (with fine-tuning)

### Implementation

#### Step 1: Prepare Training Data

Organize assets into folders by category:

```
data/game_assets/
├── character/
│   ├── player_idle.png
│   ├── enemy_orc.png
│   └── npc_vendor.png
├── ui/
│   ├── button_play.png
│   ├── health_bar.png
│   └── inventory_panel.png
├── item/
│   ├── sword_iron.png
│   ├── potion_health.png
│   └── armor_leather.png
├── background/
│   ├── forest_scene.png
│   └── dungeon_room.png
└── vfx/
    ├── explosion.png
    └── magic_particle.png
```

**Minimum images per category**: 20-50
**Recommended**: 100-500 per category

#### Step 2: Train Classifier

```bash
cd experiments/07_practical_game_ai/1_computer_vision

# Train on your data
python image_classifier.py \
    --train data/game_assets \
    --val-split 0.2 \
    --epochs 20 \
    --model-name mobilenet_v3_small \
    --save asset_classifier.pth
```

**Training time**: 10-30 minutes (depending on dataset size)

#### Step 3: Batch Prediction

```python
from image_classifier import GameAssetClassifier
from pathlib import Path
import json

# Load trained model
classifier = GameAssetClassifier()
classifier.load("checkpoints/asset_classifier.pth")

# Batch process new assets
new_assets_dir = Path("imports/new_batch")
results = {}

for asset_path in new_assets_dir.glob("*.png"):
    prediction = classifier.predict(str(asset_path))

    results[asset_path.name] = {
        "category": prediction["top_class"],
        "confidence": prediction["confidence"],
        "all_predictions": prediction["top_k"]
    }

    # Auto-organize
    category = prediction["top_class"]
    dest = Path(f"organized/{category}/{asset_path.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    asset_path.rename(dest)

    print(f"✓ {asset_path.name} → {category} ({prediction['confidence']:.1f}%)")

# Save metadata
with open("asset_metadata.json", "w") as f:
    json.dump(results, f, indent=2)
```

**Output**:
```
✓ new_character_01.png → character (96.2%)
✓ button_settings.png → ui (99.1%)
✓ potion_mana.png → item (94.8%)
✓ castle_exterior.png → background (97.5%)
```

### Advanced: Multi-Label Classification

Some assets belong to multiple categories (e.g., "character + UI" for character portraits)

```python
# Modify classifier for multi-label
from torch import nn

class MultiLabelClassifier(GameAssetClassifier):
    def __init__(self, num_classes, threshold=0.5):
        super().__init__(num_classes)
        self.threshold = threshold

    def predict_multi_label(self, image_path):
        probs = self.predict_proba(image_path)
        labels = [self.classes[i] for i, p in enumerate(probs) if p > self.threshold]
        return labels

# Usage
classifier = MultiLabelClassifier(num_classes=5, threshold=0.3)
labels = classifier.predict_multi_label("character_portrait.png")
# Output: ['character', 'ui']
```

### Integration with Asset Management

#### Unity Integration

```csharp
// C# Unity script
using UnityEngine;
using UnityEditor;
using System.Diagnostics;

public class AIAssetTagger : EditorWindow
{
    [MenuItem("Tools/AI Asset Tagger")]
    static void ShowWindow()
    {
        GetWindow<AIAssetTagger>("AI Tagger");
    }

    void OnGUI()
    {
        if (GUILayout.Button("Tag Selected Assets"))
        {
            TagSelectedAssets();
        }
    }

    void TagSelectedAssets()
    {
        foreach (var obj in Selection.objects)
        {
            string assetPath = AssetDatabase.GetAssetPath(obj);

            // Call Python classifier
            Process process = new Process();
            process.StartInfo.FileName = "python";
            process.StartInfo.Arguments = $"classifier.py --predict {assetPath}";
            process.StartInfo.RedirectStandardOutput = true;
            process.Start();

            string result = process.StandardOutput.ReadToEnd();
            process.WaitForExit();

            // Parse result and add label
            string category = ParseCategory(result);
            AssetDatabase.SetLabels(obj, new[] { category });

            Debug.Log($"Tagged {obj.name} as {category}");
        }
    }
}
```

---

## UI Element Detection & Testing

### Problem
You need to:
- Verify UI elements appear correctly
- Detect missing/broken UI components
- Test UI on different resolutions
- Automate UI regression testing

### Solution: Object Detection (YOLOv8)

### Implementation

#### Step 1: Prepare UI Detection Dataset

Create annotations in YOLO format:

```
data/ui_detection/
├── images/
│   ├── train/
│   │   ├── screenshot_001.png
│   │   └── screenshot_002.png
│   └── val/
│       └── screenshot_test.png
└── labels/
    ├── train/
    │   ├── screenshot_001.txt
    │   └── screenshot_002.txt
    └── val/
        └── screenshot_test.txt
```

**Label format** (screenshot_001.txt):
```
0 0.5 0.1 0.2 0.05    # button: center_x, center_y, width, height
1 0.8 0.05 0.15 0.1   # health_bar
2 0.1 0.9 0.3 0.08    # inventory_slot
```

Classes (data.yaml):
```yaml
names:
  - button
  - health_bar
  - inventory_slot
  - minimap
  - chat_window
  - skill_icon
```

#### Step 2: Train Detector

```bash
cd experiments/07_practical_game_ai/1_computer_vision

# Train YOLOv8
python ui_detector.py \
    --train data/ui_detection/data.yaml \
    --epochs 50 \
    --img-size 1920
```

#### Step 3: Automated UI Testing

```python
from ui_detector import UIElementDetector

detector = UIElementDetector(model_path="checkpoints/ui_detector.pt")

# Test screenshot
screenshot = "test/gameplay_screenshot.png"
detections = detector.detect(screenshot)

# Verify required elements
required_elements = ["health_bar", "minimap", "skill_icon"]
detected_classes = [d["class"] for d in detections["detections"]]

missing = set(required_elements) - set(detected_classes)

if missing:
    print(f"❌ FAIL: Missing UI elements: {missing}")
else:
    print("✅ PASS: All UI elements present")

# Check positioning
for detection in detections["detections"]:
    if detection["class"] == "health_bar":
        # Verify health bar is in top-left quadrant
        if detection["bbox"]["center_x"] > 0.5 or detection["bbox"]["center_y"] > 0.5:
            print("⚠️ WARNING: Health bar not in expected position")
```

### CI/CD Integration

```yaml
# .github/workflows/ui-test.yml
name: UI Regression Test

on: [push]

jobs:
  ui-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run game and capture screenshot
        run: |
          python scripts/launch_game.py --screenshot test_build/

      - name: Test UI elements
        run: |
          python tests/test_ui_detection.py test_build/screenshot.png

      - name: Upload artifacts
        if: failure()
        uses: actions/upload-artifact@v2
        with:
          name: failed-screenshots
          path: test_build/
```

---

## Screenshot Analysis

### Problem
Analyze player screenshots to:
- Detect bugs (visual glitches, missing elements)
- Understand player behavior (what they screenshot)
- Gather gameplay metrics
- Identify popular moments

### Solution: Multi-Model Pipeline

```python
from image_classifier import GameAssetClassifier
from ui_detector import UIElementDetector
import torch

class ScreenshotAnalyzer:
    def __init__(self):
        self.scene_classifier = GameAssetClassifier()  # Classify scene type
        self.ui_detector = UIElementDetector()  # Detect UI elements

    def analyze(self, screenshot_path):
        results = {}

        # 1. Classify scene
        scene = self.scene_classifier.predict(screenshot_path)
        results["scene_type"] = scene["top_class"]
        results["scene_confidence"] = scene["confidence"]

        # 2. Detect UI elements
        ui = self.ui_detector.detect(screenshot_path)
        results["ui_elements"] = ui["detections"]
        results["ui_coverage"] = ui["statistics"]["total_area_ratio"]

        # 3. Detect anomalies
        results["anomalies"] = self.detect_anomalies(ui)

        return results

    def detect_anomalies(self, ui_detections):
        anomalies = []

        # Check for overlapping UI
        for i, det1 in enumerate(ui_detections["detections"]):
            for det2 in ui_detections["detections"][i+1:]:
                if self.boxes_overlap(det1["bbox"], det2["bbox"]):
                    anomalies.append({
                        "type": "overlapping_ui",
                        "elements": [det1["class"], det2["class"]]
                    })

        # Check for missing critical UI
        required = ["health_bar", "minimap"]
        detected_classes = [d["class"] for d in ui_detections["detections"]]
        for req in required:
            if req not in detected_classes:
                anomalies.append({
                    "type": "missing_ui",
                    "element": req
                })

        return anomalies

# Usage
analyzer = ScreenshotAnalyzer()

# Analyze player-submitted screenshot
result = analyzer.analyze("player_screenshots/bug_report_123.png")

if result["anomalies"]:
    print(f"🐛 Found {len(result['anomalies'])} potential issues:")
    for anomaly in result["anomalies"]:
        print(f"  - {anomaly['type']}: {anomaly}")
```

---

## Asset Quality Control

### Problem
Ensure assets meet quality standards:
- Correct resolution
- Proper compression
- No visual artifacts
- Consistent style

### Solution: Automated Quality Checks

```python
from PIL import Image
import numpy as np

class AssetQualityChecker:
    def __init__(self, standards):
        self.standards = standards  # Quality standards dict

    def check(self, asset_path):
        issues = []

        img = Image.open(asset_path)

        # 1. Resolution check
        if img.size != self.standards["resolution"]:
            issues.append({
                "severity": "error",
                "type": "resolution_mismatch",
                "expected": self.standards["resolution"],
                "actual": img.size
            })

        # 2. File size check
        file_size_mb = os.path.getsize(asset_path) / (1024 * 1024)
        if file_size_mb > self.standards["max_size_mb"]:
            issues.append({
                "severity": "warning",
                "type": "file_too_large",
                "size_mb": file_size_mb
            })

        # 3. Detect compression artifacts
        artifact_score = self.detect_compression_artifacts(img)
        if artifact_score > self.standards["max_artifact_score"]:
            issues.append({
                "severity": "warning",
                "type": "compression_artifacts",
                "score": artifact_score
            })

        # 4. Check alpha channel (if PNG)
        if img.mode == "RGBA":
            has_transparency = np.any(np.array(img)[:,:,3] < 255)
            if not has_transparency and self.standards["require_transparency"]:
                issues.append({
                    "severity": "info",
                    "type": "no_transparency"
                })

        return {
            "passed": len([i for i in issues if i["severity"] == "error"]) == 0,
            "issues": issues
        }

    def detect_compression_artifacts(self, img):
        # Simple JPEG artifact detection using high-frequency analysis
        arr = np.array(img.convert("L"))

        # Compute gradient
        gx = np.diff(arr, axis=1)
        gy = np.diff(arr, axis=0)

        # High-frequency content
        hf_score = np.mean(np.abs(gx)) + np.mean(np.abs(gy))

        # Blocking artifacts (8x8 DCT blocks in JPEG)
        block_score = self.detect_blocking(arr)

        return hf_score * 0.7 + block_score * 0.3

# Usage
standards = {
    "resolution": (1024, 1024),
    "max_size_mb": 5.0,
    "max_artifact_score": 0.8,
    "require_transparency": False
}

checker = AssetQualityChecker(standards)

# Check all new assets
for asset in Path("imports/").glob("*.png"):
    result = checker.check(asset)

    if not result["passed"]:
        print(f"❌ {asset.name}:")
        for issue in result["issues"]:
            print(f"  [{issue['severity']}] {issue['type']}")
    else:
        print(f"✅ {asset.name}")
```

---

## Performance Benchmarks

### Image Classification

| Model | Size | Inference Time | Accuracy |
|-------|------|----------------|----------|
| MobileNetV3-Small | 5 MB | 10ms (CPU) | 92% |
| MobileNetV3-Large | 20 MB | 25ms (CPU) | 95% |
| EfficientNet-B0 | 20 MB | 30ms (CPU) | 96% |
| ResNet50 | 100 MB | 80ms (CPU) | 97% |

### Object Detection

| Model | Size | Inference Time | mAP |
|-------|------|----------------|-----|
| YOLOv8n | 6 MB | 15ms (GPU) | 0.85 |
| YOLOv8s | 22 MB | 25ms (GPU) | 0.90 |
| YOLOv8m | 52 MB | 45ms (GPU) | 0.92 |

---

## Next Steps

- **[NLP Use Cases](nlp.md)** - Text generation and analysis
- **[RL Use Cases](reinforcement-learning.md)** - Intelligent agents
- **[Unity Integration](../integration/unity.md)** - Deploy to Unity
- **[Performance Optimization](../best-practices/performance.md)** - Speed up models

---

**Transform your asset pipeline with AI! 🎨🤖**
