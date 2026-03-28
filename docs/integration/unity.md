# Unity Integration Guide

**Deploy AI models to Unity projects**

---

## 📋 Overview

This guide covers multiple approaches to integrate AI models into Unity:

1. **ONNX Runtime** - Run models directly in Unity (recommended)
2. **Barracuda** - Unity's neural network inference library
3. **Python Bridge** - Call Python scripts from Unity
4. **Cloud API** - Use remote AI services
5. **ML-Agents** - Unity's RL framework

---

## Method 1: ONNX Runtime (Recommended)

**Best for**: Production deployment, fast inference, cross-platform

### Prerequisites

- Unity 2020.3+ (LTS recommended)
- ONNX Runtime for Unity package
- Your trained model in ONNX format

### Step 1: Convert Model to ONNX

```bash
cd experiments/07_practical_game_ai/3_optimization

# Convert PyTorch model to ONNX
python onnx_export.py \
    --model checkpoints/classifier.pth \
    --output classifier.onnx \
    --input-shape 1 3 224 224 \
    --opset 12
```

### Step 2: Install ONNX Runtime in Unity

**Option A: Unity Package Manager**
1. Window → Package Manager
2. Add package from git URL:
   ```
   https://github.com/asus4/onnxruntime-unity.git
   ```

**Option B: Manual Installation**
1. Download [onnxruntime-unity](https://github.com/asus4/onnxruntime-unity/releases)
2. Extract to `Assets/Plugins/OnnxRuntime/`

### Step 3: Unity C# Implementation

#### Image Classifier Example

```csharp
using UnityEngine;
using Unity.Barracuda;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Linq;

public class AIImageClassifier : MonoBehaviour
{
    [SerializeField] private string modelPath = "Assets/Models/classifier.onnx";
    [SerializeField] private TextAsset classNamesFile;

    private InferenceSession session;
    private string[] classNames;

    void Start()
    {
        // Load ONNX model
        session = new InferenceSession(modelPath);

        // Load class names
        classNames = classNamesFile.text.Split('\n');

        Debug.Log($"Model loaded: {modelPath}");
        Debug.Log($"Classes: {classNames.Length}");
    }

    public string ClassifyTexture(Texture2D texture)
    {
        // Preprocess image
        var inputTensor = PreprocessTexture(texture);

        // Create input
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        // Run inference
        using (var results = session.Run(inputs))
        {
            // Get output
            var output = results.First().AsEnumerable<float>().ToArray();

            // Find top class
            int topClass = System.Array.IndexOf(output, output.Max());
            float confidence = output[topClass] * 100f;

            Debug.Log($"Prediction: {classNames[topClass]} ({confidence:F1}%)");

            return classNames[topClass];
        }
    }

    private DenseTensor<float> PreprocessTexture(Texture2D texture)
    {
        // Resize to model input size (224x224)
        var resized = ResizeTexture(texture, 224, 224);

        // Convert to tensor [1, 3, 224, 224]
        var tensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });

        // Normalize: (pixel / 255 - mean) / std
        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };

        for (int y = 0; y < 224; y++)
        {
            for (int x = 0; x < 224; x++)
            {
                Color pixel = resized.GetPixel(x, y);

                tensor[0, 0, y, x] = (pixel.r - mean[0]) / std[0];
                tensor[0, 1, y, x] = (pixel.g - mean[1]) / std[1];
                tensor[0, 2, y, x] = (pixel.b - mean[2]) / std[2];
            }
        }

        return tensor;
    }

    private Texture2D ResizeTexture(Texture2D source, int targetWidth, int targetHeight)
    {
        RenderTexture rt = RenderTexture.GetTemporary(targetWidth, targetHeight);
        RenderTexture.active = rt;

        Graphics.Blit(source, rt);

        Texture2D result = new Texture2D(targetWidth, targetHeight);
        result.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        result.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        return result;
    }

    void OnDestroy()
    {
        session?.Dispose();
    }
}
```

#### Usage in Game

```csharp
public class AssetAutoTagger : MonoBehaviour
{
    [SerializeField] private AIImageClassifier classifier;
    [SerializeField] private Texture2D testImage;

    void Start()
    {
        // Classify asset
        string category = classifier.ClassifyTexture(testImage);

        // Auto-organize or tag
        Debug.Log($"Asset category: {category}");
    }
}
```

### Step 4: Performance Optimization

```csharp
public class OptimizedClassifier : MonoBehaviour
{
    private InferenceSession session;
    private DenseTensor<float> reusableTensor;  // Reuse tensor

    void Start()
    {
        // Configure session options
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_PARALLEL
        };

        session = new InferenceSession(modelPath, options);

        // Pre-allocate tensor
        reusableTensor = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
    }

    public string ClassifyFast(Texture2D texture)
    {
        // Reuse pre-allocated tensor
        FillTensor(texture, reusableTensor);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", reusableTensor)
        };

        using (var results = session.Run(inputs))
        {
            var output = results.First().AsEnumerable<float>().ToArray();
            int topClass = System.Array.IndexOf(output, output.Max());
            return classNames[topClass];
        }
    }
}
```

---

## Method 2: Unity Barracuda

**Best for**: Simple models, Unity Cloud Build compatibility

### Installation

1. Window → Package Manager
2. Search "Barracuda"
3. Install

### Convert ONNX to Barracuda

```csharp
// Unity will auto-convert .onnx files placed in Assets
// Or use menu: Assets → Barracuda → Import ONNX Model
```

### Implementation

```csharp
using Unity.Barracuda;

public class BarracudaClassifier : MonoBehaviour
{
    [SerializeField] private NNModel modelAsset;
    [SerializeField] private TextAsset classNames;

    private Model runtimeModel;
    private IWorker worker;

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);
    }

    public string Classify(Texture2D texture)
    {
        // Prepare input
        using (var tensor = new Tensor(texture, channels: 3))
        {
            // Execute model
            worker.Execute(tensor);

            // Get output
            var output = worker.PeekOutput();

            // Find max
            int maxIndex = output.ArgMax()[0];
            string[] classes = classNames.text.Split('\n');

            return classes[maxIndex];
        }
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}
```

---

## Method 3: Python Bridge

**Best for**: Rapid prototyping, complex preprocessing

### Using UnityPython

```csharp
using UnityEngine;
using System.Diagnostics;

public class PythonBridge : MonoBehaviour
{
    private Process pythonProcess;

    void Start()
    {
        StartPythonServer();
    }

    void StartPythonServer()
    {
        pythonProcess = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = "server.py",
                RedirectStandardOutput = true,
                RedirectStandardInput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            }
        };

        pythonProcess.Start();
        Debug.Log("Python server started");
    }

    public string ClassifyImage(string imagePath)
    {
        // Send request to Python
        pythonProcess.StandardInput.WriteLine($"classify {imagePath}");

        // Read response
        string result = pythonProcess.StandardOutput.ReadLine();

        return result;
    }

    void OnApplicationQuit()
    {
        pythonProcess?.Kill();
    }
}
```

**Python Server (server.py)**:
```python
import sys
from image_classifier import GameAssetClassifier

classifier = GameAssetClassifier()
classifier.load("checkpoints/best.pth")

print("Python server ready", flush=True)

for line in sys.stdin:
    cmd, path = line.strip().split(maxsplit=1)

    if cmd == "classify":
        result = classifier.predict(path)
        print(result["top_class"], flush=True)
```

---

## Method 4: Cloud API

**Best for**: Heavy models, online games, analytics

### FastAPI Backend

See [Web Integration](web.md#fastapi-setup) for server setup.

### Unity Client

```csharp
using UnityEngine;
using UnityEngine.Networking;
using System.Collections;

public class CloudAIClient : MonoBehaviour
{
    [SerializeField] private string apiUrl = "http://localhost:8000";

    public IEnumerator ClassifyImage(byte[] imageBytes, System.Action<string> callback)
    {
        WWWForm form = new WWWForm();
        form.AddBinaryData("file", imageBytes, "image.png", "image/png");

        using (UnityWebRequest www = UnityWebRequest.Post($"{apiUrl}/classify", form))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                string json = www.downloadHandler.text;
                var response = JsonUtility.FromJson<ClassifyResponse>(json);
                callback(response.category);
            }
            else
            {
                Debug.LogError($"API Error: {www.error}");
            }
        }
    }
}

[System.Serializable]
public class ClassifyResponse
{
    public string category;
    public float confidence;
}

// Usage
public class Example : MonoBehaviour
{
    [SerializeField] private CloudAIClient aiClient;

    void Start()
    {
        Texture2D texture = GetScreenshot();
        byte[] bytes = texture.EncodeToPNG();

        StartCoroutine(aiClient.ClassifyImage(bytes, result =>
        {
            Debug.Log($"Classification: {result}");
        }));
    }
}
```

---

## Method 5: Unity ML-Agents

**Best for**: Reinforcement learning, NPC behavior

### Installation

1. Install ML-Agents package:
   ```
   Window → Package Manager → Add from git URL
   https://github.com/Unity-Technologies/ml-agents.git?path=com.unity.ml-agents
   ```

2. Install Python package:
   ```bash
   pip install mlagents
   ```

### Create Training Environment

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class NPCAgent : Agent
{
    [SerializeField] private Transform target;
    [SerializeField] private float moveSpeed = 5f;

    public override void OnEpisodeBegin()
    {
        // Reset environment
        transform.localPosition = Vector3.zero;
        target.localPosition = new Vector3(
            Random.Range(-4f, 4f),
            0f,
            Random.Range(-4f, 4f)
        );
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Agent position
        sensor.AddObservation(transform.localPosition);

        // Target position
        sensor.AddObservation(target.localPosition);

        // Distance to target
        sensor.AddObservation(Vector3.Distance(transform.localPosition, target.localPosition));
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Get actions
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];

        // Move
        Vector3 move = new Vector3(moveX, 0, moveZ) * moveSpeed * Time.deltaTime;
        transform.localPosition += move;

        // Rewards
        float distanceToTarget = Vector3.Distance(transform.localPosition, target.localPosition);

        if (distanceToTarget < 1.5f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else
        {
            SetReward(-0.001f);  // Time penalty
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Manual control for testing
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetAxis("Vertical");
    }
}
```

### Train with PPO

```bash
# Create config (config.yaml)
# See experiments/08_reinforcement_learning/README.md

# Train
mlagents-learn config.yaml --run-id=npc_training

# In Unity, press Play when prompted
```

### Use Trained Model

1. Export model: `results/npc_training/NPCAgent.onnx`
2. Drag to Unity Assets
3. Assign to Agent component → Model field

```csharp
// Agent will now use trained behavior automatically
```

---

## Complete Example: Asset Auto-Organizer

```csharp
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections.Generic;

public class AIAssetOrganizer : EditorWindow
{
    private AIImageClassifier classifier;
    private List<Texture2D> pendingAssets = new List<Texture2D>();

    [MenuItem("Tools/AI Asset Organizer")]
    static void ShowWindow()
    {
        GetWindow<AIAssetOrganizer>("AI Organizer");
    }

    void OnEnable()
    {
        classifier = FindObjectOfType<AIImageClassifier>();
        if (classifier == null)
        {
            GameObject go = new GameObject("AI Classifier");
            classifier = go.AddComponent<AIImageClassifier>();
        }
    }

    void OnGUI()
    {
        GUILayout.Label("AI Asset Organizer", EditorStyles.boldLabel);

        if (GUILayout.Button("Scan Assets Folder"))
        {
            ScanAssets();
        }

        GUILayout.Label($"Found {pendingAssets.Count} assets", EditorStyles.label);

        if (pendingAssets.Count > 0 && GUILayout.Button("Organize All"))
        {
            OrganizeAssets();
        }
    }

    void ScanAssets()
    {
        pendingAssets.Clear();

        string[] guids = AssetDatabase.FindAssets("t:Texture2D", new[] { "Assets/ImportedAssets" });

        foreach (string guid in guids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            Texture2D texture = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            if (texture != null)
            {
                pendingAssets.Add(texture);
            }
        }

        Debug.Log($"Found {pendingAssets.Count} textures");
    }

    void OrganizeAssets()
    {
        int organized = 0;

        foreach (var texture in pendingAssets)
        {
            string category = classifier.ClassifyTexture(texture);

            string currentPath = AssetDatabase.GetAssetPath(texture);
            string newPath = $"Assets/Organized/{category}/{Path.GetFileName(currentPath)}";

            // Create directory if needed
            string dir = Path.GetDirectoryName(newPath);
            if (!Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            // Move asset
            AssetDatabase.MoveAsset(currentPath, newPath);

            organized++;
            EditorUtility.DisplayProgressBar("Organizing Assets",
                $"Processing {organized}/{pendingAssets.Count}",
                (float)organized / pendingAssets.Count);
        }

        EditorUtility.ClearProgressBar();
        AssetDatabase.Refresh();

        Debug.Log($"✓ Organized {organized} assets");
        pendingAssets.Clear();
    }
}
```

---

## Performance Tips

### 1. Model Optimization
- Use quantized models (INT8/FP16)
- Simplify architecture (MobileNet instead of ResNet)
- Remove unnecessary outputs

### 2. Runtime Optimization
```csharp
// Reuse tensors
private DenseTensor<float> reusableTensor;

// Batch processing
public string[] ClassifyBatch(Texture2D[] textures)
{
    var batchTensor = new DenseTensor<float>(new[] { textures.Length, 3, 224, 224 });
    // Fill batch tensor...

    // Single inference call for all
    using (var results = session.Run(inputs))
    {
        // Process batch results
    }
}
```

### 3. Threading
```csharp
using System.Threading.Tasks;

public async Task<string> ClassifyAsync(Texture2D texture)
{
    return await Task.Run(() => Classify(texture));
}
```

### 4. Caching
```csharp
private Dictionary<string, string> classificationCache = new Dictionary<string, string>();

public string ClassifyWithCache(Texture2D texture)
{
    string hash = GetTextureHash(texture);

    if (classificationCache.TryGetValue(hash, out string cached))
    {
        return cached;
    }

    string result = Classify(texture);
    classificationCache[hash] = result;

    return result;
}
```

---

## Troubleshooting

### Issue: Model not loading

**Check**:
- ONNX file is in `StreamingAssets/` or `Resources/`
- Path is correct (use `Application.streamingAssetsPath`)
- Model was exported with correct opset (12-14)

### Issue: Slow inference

**Solutions**:
- Use GPU provider (if available)
- Reduce input size
- Quantize model
- Batch multiple inferences

### Issue: Incorrect predictions

**Check**:
- Preprocessing matches training (normalization, size)
- Input tensor shape is correct
- Class names order matches training

---

## Next Steps

- **[Unreal Integration](unreal.md)** - Deploy to Unreal Engine
- **[Performance Optimization](../best-practices/performance.md)** - Speed up inference
- **[Model Compression](../advanced/model-compression.md)** - Reduce model size

---

**Deploy AI to Unity like a pro! 🎮🤖**
