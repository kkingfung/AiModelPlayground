# Evaluate Model Command

Evaluate a trained model and generate comprehensive metrics.

## Usage

When the user types `/evaluate [model-name]`:

1. **Locate model and evaluation script**:
   - Find saved model: `models/[model-name].pth`
   - Find evaluation script: `experiments/0X_*/evaluate.py`
   - If no evaluate.py exists, create one

2. **Load model and data**:
   - Load model checkpoint
   - Load test dataset
   - Set model to eval mode

3. **Compute metrics**:
   - Accuracy
   - Precision, Recall, F1
   - Confusion matrix
   - Per-class metrics (if classification)
   - Loss value

4. **Generate visualizations**:
   - Confusion matrix heatmap
   - Per-class accuracy bar chart
   - Sample predictions (correct and incorrect)
   - Loss/accuracy curves (if training history available)

5. **Create evaluation report**:
   - Save as Markdown in experiment folder
   - Include all metrics and visualizations

## Evaluation Report Template

```markdown
# Evaluation Report: [Model Name]

**Date**: 2026-03-25
**Model**: models/[model-name].pth
**Dataset**: [Dataset name] (test set)

---

## Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy | 97.89% |
| Precision (macro) | 97.85% |
| Recall (macro) | 97.82% |
| F1 Score (macro) | 97.83% |
| Loss | 0.0312 |

---

## Per-Class Metrics

| Class | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| 0 | 98.2% | 98.5% | 98.0% | 98.2% |
| 1 | 99.1% | 99.3% | 99.0% | 99.1% |
| ... | ... | ... | ... | ... |

---

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

---

## Sample Predictions

### Correct Predictions
![Correct Predictions](correct_predictions.png)

### Incorrect Predictions
![Incorrect Predictions](incorrect_predictions.png)

---

## Analysis

### Strengths
- [What the model does well]

### Weaknesses
- [Where the model struggles]

### Recommendations
- [Suggestions for improvement]

---

## Model Information

- **Architecture**: [Brief description]
- **Parameters**: [Total count]
- **Training time**: [Duration]
- **Hardware**: [CPU/GPU used]
```

## Output Format

```
=== Evaluating [Model Name] ===

Loading model: models/model-name.pth
Loading test data...

Running evaluation:
Processing: 100%|████████████| 10000/10000 [00:05<00:00]

Results:
┌─────────────────────┬─────────┐
│ Metric              │ Value   │
├─────────────────────┼─────────┤
│ Accuracy            │ 97.89%  │
│ Precision (macro)   │ 97.85%  │
│ Recall (macro)      │ 97.82%  │
│ F1 Score (macro)    │ 97.83%  │
│ Loss                │ 0.0312  │
└─────────────────────┴─────────┘

Generating visualizations...
✅ confusion_matrix.png
✅ per_class_accuracy.png
✅ sample_predictions.png

Evaluation report saved:
  experiments/0X_model-name/EVALUATION_REPORT.md

Open report to see detailed analysis and recommendations.
```

## Advanced Metrics

For specific model types, include:

**Classification**:
- ROC curve and AUC
- Precision-Recall curve
- Top-K accuracy

**Regression**:
- MSE, RMSE, MAE
- R² score
- Residual plots

**NLP**:
- BLEU score
- Perplexity
- Example generations

**Computer Vision**:
- IoU (segmentation)
- mAP (detection)
- Sample visualizations
