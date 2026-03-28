# Model Evaluation Patterns

Comprehensive guide to evaluating ML models with proper metrics and visualizations.

## Classification Metrics

### Basic Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro')
    }
    return metrics
```

### Per-Class Metrics

```python
from sklearn.metrics import classification_report

def detailed_metrics(y_true, y_pred, class_names=None):
    """Generate detailed classification report."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print(report)
    return report
```

### Confusion Matrix

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False):
    """
    Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: If True, normalize the matrix
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
```

### ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np

def plot_roc_curve(y_true, y_scores, num_classes):
    """
    Plot ROC curve for multi-class classification.

    Args:
        y_true: One-hot encoded true labels
        y_scores: Predicted probabilities
        num_classes: Number of classes
    """
    plt.figure(figsize=(10, 8))

    # Compute ROC curve for each class
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr,
            label=f'Class {i} (AUC = {roc_auc:.2f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300)
    plt.show()
```

## Complete Evaluation Function

```python
def evaluate_model(model, test_loader, device, class_names=None):
    """
    Comprehensive model evaluation.

    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        device: Device to run on
        class_names: List of class names

    Returns:
        dict: Evaluation metrics and predictions
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Evaluating"):
            data, labels = data.to(device), labels.to(device)

            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1': f1_score(all_labels, all_preds, average='macro')
    }

    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("="*50)

    # Generate confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names)

    # Generate classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return {
        'metrics': metrics,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
```

## Visualization Patterns

### Training History Plot

```python
def plot_training_history(history):
    """
    Plot training and validation loss/accuracy.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()
```

### Prediction Visualization

```python
def visualize_predictions(model, test_loader, device, num_images=10):
    """
    Visualize sample predictions.

    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Device to run on
        num_images: Number of images to visualize
    """
    model.eval()

    # Get a batch
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(min(num_images, len(images))):
        img = images[i].cpu()
        true_label = labels[i].cpu().item()
        pred_label = preds[i].cpu().item()

        # Denormalize if needed
        img = img.permute(1, 2, 0)  # CHW -> HWC

        axes[i].imshow(img)
        axes[i].set_title(
            f'True: {true_label}, Pred: {pred_label}',
            color='green' if true_label == pred_label else 'red'
        )
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=300)
    plt.show()
```

### Top-K Accuracy

```python
def top_k_accuracy(output, target, k=5):
    """
    Compute top-k accuracy.

    Args:
        output: Model predictions (logits)
        target: True labels
        k: k for top-k

    Returns:
        float: Top-k accuracy
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()
```

## Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression(y_true, y_pred):
    """
    Compute regression metrics.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

    print("\nRegression Metrics:")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    print("="*50)

    return metrics


def plot_regression_results(y_true, y_pred):
    """Plot regression predictions vs actual."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.5)
    ax1.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', lw=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title('Predictions vs True Values')
    ax1.grid(True)

    # Residual plot
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predictions')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=300)
    plt.show()
```

## Model Comparison

```python
def compare_models(models_results):
    """
    Compare multiple models.

    Args:
        models_results: Dict mapping model names to their metrics
    """
    df = pd.DataFrame(models_results).T
    print("\nModel Comparison:")
    print(df.to_string())

    # Bar plot
    df.plot(kind='bar', figsize=(12, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.legend(title='Metrics')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()


# Usage example
results = {
    'Model A': {'accuracy': 0.95, 'f1': 0.94, 'precision': 0.96},
    'Model B': {'accuracy': 0.93, 'f1': 0.92, 'precision': 0.94},
    'Model C': {'accuracy': 0.97, 'f1': 0.96, 'precision': 0.97}
}
compare_models(results)
```
