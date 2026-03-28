# Data Loading Patterns

Efficient data loading and preprocessing patterns for PyTorch.

## Basic Dataset and DataLoader

### Custom Dataset Pattern

```python
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class CustomDataset(Dataset):
    """
    Template for custom PyTorch dataset.

    Responsibilities:
    - Store data references (not the data itself)
    - Load data on demand (__getitem__)
    - Return data in expected format
    """

    def __init__(self, data_path, transform=None):
        """
        Initialize dataset.

        Args:
            data_path: Path to data
            transform: Optional transform to apply
        """
        self.data_path = data_path
        self.transform = transform

        # Load metadata (file list, labels, etc.)
        # Don't load actual data here!
        self.file_list = self._get_file_list()
        self.labels = self._get_labels()

    def __len__(self):
        """Return total number of samples."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Load and return a single sample.

        Args:
            idx: Index of sample to load

        Returns:
            tuple: (data, label)
        """
        # Load data (lazy loading - only when requested)
        data = self._load_data(self.file_list[idx])
        label = self.labels[idx]

        # Apply transforms if any
        if self.transform:
            data = self.transform(data)

        return data, label

    def _get_file_list(self):
        # Implementation to get list of files
        pass

    def _get_labels(self):
        # Implementation to get labels
        pass

    def _load_data(self, file_path):
        # Implementation to load single file
        pass
```

### DataLoader Configuration

```python
# Training DataLoader (with augmentation)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,           # Shuffle for each epoch
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    drop_last=True,         # Drop incomplete batch
    persistent_workers=True # Keep workers alive between epochs
)

# Validation/Test DataLoader (no augmentation)
val_loader = DataLoader(
    val_dataset,
    batch_size=128,         # Can be larger (no backprop)
    shuffle=False,          # No need to shuffle
    num_workers=4,
    pin_memory=True
)
```

## Common Dataset Implementations

### Image Classification Dataset

```python
from PIL import Image
import os

class ImageClassificationDataset(Dataset):
    """
    Dataset for image classification.

    Expected directory structure:
    data_root/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img1.jpg
            img2.jpg
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Build file list and labels
        self.samples = []
        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            self.class_to_idx[class_name] = idx

            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

# Usage
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataset = ImageClassificationDataset('data/train', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### CSV/Tabular Dataset

```python
import pandas as pd

class TabularDataset(Dataset):
    """Dataset for tabular/CSV data."""

    def __init__(self, csv_file, feature_cols, target_col):
        """
        Args:
            csv_file: Path to CSV file
            feature_cols: List of feature column names
            target_col: Target column name
        """
        # Load CSV
        self.df = pd.read_csv(csv_file)

        # Extract features and targets
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.targets = self.df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx])
        target = torch.tensor(self.targets[idx])
        return features, target

# Usage
dataset = TabularDataset(
    'data/train.csv',
    feature_cols=['feature1', 'feature2', 'feature3'],
    target_col='target'
)
```

### Text Dataset

```python
class TextDataset(Dataset):
    """Dataset for text classification."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: List of text strings
            labels: List of integer labels
            tokenizer: Tokenizer instance (e.g., from transformers)
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }

# Custom collate function for batching
def collate_fn(batch):
    """Collate function for text data."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# Usage
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = TextDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

## Data Transforms

### Image Transforms

```python
from torchvision import transforms

# Training transforms (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Validation transforms (minimal)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

### Custom Transform

```python
class CustomTransform:
    """Custom transform example."""

    def __init__(self, param):
        self.param = param

    def __call__(self, image):
        # Apply custom transformation
        # image is PIL Image
        return transformed_image

# Use in Compose
transform = transforms.Compose([
    CustomTransform(param=0.5),
    transforms.ToTensor()
])
```

## Data Splitting

### Train/Val/Test Split

```python
from torch.utils.data import random_split

# Split dataset
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
```

### K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold + 1}")

    # Create subset datasets
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    # Create loaders
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

    # Train model on this fold
    train_model(model, train_loader, val_loader)
```

## Advanced Patterns

### Weighted Random Sampling

```python
from torch.utils.data import WeightedRandomSampler

# For imbalanced datasets
def get_class_weights(dataset):
    """Calculate sampling weights for imbalanced dataset."""
    labels = [label for _, label in dataset]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    return sample_weights

sample_weights = get_class_weights(train_dataset)
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    sampler=sampler  # Use sampler instead of shuffle
)
```

### Caching Dataset

```python
class CachedDataset(Dataset):
    """Dataset that caches loaded samples in memory."""

    def __init__(self, base_dataset, cache_size=1000):
        self.base_dataset = base_dataset
        self.cache = {}
        self.cache_size = cache_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]

        # Load from base dataset
        sample = self.base_dataset[idx]

        # Add to cache if not full
        if len(self.cache) < self.cache_size:
            self.cache[idx] = sample

        return sample
```

### Prefetching Data

```python
class DataPrefetcher:
    """Prefetch data to GPU for faster training."""

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data, self.next_target = next(self.loader)
        except StopIteration:
            self.next_data = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        target = self.next_target
        if data is not None:
            data.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return data, target

# Usage
prefetcher = DataPrefetcher(train_loader)
data, target = prefetcher.next()
while data is not None:
    # Training code
    data, target = prefetcher.next()
```

## Performance Tips

1. **Use num_workers**: Parallel data loading (typically 4-8)
2. **Use pin_memory**: Faster CPU-to-GPU transfer
3. **Use persistent_workers**: Keep workers alive between epochs
4. **Larger batch size for validation**: No gradients needed
5. **Cache small datasets**: If dataset fits in RAM
6. **Prefetch to GPU**: Overlap data transfer with computation
7. **Optimize transforms**: Do expensive transforms offline if possible

## Debugging Data Loading

```python
def debug_dataloader(loader):
    """Debug DataLoader to find issues."""
    print(f"Dataset size: {len(loader.dataset)}")
    print(f"Batch size: {loader.batch_size}")
    print(f"Num batches: {len(loader)}")

    # Load one batch
    data, labels = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  Data: {data.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"\nData statistics:")
    print(f"  Min: {data.min():.4f}")
    print(f"  Max: {data.max():.4f}")
    print(f"  Mean: {data.mean():.4f}")
    print(f"  Std: {data.std():.4f}")
    print(f"\nLabels:")
    print(f"  Unique: {labels.unique()}")
    print(f"  Counts: {[(l.item(), (labels == l).sum().item()) for l in labels.unique()]}")

debug_dataloader(train_loader)
```
