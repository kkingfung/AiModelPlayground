# Neural Network Architectures Reference

Quick reference for common neural network architectures and their use cases.

## Feedforward Networks (MLP)

### When to Use
- Tabular data
- Simple classification/regression
- Fully connected problems

### Basic Architecture

```python
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout=0.5):
        super(MLP, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Usage
model = MLP(
    input_size=784,
    hidden_sizes=[512, 256, 128],
    output_size=10,
    dropout=0.5
)
```

## Convolutional Neural Networks (CNN)

### When to Use
- Image classification
- Object detection
- Image segmentation
- Any grid-like data

### LeNet-5 Style (Small)

```python
class LeNet5(nn.Module):
    """
    Classic LeNet-5 for MNIST-like tasks.
    Input: 28x28 grayscale images
    """
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### ResNet-Style (Deep)

```python
class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        out += self.skip(identity)
        out = F.relu(out)

        return out


class SimpleResNet(nn.Module):
    """Simplified ResNet for demonstration."""
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

## Recurrent Neural Networks (RNN/LSTM/GRU)

### When to Use
- Sequential data (time series, text)
- Variable-length inputs
- Temporal dependencies

### LSTM for Text Classification

```python
class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier.
    Input: (batch_size, seq_len, embedding_dim)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # (batch, hidden_dim*2)

        # Classifier
        out = self.dropout(hidden)
        out = self.fc(out)

        return out
```

### GRU for Sequence Prediction

```python
class GRUPredictor(nn.Module):
    """GRU for sequence-to-sequence prediction."""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUPredictor, self).__init__()

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, hidden = self.gru(x)
        # out: (batch, seq_len, hidden_size)

        # Predict for each timestep
        predictions = self.fc(out)  # (batch, seq_len, output_size)

        return predictions
```

## Transformer Architecture

### When to Use
- NLP tasks (translation, summarization, Q&A)
- Long-range dependencies
- Parallel processing of sequences

### Simple Transformer

```python
class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier.
    Input: (batch_size, seq_len)
    """
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, num_classes, max_len=512):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.size(1)

        # Embedding + positional encoding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = embedded + self.pos_encoding[:, :seq_len, :]

        # Transformer
        transformed = self.transformer(embedded)

        # Use [CLS] token (first token) or mean pooling
        pooled = transformed.mean(dim=1)  # (batch, embedding_dim)

        # Classifier
        out = self.fc(pooled)

        return out
```

## Autoencoder

### When to Use
- Dimensionality reduction
- Anomaly detection
- Denoising
- Feature learning

### Basic Autoencoder

```python
class Autoencoder(nn.Module):
    """Simple autoencoder for dimensionality reduction."""
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # For normalized inputs [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)
```

### Variational Autoencoder (VAE)

```python
class VAE(nn.Module):
    """Variational Autoencoder for generative modeling."""
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

## Architecture Selection Guide

| Task | Recommended Architecture | Alternatives |
|------|-------------------------|--------------|
| Image Classification | ResNet, EfficientNet | VGG, MobileNet |
| Object Detection | YOLO, Faster R-CNN | RetinaNet, SSD |
| Image Segmentation | U-Net, Mask R-CNN | DeepLab, FCN |
| Text Classification | BERT, RoBERTa | LSTM, CNN |
| Sequence-to-Sequence | Transformer, T5 | LSTM + Attention |
| Time Series | LSTM, GRU | Transformer, TCN |
| Anomaly Detection | Autoencoder, VAE | Isolation Forest |
| Generative | GAN, VAE, Diffusion | Flow-based models |

## Quick Architecture Tips

1. **Start Simple**: Begin with a simple architecture and add complexity only if needed
2. **Batch Normalization**: Helps training deep networks
3. **Skip Connections**: Enable very deep networks (ResNet-style)
4. **Dropout**: Prevents overfitting (typical: 0.2-0.5)
5. **Activation Functions**:
   - ReLU: Default choice
   - LeakyReLU: Prevents dying neurons
   - GELU: Better for Transformers
   - Sigmoid/Tanh: Output layers for specific ranges
6. **Pooling**: Max pooling for features, Average pooling for final layers
