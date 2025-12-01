# Voice Detection Dataset

This directory contains example datasets for the OpenVoiceDetect model.

## Dataset Format

The model expects 128-dimensional feature vectors extracted from audio, typically:

- **MFCCs** (Mel-frequency cepstral coefficients)
- **Mel-spectrogram features**
- **Other audio features** that capture voice characteristics

## Files

### sample_voice_data.csv

A small CSV dataset with 10 samples showing the expected format:

- Columns: `feature_0` through `feature_127` (128 features)
- Label column: `label` (0 = no voice, 1 = voice)

### generate_dataset.py

Python script to generate synthetic datasets for training and testing.

**Usage:**

```bash
cd datasets
python generate_dataset.py
```

This will create:

- `train_features.npy` - Training features (1000 samples x 128 features)
- `train_labels.npy` - Training labels (1000 samples)
- `test_features.npy` - Test features (200 samples x 128 features)
- `test_labels.npy` - Test labels (200 samples)
- `train_sample.json` - Small JSON sample of training data (10 samples)
- `test_sample.json` - Small JSON sample of test data (10 samples)

## Loading the Dataset

### Python with NumPy

```python
import numpy as np

# Load training data
train_features = np.load('datasets/train_features.npy')
train_labels = np.load('datasets/train_labels.npy')

print(f"Training features shape: {train_features.shape}")
print(f"Training labels shape: {train_labels.shape}")
```

### Python with Pandas (CSV)

```python
import pandas as pd

# Load CSV data
df = pd.read_csv('datasets/sample_voice_data.csv')
features = df.iloc[:, :-1].values  # All columns except last
labels = df.iloc[:, -1].values     # Last column
```

### Python with JSON

```python
import json
import numpy as np

with open('datasets/train_sample.json', 'r') as f:
    data = json.load(f)

features = np.array([sample['features'] for sample in data['samples']])
labels = np.array([sample['label'] for sample in data['samples']])
```

## Using with the Model

### Example 1: Inference with Pre-trained Model

```python
import torch
import numpy as np
from model import OpenVoiceDetect

# Load model
model = OpenVoiceDetect(input_size=128, hidden_size=64)
checkpoint = torch.load('pytorch_model.bin')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test dataset
test_features = np.load('datasets/test_features.npy')
test_labels = np.load('datasets/test_labels.npy')

# Convert to torch tensors
test_tensor = torch.FloatTensor(test_features)

# Get predictions
predictions = model.predict(test_tensor)

# Calculate accuracy
accuracy = (predictions.numpy() == test_labels).mean()
print(f"Test Accuracy: {accuracy:.2%}")

# Show some example predictions
for i in range(5):
    pred = predictions[i].item()
    actual = test_labels[i]
    result = "✓" if pred == actual else "✗"
    print(f"Sample {i+1}: Predicted={pred}, Actual={actual} {result}")
```

### Example 2: Training the Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import OpenVoiceDetect

# Generate or load dataset
train_features = np.load('datasets/train_features.npy')
train_labels = np.load('datasets/train_labels.npy')
test_features = np.load('datasets/test_features.npy')
test_labels = np.load('datasets/test_labels.npy')

# Convert to torch tensors
X_train = torch.FloatTensor(train_features)
y_train = torch.LongTensor(train_labels)
X_test = torch.FloatTensor(test_features)
y_test = torch.LongTensor(test_labels)

# Initialize model
model = OpenVoiceDetect(input_size=128, hidden_size=64)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
batch_size = 32

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluate every 10 epochs
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test).float().mean()

        avg_loss = total_loss / (len(X_train) / batch_size)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2%}")

# Save trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'trained_model.pth')
print("Model saved to trained_model.pth")
```

### Example 3: Using PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import OpenVoiceDetect

class VoiceDataset(Dataset):
    """Custom dataset for voice detection."""

    def __init__(self, features_file, labels_file):
        self.features = np.load(features_file)
        self.labels = np.load(labels_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])[0]

# Create datasets
train_dataset = VoiceDataset('datasets/train_features.npy', 'datasets/train_labels.npy')
test_dataset = VoiceDataset('datasets/test_features.npy', 'datasets/test_labels.npy')

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
model = OpenVoiceDetect(input_size=128, hidden_size=64)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training with DataLoader
for epoch in range(20):
    model.train()
    for batch_features, batch_labels in train_loader:
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}: Test Accuracy = {accuracy:.2f}%")
```

### Example 4: Batch Prediction on CSV Data

```python
import pandas as pd
import torch
from model import OpenVoiceDetect

# Load model
model = OpenVoiceDetect(input_size=128, hidden_size=64)
checkpoint = torch.load('pytorch_model.bin')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load CSV dataset
df = pd.read_csv('datasets/sample_voice_data.csv')
features = df.iloc[:, :-1].values  # All columns except last
labels = df.iloc[:, -1].values     # Last column

# Convert to tensor
features_tensor = torch.FloatTensor(features)

# Make predictions
with torch.no_grad():
    predictions = model.predict(features_tensor)

# Add predictions to dataframe
df['predicted'] = predictions.numpy()
df['correct'] = df['label'] == df['predicted']

print("\nPrediction Results:")
print(df[['label', 'predicted', 'correct']])
print(f"\nAccuracy: {df['correct'].mean():.2%}")
```

### Example 5: Single Sample Prediction

```python
import torch
import numpy as np
from model import OpenVoiceDetect

# Load model
model = OpenVoiceDetect(input_size=128, hidden_size=64)
checkpoint = torch.load('pytorch_model.bin')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create a single sample (in practice, this would be extracted from audio)
single_sample = np.random.randn(128)

# Convert to tensor (add batch dimension)
sample_tensor = torch.FloatTensor(single_sample).unsqueeze(0)

# Get prediction
prediction = model.predict(sample_tensor)

# Get prediction probabilities
with torch.no_grad():
    logits = model(sample_tensor)
    probabilities = torch.softmax(logits, dim=1)

print(f"Prediction: {prediction.item()}")
print(f"Confidence - No Voice: {probabilities[0][0]:.2%}")
print(f"Confidence - Voice: {probabilities[0][1]:.2%}")
```

### Example 6: Model Evaluation with Metrics

```python
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model import OpenVoiceDetect

# Load model
model = OpenVoiceDetect(input_size=128, hidden_size=64)
checkpoint = torch.load('pytorch_model.bin')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test data
test_features = np.load('datasets/test_features.npy')
test_labels = np.load('datasets/test_labels.npy')
test_tensor = torch.FloatTensor(test_features)

# Get predictions
predictions = model.predict(test_tensor).numpy()

# Calculate metrics
print("\nClassification Report:")
print(classification_report(test_labels, predictions,
                          target_names=['No Voice', 'Voice']))

print("\nConfusion Matrix:")
cm = confusion_matrix(test_labels, predictions)
print(f"                Predicted")
print(f"              No Voice  Voice")
print(f"Actual No Voice  {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"       Voice     {cm[1][0]:4d}    {cm[1][1]:4d}")
```

## Real-World Data

For production use, you would need to:

1. **Collect real audio data** with voice and non-voice samples
2. **Extract features** using libraries like librosa:

   ```python
   import librosa

   # Load audio file
   audio, sr = librosa.load('audio.wav', sr=16000)

   # Extract MFCCs (13 coefficients x ~10 frames = 130 features, trim to 128)
   mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
   features = mfccs.flatten()[:128]
   ```

3. **Label your data** manually or use existing labeled datasets like:
   - LibriSpeech (speech)
   - AudioSet (various audio categories)
   - MUSAN (music, speech, noise)

## Dataset Statistics

The synthetic datasets generated have approximately:

- 50% voice samples
- 50% non-voice samples

This is balanced for demonstration purposes. Real-world datasets may have different distributions based on your use case.
