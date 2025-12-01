---
language:
  - en
license: mit
tags:
  - audio
  - voice-activity-detection
  - pytorch
  - audio-classification
library_name: pytorch
pipeline_tag: audio-classification
---

# Open Voice Detect

A simple binary classification model for voice activity detection (VAD).

## Model Description

`open-voice-detect` is a lightweight neural network designed to detect the presence of voice in audio features. This is a minimal "hello world" implementation demonstrating how to structure a PyTorch model for Hugging Face.

## Model Architecture

- **Input**: 128-dimensional feature vector (e.g., MFCCs, mel-spectrogram features)
- **Hidden layers**: 2 fully connected layers with ReLU activation and dropout
- **Output**: 2 classes (voice present / no voice)

## Intended Use

This model is a demonstration/starting point for:

- Learning how to structure models for Hugging Face
- Building voice activity detection systems
- Understanding basic audio classification

## How to Use

### Quick Inference

```python
import torch
from model import OpenVoiceDetect

# Load model
model = OpenVoiceDetect(input_size=128, hidden_size=64)
checkpoint = torch.load('pytorch_model.bin')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input (batch_size, 128)
# In practice, extract 128-dimensional features from audio
audio_features = torch.randn(1, 128)

# Get prediction
prediction = model.predict(audio_features)
print(f"Voice detected: {prediction.item() == 1}")
```

## Training with Datasets

This repository includes example datasets in the `datasets/` directory that you can use to train the model.

### Step 1: Generate Training Data

First, generate synthetic training and test datasets:

```bash
cd datasets
python generate_dataset.py
cd ..
```

This creates:

- `train_features.npy` - 1000 training samples with 128 features each
- `train_labels.npy` - Training labels (0=no voice, 1=voice)
- `test_features.npy` - 200 test samples
- `test_labels.npy` - Test labels

### Step 2: Train the Model (Easy Way)

Use the provided training script:

```bash
python train.py --epochs 50 --batch-size 32 --lr 0.001
```

Options:

- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--hidden-size`: Hidden layer size (default: 64)
- `--save-path`: Where to save trained model (default: trained_model.pth)

The script will:

- Train the model with progress tracking
- Save the best model automatically
- Show detailed evaluation metrics
- Display confusion matrix

### Alternative: Manual Training in Python

See the [datasets/README.md](datasets/README.md) and [train.py](train.py) for detailed examples including:

- Complete training loops with mini-batches
- Using PyTorch DataLoader for efficient training
- Batch prediction on CSV data
- Single sample prediction with confidence scores
- Model evaluation with sklearn metrics (classification report, confusion matrix)
- Real-world feature extraction from audio files using librosa

## Training Data

The `datasets/` directory includes:

- **sample_voice_data.csv** - Small CSV with 10 example samples
- **generate_dataset.py** - Script to generate synthetic training/test data
- Comprehensive examples and documentation

For production use, you would need real audio data:

- Audio samples with voice activity (speech, singing, etc.)
- Audio samples without voice (silence, noise, music)

The synthetic datasets are balanced (50% voice, 50% no voice) for demonstration purposes.

## Model Details

- **Model type**: Binary classifier
- **Framework**: PyTorch
- **Parameters**: ~10K
- **Input size**: 128 features
- **Output**: Binary classification (0: no voice, 1: voice)

## Limitations

- This is a demonstration model with random weights
- Not trained on real data
- Requires feature extraction preprocessing (not included)
- For production use, train on appropriate voice detection datasets

## License

MIT
