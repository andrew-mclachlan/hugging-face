# Open Voice Detect - Usage Examples

This folder contains example scripts showing different ways to use the Open Voice Detect model.

## Available Examples

### 1. `quick_start.py` - Minimal Example

The simplest way to load and use the model.

```bash
cd examples
python quick_start.py
```

### 2. `use_model.py` - Comprehensive Demo

Full-featured example with detailed explanations, batch processing, and usage patterns.

```bash
cd examples
python use_model.py
```

Features:

- Model loading with device selection (CPU/CUDA)
- Single sample prediction
- Batch processing
- Probability outputs
- Real-world usage guidance

### 3. `use_dataset.py` - Working with Datasets

Demonstrates loading and using the provided datasets for evaluation.

```bash
python examples/use_dataset.py
```

Features:

- Loading test datasets
- Making predictions on multiple samples
- Prediction probabilities and confidence scores
- Full test set evaluation
- Class-wise accuracy metrics

**Note**: Run `python datasets/generate_dataset.py` first to create the datasets.

### 4. `load_from_hub.py` - Hugging Face Hub Integration

Shows how to load the model from Hugging Face Hub (requires uploading first).

```bash
cd examples
python load_from_hub.py
```

## Requirements

Install dependencies:

```bash
pip install torch numpy
pip install huggingface_hub  # Only for load_from_hub.py
```

For real audio processing, also install:

```bash
pip install librosa soundfile
```

## Using with Real Audio

The model expects 128-dimensional feature vectors. Here's how to extract them from audio:

```python
import librosa
import numpy as np

# Load audio file
audio, sr = librosa.load('your_audio.wav', sr=16000)

# Extract MFCC features (128 dimensions)
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)

# Average over time to get single feature vector
features = np.mean(mfcc, axis=1)

# Use with model
prediction = model.predict(torch.from_numpy(features).unsqueeze(0))
```

## Model Architecture

- **Input**: 128-dimensional feature vector
- **Hidden layers**: 2 fully connected layers with ReLU and dropout
- **Output**: 2 classes (0: no voice, 1: voice present)

## Training the Model

The repository now includes example datasets and training scripts!

### Quick Training

```bash
# Generate synthetic datasets
cd datasets && python generate_dataset.py && cd ..

# Train the model
python train.py --epochs 50 --batch-size 32

# Test with the trained model
python examples/use_dataset.py
```

See the main [README.md](../README.md) and [datasets/README.md](../datasets/README.md) for:

- Detailed training instructions
- Manual training examples
- Using PyTorch DataLoader
- Model evaluation with metrics

## Notes

- The initial model weights are randomly initialized (for demonstration)
- Example datasets are provided in the `datasets/` directory
- Use `train.py` to train the model on the synthetic datasets
- For production use, train on real voice activity detection data
- Feature extraction is not included; use librosa or torchaudio
