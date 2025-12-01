---
language:
  - en
license: mit
task_categories:
  - audio-classification
task_ids:
  - audio-intent-classification
tags:
  - audio
  - voice-activity-detection
  - voice-detection
  - speech-detection
  - binary-classification
  - synthetic-data
pretty_name: Open Voice Detect Dataset
size_categories:
  - n<1K
---

# Dataset Card for Open Voice Detect

## Dataset Description

### Dataset Summary

The **Open Voice Detect Dataset** is a synthetic audio feature dataset designed for voice activity detection (VAD) tasks. It contains 128-dimensional feature vectors that simulate audio characteristics commonly extracted from real audio signals (such as MFCCs or mel-spectrogram features). This dataset serves as a demonstration and starting point for building voice activity detection systems.

The dataset includes:

- **Training set**: 1,000 samples
- **Test set**: 200 samples
- **Binary labels**: 0 (no voice) and 1 (voice present)

### Supported Tasks and Leaderboards

- **audio-classification**: The dataset can be used to train binary classifiers for voice activity detection
- **voice-activity-detection**: Primary task - detecting presence or absence of voice in audio

### Languages

- English (en) - The dataset is designed for voice detection in English audio, though the synthetic nature makes it language-agnostic

## Dataset Structure

### Data Instances

Each instance in the dataset consists of:

- **features**: A 128-dimensional feature vector (numpy array or list of floats)
- **label**: Binary label (0 = no voice, 1 = voice present)

Example from JSON format:

```json
{
  "features": [0.23, -0.45, 0.67, ..., -0.91],
  "label": 1
}
```

### Data Fields

- `features`: A list or array of 128 floating-point values representing audio features
  - In production, these would typically be MFCC coefficients, mel-spectrogram features, or other audio representations
  - Features are normalized with values roughly in the range [-1, 1]
- `label`: An integer indicating the presence of voice
  - `0`: No voice detected (silence, noise, or non-speech audio)
  - `1`: Voice detected (speech, singing, or vocal sounds)

### Data Splits

The dataset is split into two subsets:

| Split | Samples | Voice Samples | Non-Voice Samples |
| ----- | ------- | ------------- | ----------------- |
| train | 1,000   | ~500          | ~500              |
| test  | 200     | ~100          | ~100              |

The splits are balanced with approximately 50% voice and 50% non-voice samples.

### Data Format

The dataset is available in multiple formats:

1. **NumPy arrays (.npy)**:

   - `train_features.npy`: (1000, 128) - Training features
   - `train_labels.npy`: (1000,) - Training labels
   - `test_features.npy`: (200, 128) - Test features
   - `test_labels.npy`: (200,) - Test labels

2. **JSON files**:

   - `train_sample.json`: 10 sample training instances
   - `test_sample.json`: 10 sample test instances

3. **CSV format**:
   - `sample_voice_data.csv`: 10 samples with columns `feature_0` through `feature_127` and `label`

## Dataset Creation

### Curation Rationale

This dataset was created to:

1. Provide a simple, ready-to-use dataset for learning voice activity detection
2. Demonstrate how to structure audio feature datasets for machine learning
3. Enable quick prototyping and testing of voice detection models
4. Serve as a template for creating real-world voice detection datasets

### Source Data

#### Initial Data Collection and Normalization

The dataset is **synthetically generated** using the `generate_dataset.py` script. The synthetic features are designed to roughly simulate characteristics of real audio features:

- **Voice samples** (label=1): Higher variance in mid-frequency ranges (features 20-60) to simulate typical speech characteristics
- **Non-voice samples** (label=0): More uniform, lower variance distribution to simulate silence or background noise

#### Who are the source language producers?

N/A - This is synthetic data.

### Annotations

#### Annotation process

Labels are automatically generated during the synthetic data creation process. In a real-world scenario, you would need:

1. Collect actual audio recordings
2. Extract features using libraries like librosa
3. Manually label or use existing labeled datasets

#### Who are the annotators?

N/A - Automatically generated labels.

### Personal and Sensitive Information

None. The dataset is entirely synthetic and contains no real audio, personal information, or sensitive data.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset is for educational and demonstration purposes. It has minimal social impact as it contains no real speech or personal data.

### Discussion of Biases

Since the data is synthetic:

- It does not reflect real-world audio characteristics
- Voice and non-voice samples are perfectly balanced (50/50 split)
- It lacks the complexity and variability of real audio environments
- Models trained on this data will not generalize to real-world scenarios

### Other Known Limitations

- **Synthetic data**: Not suitable for production use without training on real data
- **Simplified features**: Real audio features have more complex patterns
- **No noise variations**: Missing real-world challenges like background noise, multiple speakers, music, etc.
- **No temporal information**: Features are treated as independent samples without sequence context
- **Limited diversity**: Does not capture different accents, languages, recording conditions, or speaker characteristics

## Additional Information

### Dataset Curators

Created as part of the Open Voice Detect project for educational purposes.

### Licensing Information

MIT License - Free to use for any purpose with attribution.

### Citation Information

```bibtex
@misc{open_voice_detect_dataset,
  title={Open Voice Detect Dataset},
  author={Open Voice Detect Project},
  year={2024},
  note={Synthetic dataset for voice activity detection demonstration}
}
```

### Contributions

This dataset serves as a starting point. For production use, consider:

1. **Real audio datasets**:

   - [LibriSpeech](https://www.openslr.org/12): Large-scale English speech corpus
   - [Common Voice](https://commonvoice.mozilla.org/): Multilingual speech dataset
   - [AudioSet](https://research.google.com/audioset/): Large-scale audio event dataset
   - [MUSAN](https://www.openslr.org/17/): Music, speech, and noise corpus

2. **Feature extraction**:

   ```python
   import librosa

   # Load audio file
   audio, sr = librosa.load('audio.wav', sr=16000)

   # Extract MFCCs (13 coefficients x ~10 frames = ~130 features)
   mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
   features = mfccs.flatten()[:128]  # Trim to 128 features
   ```

3. **Proper labeling**: Use existing VAD datasets or manual annotation for accurate labels

## Usage

### Loading the Dataset

**With NumPy:**

```python
import numpy as np

# Load training data
train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')

# Load test data
test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')

print(f"Training samples: {train_features.shape[0]}")
print(f"Test samples: {test_features.shape[0]}")
print(f"Feature dimensions: {train_features.shape[1]}")
```

**With Pandas (CSV):**

```python
import pandas as pd

df = pd.read_csv('sample_voice_data.csv')
features = df.iloc[:, :-1].values  # All columns except last
labels = df.iloc[:, -1].values     # Last column (label)
```

**With JSON:**

```python
import json
import numpy as np

with open('train_sample.json', 'r') as f:
    data = json.load(f)

features = np.array([sample['features'] for sample in data['samples']])
labels = np.array([sample['label'] for sample in data['samples']])
```

### Generating New Data

```bash
cd datasets
python generate_dataset.py
```

This will regenerate all dataset files with fresh synthetic data.

### Example Training Code

See the main repository for complete training examples using PyTorch, including:

- Basic training loops
- DataLoader usage
- Model evaluation
- Inference examples

## Contact

For questions or issues, please refer to the main repository documentation.
