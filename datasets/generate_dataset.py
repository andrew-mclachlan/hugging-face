"""
Script to generate synthetic voice detection dataset.

This script creates synthetic audio feature data for training or testing
the OpenVoiceDetect model. In a real scenario, you would extract MFCC or
mel-spectrogram features from actual audio files.
"""

import numpy as np
import json


def generate_voice_features(n_samples=100, seed=42):
    """
    Generate synthetic audio features that simulate voice characteristics.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (features, labels) where features is shape (n_samples, 128)
        and labels is shape (n_samples,)
    """
    np.random.seed(seed)

    features = []
    labels = []

    for i in range(n_samples):
        # Randomly decide if this sample contains voice
        has_voice = np.random.choice([0, 1])

        if has_voice:
            # Voice samples tend to have stronger energy in certain frequency bands
            # Simulate this with higher variance in certain feature ranges
            feature_vector = np.random.randn(128) * 0.7
            feature_vector[20:60] += np.random.randn(40) * 0.5  # Mid frequencies
        else:
            # No voice samples (silence/noise) have more uniform distribution
            feature_vector = np.random.randn(128) * 0.3

        features.append(feature_vector)
        labels.append(has_voice)

    return np.array(features), np.array(labels)


def save_as_json(features, labels, filename):
    """Save dataset as JSON format."""
    data = {
        "samples": [
            {
                "features": features[i].tolist(),
                "label": int(labels[i])
            }
            for i in range(len(features))
        ]
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(features)} samples to {filename}")


def save_as_npy(features, labels, features_file, labels_file):
    """Save dataset as numpy arrays."""
    np.save(features_file, features)
    np.save(labels_file, labels)

    print(f"Saved features to {features_file}")
    print(f"Saved labels to {labels_file}")


if __name__ == "__main__":
    # Generate training dataset
    print("Generating training dataset...")
    train_features, train_labels = generate_voice_features(n_samples=1000, seed=42)
    save_as_npy(train_features, train_labels,
                "train_features.npy", "train_labels.npy")
    save_as_json(train_features[:10], train_labels[:10],
                 "train_sample.json")  # Save small sample as JSON

    # Generate test dataset
    print("\nGenerating test dataset...")
    test_features, test_labels = generate_voice_features(n_samples=200, seed=123)
    save_as_npy(test_features, test_labels,
                "test_features.npy", "test_labels.npy")
    save_as_json(test_features[:10], test_labels[:10],
                 "test_sample.json")  # Save small sample as JSON

    print("\nDataset generation complete!")
    print(f"Training set: {len(train_features)} samples")
    print(f"Test set: {len(test_features)} samples")
    print(f"Voice samples in training: {train_labels.sum()}")
    print(f"Non-voice samples in training: {len(train_labels) - train_labels.sum()}")
