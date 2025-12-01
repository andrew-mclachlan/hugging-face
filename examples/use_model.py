#!/usr/bin/env python3
"""
Script to use the Open Voice Detect model for voice activity detection.
This demonstrates how to load and use the model with audio features.
"""

from pathlib import Path
import sys
import torch
import numpy as np

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import OpenVoiceDetect


def load_model(model_path='../pytorch_model.bin', device='cpu'):
    """
    Load the Open Voice Detect model from checkpoint.

    Args:
        model_path: Path to the model weights file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded model in evaluation mode
    """
    # Initialize model with default architecture
    model = OpenVoiceDetect(input_size=128, hidden_size=64)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    model.eval()
    model.to(device)

    print(f"Model loaded successfully on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def extract_audio_features(audio_data, sample_rate=16000):
    """
    Placeholder function for audio feature extraction.

    In a real application, you would:
    1. Load audio file using librosa or torchaudio
    2. Extract MFCCs or mel-spectrogram features
    3. Return 128-dimensional feature vector

    Args:
        audio_data: Audio waveform or path to audio file
        sample_rate: Sample rate of audio

    Returns:
        128-dimensional feature vector
    """
    # For demonstration, return random features
    # Replace this with actual feature extraction
    features = np.random.randn(128).astype(np.float32)

    print("NOTE: Using random features for demonstration.")
    print("In production, extract real features using librosa or torchaudio.")

    return features


def predict_voice_activity(model, audio_features, device='cpu'):
    """
    Predict whether voice is present in the audio features.

    Args:
        model: Trained OpenVoiceDetect model
        audio_features: 128-dimensional feature vector or batch of features
        device: Device to run inference on

    Returns:
        Dictionary with prediction results
    """
    # Convert to tensor if numpy array
    if isinstance(audio_features, np.ndarray):
        features_tensor = torch.from_numpy(audio_features)
    else:
        features_tensor = audio_features

    # Add batch dimension if single sample
    if features_tensor.dim() == 1:
        features_tensor = features_tensor.unsqueeze(0)

    features_tensor = features_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        logits = model(features_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

    results = {
        'prediction': predictions.cpu().numpy(),
        'voice_detected': bool(predictions[0].item() == 1),
        'confidence': probabilities[0][predictions[0]].item(),
        'probabilities': {
            'no_voice': probabilities[0][0].item(),
            'voice': probabilities[0][1].item()
        }
    }

    return results


def main():
    """
    Main function demonstrating model usage.
    """
    print("=" * 60)
    print("Open Voice Detect - Voice Activity Detection Demo")
    print("=" * 60)
    print()

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()

    # Load the model
    model = load_model(model_path='../pytorch_model.bin', device=device)
    print()

    # Example 1: Single audio sample
    print("Example 1: Single Audio Sample")
    print("-" * 60)
    audio_features = extract_audio_features(None)
    results = predict_voice_activity(model, audio_features, device=device)

    print(f"Voice detected: {results['voice_detected']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"Probabilities:")
    print(f"  - No voice: {results['probabilities']['no_voice']:.2%}")
    print(f"  - Voice:    {results['probabilities']['voice']:.2%}")
    print()

    # Example 2: Batch processing
    print("Example 2: Batch Processing (5 samples)")
    print("-" * 60)
    batch_size = 5
    batch_features = np.random.randn(batch_size, 128).astype(np.float32)

    batch_tensor = torch.from_numpy(batch_features).to(device)

    with torch.no_grad():
        logits = model(batch_tensor)
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)

    for i in range(batch_size):
        voice_present = predictions[i].item() == 1
        confidence = probabilities[i][predictions[i]].item()
        print(f"Sample {i+1}: Voice {'detected' if voice_present else 'not detected'} "
              f"(confidence: {confidence:.2%})")
    print()

    # Example 3: Real-world usage pattern
    print("Example 3: Real-world Usage Pattern")
    print("-" * 60)
    print("To use with actual audio files:")
    print()
    print("1. Install audio processing library:")
    print("   pip install librosa soundfile")
    print()
    print("2. Load and process audio:")
    print("   import librosa")
    print("   audio, sr = librosa.load('audio.wav', sr=16000)")
    print("   mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)")
    print("   features = np.mean(mfcc, axis=1)  # Average over time")
    print()
    print("3. Make prediction:")
    print("   result = predict_voice_activity(model, features)")
    print("   print(f'Voice detected: {result[\"voice_detected\"]}')")
    print()

    print("=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
