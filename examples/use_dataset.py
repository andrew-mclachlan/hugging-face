"""
Quick example: Using the datasets with the model

This script demonstrates loading datasets and making predictions.

Usage:
    python examples/use_dataset.py
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import OpenVoiceDetect


def main():
    print("=" * 70)
    print("OpenVoiceDetect - Dataset Usage Example")
    print("=" * 70)

    # Load model
    print("\n1. Loading pre-trained model...")
    model = OpenVoiceDetect(input_size=128, hidden_size=64)
    checkpoint = torch.load('pytorch_model.bin')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("   Model loaded successfully!")

    # Load dataset
    print("\n2. Loading test dataset...")
    try:
        test_features = np.load('datasets/test_features.npy')
        test_labels = np.load('datasets/test_labels.npy')
        print(f"   Loaded {len(test_features)} test samples")
    except FileNotFoundError:
        print("   Dataset not found. Please generate it first:")
        print("   cd datasets && python generate_dataset.py")
        return

    # Make predictions on first 10 samples
    print("\n3. Making predictions on first 10 samples...")
    print("-" * 70)
    print("Sample | Predicted | Actual | Result")
    print("-" * 70)

    sample_features = torch.FloatTensor(test_features[:10])
    predictions = model.predict(sample_features)

    correct = 0
    for i in range(10):
        pred = predictions[i].item()
        actual = test_labels[i]
        result = "✓ CORRECT" if pred == actual else "✗ WRONG"
        if pred == actual:
            correct += 1

        pred_label = "Voice" if pred == 1 else "No Voice"
        actual_label = "Voice" if actual == 1 else "No Voice"

        print(f"  {i+1:2d}   | {pred_label:9s} | {actual_label:9s} | {result}")

    print("-" * 70)
    print(f"Accuracy on 10 samples: {correct}/10 ({correct*10}%)")

    # Get prediction probabilities
    print("\n4. Prediction probabilities for first sample:")
    with torch.no_grad():
        logits = model(sample_features[0].unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)

    print(f"   No Voice confidence: {probabilities[0][0]:.1%}")
    print(f"   Voice confidence:    {probabilities[0][1]:.1%}")

    # Evaluate on full test set
    print("\n5. Evaluating on full test set...")
    test_tensor = torch.FloatTensor(test_features)
    all_predictions = model.predict(test_tensor)
    accuracy = (all_predictions.numpy() == test_labels).mean()

    print(f"   Test Accuracy: {accuracy:.2%}")

    # Class-wise accuracy
    voice_mask = test_labels == 1
    no_voice_mask = test_labels == 0

    voice_acc = (all_predictions.numpy()[voice_mask] == test_labels[voice_mask]).mean()
    no_voice_acc = (all_predictions.numpy()[no_voice_mask] == test_labels[no_voice_mask]).mean()

    print(f"   No Voice accuracy: {no_voice_acc:.2%}")
    print(f"   Voice accuracy:    {voice_acc:.2%}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("- Run 'python train.py' to train the model on the datasets")
    print("- See 'datasets/README.md' for more examples")


if __name__ == "__main__":
    main()
