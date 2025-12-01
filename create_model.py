#!/usr/bin/env python3
"""
Script to create and save the open-voice-detect model for Hugging Face.
"""

import torch
import json
from model import OpenVoiceDetect


def create_and_save_model():
    """Create a simple voice detection model and save it."""

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Initialize model
    model = OpenVoiceDetect(
        input_size=config['input_size'],
        hidden_size=config['hidden_size']
    )

    # Set to eval mode
    model.eval()

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, 'pytorch_model.bin')

    print("Model saved successfully to pytorch_model.bin")

    # Test the model with dummy input
    dummy_input = torch.randn(1, config['input_size'])
    output = model(dummy_input)
    print(f"\nModel test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")

    prediction = model.predict(dummy_input)
    print(f"Prediction: {prediction.item()} ({'voice' if prediction.item() == 1 else 'no voice'})")


if __name__ == "__main__":
    create_and_save_model()
