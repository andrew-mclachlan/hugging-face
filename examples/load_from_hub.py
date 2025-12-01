#!/usr/bin/env python3
"""
Example showing how to load the model from Hugging Face Hub.

Note: This requires the model to be uploaded to Hugging Face Hub first.
To upload, use: huggingface-cli upload <your-username>/open-voice-detect
"""

import torch
from huggingface_hub import hf_hub_download
import sys
from pathlib import Path

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import OpenVoiceDetect


def load_from_huggingface(repo_id, filename="pytorch_model.bin", device="cpu"):
    """
    Load model from Hugging Face Hub.

    Args:
        repo_id: Repository ID (e.g., "username/open-voice-detect")
        filename: Model file name
        device: Device to load model on

    Returns:
        Loaded model
    """
    print(f"Downloading model from {repo_id}...")

    # Download model file from hub
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir="./hf_cache"
    )

    print(f"Model downloaded to: {model_path}")

    # Initialize and load model
    model = OpenVoiceDetect(input_size=128, hidden_size=64)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    print("Model loaded successfully!")
    return model


if __name__ == "__main__":
    # Example usage - replace with your actual repo ID
    REPO_ID = "adi/open-voice-detect"

    print("=" * 60)
    print("Loading Open Voice Detect from Hugging Face Hub")
    print("=" * 60)
    print()

    try:
        model = load_from_huggingface(REPO_ID)

        # Test with sample data
        audio_features = torch.randn(1, 128)
        prediction = model.predict(audio_features)

        print()
        print(f"Test prediction: Voice {'detected' if prediction.item() == 1 else 'not detected'}")
        print()
        print("Note: To upload your model to Hugging Face Hub:")
        print("1. Install: pip install huggingface_hub")
        print("2. Login: huggingface-cli login")
        print("3. Upload: huggingface-cli upload <username>/open-voice-detect .")
        print()

    except Exception as e:
        print(f"Error: {e}")
        print()
        print("Make sure to:")
        print("1. Replace REPO_ID with your actual repository ID")
        print("2. Upload the model to Hugging Face Hub first")
        print("3. Install huggingface_hub: pip install huggingface_hub")
