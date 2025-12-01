#!/usr/bin/env python3
"""
Quick start example for Open Voice Detect model.
Minimal code to load and run inference.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path to import model
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import OpenVoiceDetect


# Load model
model = OpenVoiceDetect(input_size=128, hidden_size=64)
checkpoint = torch.load('../pytorch_model.bin', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create sample input (128-dimensional features)
# In practice, extract these from audio using librosa or torchaudio
audio_features = torch.randn(1, 128)

# Get prediction
prediction = model.predict(audio_features)
voice_detected = prediction.item() == 1

print(f"Voice detected: {voice_detected}")

# Get probabilities for more detail
with torch.no_grad():
    logits = model(audio_features)
    probs = torch.softmax(logits, dim=1)
    print(f"Confidence: {probs[0][prediction].item():.2%}")
    print(f"Probabilities - No voice: {probs[0][0].item():.2%}, Voice: {probs[0][1].item():.2%}")
