import torch
import torch.nn as nn


class OpenVoiceDetect(nn.Module):
    """
    A simple voice detection model that classifies audio features
    as containing voice or not.
    """

    def __init__(self, input_size=128, hidden_size=64):
        super(OpenVoiceDetect, self).__init__()
        self.config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_classes': 2  # Binary: voice or no voice
        }

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, self.config['num_classes'])
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)

    def predict(self, x):
        """
        Predict voice presence

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Predictions (0 or 1) indicating no voice or voice presence
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
