"""
Complete training script for OpenVoiceDetect model.

This script demonstrates:
- Loading datasets from the datasets/ directory
- Training with mini-batches
- Validation during training
- Saving the best model
- Evaluation with detailed metrics

Usage:
    python train.py --epochs 50 --batch-size 32 --lr 0.001
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import OpenVoiceDetect
import os


class VoiceDataset(Dataset):
    """Custom dataset for voice detection."""

    def __init__(self, features_file, labels_file):
        self.features = np.load(features_file)
        self.labels = np.load(labels_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        # Forward pass
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train OpenVoiceDetect model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--save-path', type=str, default='trained_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--data-dir', type=str, default='datasets',
                       help='Directory containing dataset files')
    args = parser.parse_args()

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check if dataset files exist
    train_features_path = os.path.join(args.data_dir, 'train_features.npy')
    train_labels_path = os.path.join(args.data_dir, 'train_labels.npy')
    test_features_path = os.path.join(args.data_dir, 'test_features.npy')
    test_labels_path = os.path.join(args.data_dir, 'test_labels.npy')

    if not all(os.path.exists(p) for p in [train_features_path, train_labels_path,
                                            test_features_path, test_labels_path]):
        print(f"\nDataset files not found in {args.data_dir}/")
        print("Please run: cd datasets && python generate_dataset.py")
        return

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = VoiceDataset(train_features_path, train_labels_path)
    test_dataset = VoiceDataset(test_features_path, test_labels_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Initialize model
    model = OpenVoiceDetect(input_size=128, hidden_size=args.hidden_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 70)

    best_accuracy = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                           optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}%")

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
            }, args.save_path)

    print("-" * 70)
    print(f"\nTraining complete!")
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {args.save_path}")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Load best model
    checkpoint = torch.load(args.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Get detailed metrics
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            predictions = model.predict(batch_features)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean()
    voice_correct = ((all_predictions == 1) & (all_labels == 1)).sum()
    voice_total = (all_labels == 1).sum()
    no_voice_correct = ((all_predictions == 0) & (all_labels == 0)).sum()
    no_voice_total = (all_labels == 0).sum()

    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"\nClass Performance:")
    print(f"  No Voice: {no_voice_correct}/{no_voice_total} ({no_voice_correct/no_voice_total:.2%})")
    print(f"  Voice:    {voice_correct}/{voice_total} ({voice_correct/voice_total:.2%})")

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"              No Voice  Voice")
    print(f"Actual No Voice  {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"       Voice     {cm[1][0]:4d}    {cm[1][1]:4d}")


if __name__ == "__main__":
    main()
